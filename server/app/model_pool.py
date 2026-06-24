"""A pool of Nemotron streaming-ASR model replicas.

NeMo ASR models are not safe to call concurrently on the same instance, and a
single GPU serialises its own work anyway. To serve several requests in
parallel we load one model replica per GPU (configurable), pin each to a device,
and give each its own single-thread executor.

Two access patterns are supported:

* **Offline** (``pool.transcribe``): a worker is checked out for a single
  ``model.transcribe`` call and returned immediately.
* **Streaming** (``pool.stream_session``): a worker is held for the entire
  duration of a WebSocket session.  The session drives NeMo's native
  cache-aware streaming — feeding audio chunk by chunk through
  ``conformer_stream_step`` while carrying the encoder cache and the running
  RNN-T hypothesis forward between steps.

Because the streaming cache state and the decoder's language prompt are mutable
model state, every model touch for a session runs on that worker's single
thread, and a worker only ever serves one session at a time.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import logfire
import torch

from .config import settings

SAMPLING_RATE = 16_000


def _extract_text(transcribed_texts: Any) -> str:
    """Pull the running transcript out of a ``conformer_stream_step`` result.

    For RNN-T models ``transcribed_texts`` is a list of ``Hypothesis`` objects;
    for CTC it is a list of strings. We only ever stream a single audio stream,
    so we take element 0.
    """
    if not transcribed_texts:
        return ""
    first = transcribed_texts[0]
    text = getattr(first, "text", first)
    return (text or "").strip()


class StreamingSession:
    """Per-session cache-aware streaming state, bound to one worker.

    All methods that touch the model run on the worker's single thread via its
    executor, so the mutable cache/hypothesis state is never accessed
    concurrently.
    """

    def __init__(self, worker: "Worker", target_lang: str) -> None:
        self.worker = worker
        self.target_lang = target_lang

        # Set in _start (all live on the worker's device / thread).
        self.buffer: Any = None
        self.streaming_cfg: Any = None
        self.cache_last_channel: Any = None
        self.cache_last_time: Any = None
        self.cache_last_channel_len: Any = None
        self.previous_hypotheses: Any = None
        self.pred_out_stream: Any = None
        self.step_num = 0
        self.samples_seen = 0
        self._last_text = ""

    # ── worker-thread methods ────────────────────────────────────────────────
    def _start(self) -> None:
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        model = self.worker.model
        self.worker.set_language(self.target_lang)
        self.buffer = CacheAwareStreamingAudioBuffer(
            model=model,
            online_normalization=self.worker.online_normalization,
            pad_and_drop_preencoded=False,
        )
        self.streaming_cfg = model.encoder.streaming_cfg
        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)

    def _current_chunk_size(self) -> int:
        """Chunk size (in feature frames) the next streaming step will consume."""
        cs = self.streaming_cfg.chunk_size
        if isinstance(cs, list):
            return cs[0] if self.buffer.buffer_idx == 0 else cs[1]
        return cs

    def _drain(self, final: bool) -> str:
        """Run streaming steps for every full chunk currently buffered.

        Mid-stream (``final=False``) only complete chunks are processed; a
        trailing partial chunk is left in the buffer until more audio arrives.
        At ``final=True`` the remaining tail is flushed and the last step keeps
        all encoder outputs (so the look-ahead frames are not dropped).
        """
        model = self.worker.model
        text = self._last_text
        while self.buffer.buffer is not None:
            idx = self.buffer.buffer_idx
            remaining = self.buffer.buffer.size(-1) - idx
            if remaining <= 0:
                break
            if not final and remaining < self._current_chunk_size():
                break

            try:
                chunk_audio, chunk_lengths = next(iter(self.buffer))
            except StopIteration:
                break

            chunk_audio = chunk_audio.to(self.worker.compute_dtype)
            keep_all = final and self.buffer.is_buffer_empty()
            drop = (
                0
                if self.step_num == 0
                else self.streaming_cfg.drop_extra_pre_encoded
            )
            with torch.inference_mode():
                (
                    self.pred_out_stream,
                    transcribed_texts,
                    self.cache_last_channel,
                    self.cache_last_time,
                    self.cache_last_channel_len,
                    self.previous_hypotheses,
                ) = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=self.cache_last_channel,
                    cache_last_time=self.cache_last_time,
                    cache_last_channel_len=self.cache_last_channel_len,
                    keep_all_outputs=keep_all,
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=self.pred_out_stream,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            self.step_num += 1
            text = _extract_text(transcribed_texts)

        self._last_text = text
        return text

    def _feed(self, samples) -> str:
        self.samples_seen += len(samples)
        # append_audio preprocesses to mel features and concatenates onto the
        # rolling feature buffer (on the model device).
        self.buffer.append_audio(samples)
        return self._drain(final=False)

    def _finish(self) -> str:
        return self._drain(final=True)

    # ── async wrappers (dispatch to the worker thread) ───────────────────────
    async def _run(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.worker.executor, fn, *args)

    async def start(self) -> None:
        await self._run(self._start)

    async def add_audio(self, samples) -> str:
        return await self._run(self._feed, samples)

    async def finalize(self) -> str:
        return await self._run(self._finish)

    @property
    def elapsed_seconds(self) -> float:
        return self.samples_seen / SAMPLING_RATE


class Worker:
    """A single model replica bound to one device and one worker thread."""

    def __init__(self, index: int, device: str, model_name: str):
        self.index = index
        self.device = device
        self.model_name = model_name
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"asr-{index}")
        self.model: Any = None
        self.compute_dtype = torch.float32  # cache-aware models require float32
        self.online_normalization = False
        self._supports_prompt = False
        self._current_lang: str | None = None

    def load(self) -> None:
        """Load and configure the model. Runs in the worker thread."""
        import nemo.collections.asr as nemo_asr

        logfire.info(
            "worker {index}: loading {model} onto {device}",
            index=self.index,
            model=self.model_name,
            device=self.device,
        )
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name, map_location=self.device
        )
        model = model.to(device=self.device, dtype=self.compute_dtype)
        model.eval()

        # Select the streaming look-ahead (chunk/shift sizes are derived from it).
        if hasattr(model.encoder, "set_default_att_context_size"):
            try:
                model.encoder.set_default_att_context_size(
                    att_context_size=list(settings.att_context_size)
                )
            except Exception as exc:  # pragma: no cover - defensive
                logfire.warning(
                    "worker {index}: could not set att_context_size={acs} ({error})",
                    index=self.index,
                    acs=settings.att_context_size,
                    error=str(exc),
                )

        self._configure_decoding(model)

        # Per-chunk feature normalization is only meaningful when the model
        # actually normalizes its input features.
        try:
            normalize = model.cfg.preprocessor.normalize
        except Exception:  # pragma: no cover - defensive
            normalize = None
        self.online_normalization = (
            settings.online_normalization and normalize in ("per_feature", "all_feature")
        )

        self._supports_prompt = hasattr(model, "set_inference_prompt")

        self.model = model
        logfire.info(
            "worker {index}: ready (att_context={acs}, online_norm={on}, prompt={pr})",
            index=self.index,
            acs=getattr(model.encoder, "att_context_size", None),
            on=self.online_normalization,
            pr=self._supports_prompt,
        )

    def _configure_decoding(self, model: Any) -> None:
        """Force streaming-compatible RNN-T decoding (fused batch disabled)."""
        if not hasattr(model, "change_decoding_strategy"):
            return
        try:
            from omegaconf import OmegaConf, open_dict

            decoding_cfg = getattr(model.cfg, "decoding", None)
            if decoding_cfg is None:
                return
            decoding_cfg = OmegaConf.create(OmegaConf.to_container(decoding_cfg, resolve=True))
            with open_dict(decoding_cfg):
                # Fused batched decoding is incompatible with streaming partial
                # hypotheses; greedy_batch is the standard streaming strategy.
                decoding_cfg.fused_batch_size = -1
                if "strategy" in decoding_cfg:
                    decoding_cfg.strategy = "greedy_batch"
            if hasattr(model, "cur_decoder"):
                model.change_decoding_strategy(decoding_cfg, decoder_type="rnnt")
            else:
                model.change_decoding_strategy(decoding_cfg)
        except Exception as exc:  # pragma: no cover - defensive
            logfire.warning(
                "worker {index}: could not adjust decoding strategy ({error}); "
                "using model defaults",
                index=self.index,
                error=str(exc),
            )

    def set_language(self, lang: str) -> None:
        """Prompt the decoder for ``lang`` (a locale or 'auto'). Worker thread."""
        if not self._supports_prompt or lang == self._current_lang:
            return
        try:
            self.model.set_inference_prompt(lang)
            decoding = getattr(self.model, "decoding", None)
            if decoding is not None and hasattr(decoding, "set_strip_lang_tags"):
                decoding.set_strip_lang_tags(settings.strip_lang_tags)
            self._current_lang = lang
        except Exception as exc:  # pragma: no cover - defensive
            logfire.warning(
                "worker {index}: could not set language prompt to {lang} ({error})",
                index=self.index,
                lang=lang,
                error=str(exc),
            )

    def transcribe(self, paths: list[str], timestamps: bool, target_lang: str) -> list[Any]:
        """Offline transcription. Executes in the worker thread (one at a time)."""
        self.set_language(target_lang)
        with torch.inference_mode():
            return self.model.transcribe(
                paths, timestamps=timestamps, batch_size=len(paths)
            )


class ModelPool:
    def __init__(self) -> None:
        self._workers: list[Worker] = []
        self._idle: asyncio.Queue[Worker] = asyncio.Queue()
        self._ready = False

    @property
    def size(self) -> int:
        return len(self._workers)

    @property
    def ready(self) -> bool:
        return self._ready

    def _resolve_devices(self) -> list[str]:
        if settings.devices:
            return settings.devices

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logfire.warning("no CUDA devices visible; falling back to CPU (slow)")
            workers = settings.num_workers or 1
            return ["cpu"] * workers

        workers = settings.num_workers or gpu_count
        return [f"cuda:{i % gpu_count}" for i in range(workers)]

    async def startup(self) -> None:
        devices = self._resolve_devices()
        self._workers = [
            Worker(i, dev, settings.model_name) for i, dev in enumerate(devices)
        ]
        logfire.info(
            "loading {count} worker(s): {devices}",
            count=len(self._workers),
            devices=devices,
        )

        loop = asyncio.get_running_loop()
        await asyncio.gather(
            *(loop.run_in_executor(w.executor, w.load) for w in self._workers)
        )

        self._idle = asyncio.Queue()
        for worker in self._workers:
            self._idle.put_nowait(worker)
        self._ready = True
        logfire.info("model pool ready with {count} worker(s)", count=len(self._workers))

    async def shutdown(self) -> None:
        for worker in self._workers:
            worker.executor.shutdown(wait=True, cancel_futures=False)
        self._workers = []
        self._ready = False

    async def transcribe(
        self,
        paths: list[str],
        *,
        timestamps: bool = False,
        target_lang: str | None = None,
    ) -> list[Any]:
        if not self._ready:
            raise RuntimeError("model pool is not ready")

        lang = target_lang or settings.target_lang
        with logfire.span(
            "transcribe", files=len(paths), timestamps=timestamps, target_lang=lang
        ):
            worker = await self._idle.get()
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    worker.executor,
                    lambda: worker.transcribe(paths, timestamps, lang),
                )
            finally:
                self._idle.put_nowait(worker)

    @asynccontextmanager
    async def stream_session(
        self, target_lang: str | None = None
    ) -> AsyncIterator[StreamingSession]:
        """Check out a worker for the lifetime of a streaming session.

        Yields a started :class:`StreamingSession`. The worker is returned to
        the pool when the context exits, so the number of concurrent streaming
        sessions is capped at the pool size.
        """
        if not self._ready:
            raise RuntimeError("model pool is not ready")

        lang = target_lang or settings.target_lang
        worker = await self._idle.get()
        session = StreamingSession(worker, lang)
        try:
            await session.start()
            yield session
        finally:
            self._idle.put_nowait(worker)


pool = ModelPool()
