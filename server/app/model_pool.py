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
import json
import os
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import logfire
import numpy as np
import torch

from .config import settings

SAMPLING_RATE = 16_000


def _write_prompt_manifest(paths: list[str], lang: str) -> str:
    """Write a temp NeMo manifest tagging each clip with its language prompt.

    Returns the manifest path; the caller is responsible for deleting it. Each
    entry carries ``lang`` (the cut's supervision language) and a ``langID``
    ``prompt_mode`` so the prompt model conditions on ``lang`` deterministically.
    """
    fd, manifest_path = tempfile.mkstemp(suffix=".json", prefix="asr_manifest_")
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        for path in paths:
            with wave.open(path, "rb") as wav:
                duration = wav.getnframes() / float(wav.getframerate())
            entry = {
                "audio_filepath": path,
                "duration": duration,
                "text": "",
                "lang": lang,
                "prompt_mode": "langID",
            }
            fp.write(json.dumps(entry) + "\n")
    return manifest_path


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
        self.bufferer: Any = None  # BatchedCacheFeatureBufferer (one slot)
        self.frame_samples = 0  # audio samples consumed per streaming step
        self.drop_extra_pre_encoded = 0
        self.cache_last_channel: Any = None
        self.cache_last_time: Any = None
        self.cache_last_channel_len: Any = None
        self.previous_hypotheses: Any = None
        self.pred_out_stream: Any = None
        self.pending = np.empty(0, dtype=np.float32)  # not-yet-stepped audio tail
        self.step_num = 0
        self.samples_seen = 0
        self._last_text = ""

    # ── worker-thread methods ────────────────────────────────────────────────
    def _start(self) -> None:
        from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import (
            BatchedCacheFeatureBufferer,
        )
        from omegaconf import OmegaConf, open_dict

        model = self.worker.model
        self.worker.set_language(self.target_lang)

        # Cache-aware streaming framing, taken from the encoder's own streaming
        # config. ``chunk_size`` / ``pre_encode_cache_size`` are per-step feature
        # frame counts (the trailing element is the steady-state value); each
        # streaming step consumes ``chunk_size`` feature frames of audio and is
        # given ``pre_encode_cache_size`` extra frames of left look-back.
        scfg = model.encoder.streaming_cfg
        chunk_frames = self._steady(scfg.chunk_size)
        pre_encode = self._steady(scfg.pre_encode_cache_size)
        self.drop_extra_pre_encoded = self._steady(scfg.drop_extra_pre_encoded)

        window_stride = float(model.cfg.preprocessor.window_stride)
        chunk_secs = chunk_frames * window_stride
        buffer_secs = (chunk_frames + pre_encode) * window_stride
        self.frame_samples = int(round(chunk_secs * SAMPLING_RATE))

        # The bufferer builds its own preprocessor from this config; force the
        # inference-time settings so features are deterministic per chunk.
        pre_cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.preprocessor, resolve=True))
        with open_dict(pre_cfg):
            pre_cfg.dither = 0.0
            pre_cfg.pad_to = 0
        self.bufferer = BatchedCacheFeatureBufferer(
            num_slots=1,
            sample_rate=SAMPLING_RATE,
            buffer_size_in_secs=buffer_secs,
            chunk_size_in_secs=chunk_secs,
            preprocessor_cfg=pre_cfg,
            device=torch.device(self.worker.device),
        )

        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)
        self.pending = np.empty(0, dtype=np.float32)

    @staticmethod
    def _steady(value: Any) -> int:
        """Return the steady-state value of a per-step streaming-cfg field.

        These fields are either a scalar or a ``[first_step, steady_state]``
        list; we drive fixed-size chunks, so the steady-state value applies to
        every step (the bufferer zero-pads the first chunk's look-back).
        """
        return int(value[-1] if isinstance(value, (list, tuple)) else value)

    def _step(self, chunk: np.ndarray, valid: int, is_last: bool) -> str:
        """Run one cache-aware streaming step over a fixed-size audio chunk.

        ``chunk`` has exactly ``frame_samples`` samples (zero-padded when the
        final chunk is short); ``valid`` is the count of real samples.
        """
        from nemo.collections.asr.inference.streaming.framing.request import Frame

        model = self.worker.model
        frame = Frame(
            samples=torch.from_numpy(chunk),
            stream_id=0,
            is_first=(self.step_num == 0),
            is_last=is_last,
            length=valid,
        )
        # Roll the chunk into the rolling feature buffer (mel features incl. the
        # pre-encode look-back) for this single stream.
        feature_buffers, right_paddings = self.bufferer.update([frame])
        feat = feature_buffers[0].unsqueeze(0).to(self.worker.compute_dtype)
        feat_len = feat.shape[-1] - int(right_paddings[0])
        length = torch.tensor([feat_len], device=feat.device)

        with torch.inference_mode():
            (
                self.pred_out_stream,
                transcribed_texts,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=feat,
                processed_signal_length=length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=is_last,
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.pred_out_stream,
                drop_extra_pre_encoded=self.drop_extra_pre_encoded,
                return_transcription=True,
            )
        self.step_num += 1
        return _extract_text(transcribed_texts)

    def _feed(self, samples) -> str:
        samples = np.asarray(samples, dtype=np.float32)
        self.samples_seen += len(samples)
        self.pending = np.concatenate((self.pending, samples))
        text = self._last_text
        # Emit one step per full chunk; keep the trailing remainder for later.
        while len(self.pending) >= self.frame_samples:
            chunk = self.pending[: self.frame_samples]
            self.pending = self.pending[self.frame_samples :]
            text = self._step(chunk, valid=self.frame_samples, is_last=False)
        self._last_text = text
        return text

    def _finish(self) -> str:
        # Flush the tail (plus the encoder's look-ahead) in one final step with
        # keep_all_outputs=True so the trailing frames are not dropped. A full
        # frame of silence is used when no audio remains.
        valid = min(len(self.pending), self.frame_samples)
        chunk = np.zeros(self.frame_samples, dtype=np.float32)
        if valid > 0:
            chunk[:valid] = self.pending[:valid]
        self.pending = np.empty(0, dtype=np.float32)
        text = self._step(chunk, valid=valid or self.frame_samples, is_last=True)
        self._last_text = text
        return text

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
        if not self._supports_prompt:
            with torch.inference_mode():
                return self.model.transcribe(
                    paths, timestamps=timestamps, batch_size=len(paths)
                )

        # The prompt model picks its language prompt per input cut from the
        # manifest's ``lang`` field. Neither set_inference_prompt nor
        # transcribe()'s ``target_lang`` kwarg drives the offline dataloader
        # (its ``default_lang`` is a dead key in NeMo), and plain dict inputs are
        # rejected — but a single ``.json`` path is read as a manifest, so we
        # hand transcribe() one we build ourselves:
        #   lang        -> cut supervision language ('auto' is a valid prompt key
        #                  for language-agnostic decoding)
        #   prompt_mode -> 'langID' forces that language deterministically rather
        #                  than the dataset's default 'unified' mode, which
        #                  randomly substitutes the 'auto' prompt.
        manifest_path = _write_prompt_manifest(paths, target_lang)
        try:
            with torch.inference_mode():
                return self.model.transcribe(
                    [manifest_path], timestamps=timestamps, batch_size=len(paths)
                )
        finally:
            try:
                os.remove(manifest_path)
            except OSError:
                pass


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
