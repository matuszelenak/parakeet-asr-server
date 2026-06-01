"""A pool of Parakeet model replicas for parallel transcription.

NeMo ASR models are not safe to call concurrently on the same instance, and a
single GPU serialises its own work anyway. To serve several requests in
parallel we load one model replica per GPU (configurable), pin each to a device,
and give each its own single-thread executor. Requests check out an idle worker
from an asyncio queue, so up to `num_workers` transcriptions run at once while
each replica processes one request at a time.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import logfire
import torch

from .config import settings


class Worker:
    """A single model replica bound to one device and one worker thread."""

    def __init__(self, index: int, device: str, model_name: str, longform_context: int):
        self.index = index
        self.device = device
        self.model_name = model_name
        self.longform_context = longform_context
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"asr-{index}")
        self.model: Any = None
        self._orig_attention: tuple[str, list[int]] | None = None
        self._longform_active = False
        self._supports_longform = True

    def load(self) -> None:
        """Load the model onto this worker's device. Runs in the worker thread."""
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
        model.to(self.device)
        model.eval()

        # Remember the default attention config so long-form mode is reversible.
        self._supports_longform = hasattr(model, "change_attention_model")
        try:
            enc = model.cfg.encoder
            self._orig_attention = (enc.self_attention_model, list(enc.att_context_size))
        except Exception:  # pragma: no cover - defensive
            self._orig_attention = ("rel_pos", [-1, -1])

        self.model = model
        logfire.info(
            "worker {index}: ready (default attention={attention})",
            index=self.index,
            attention=self._orig_attention,
        )

    def _set_longform(self, enable: bool) -> None:
        """Toggle local-attention windowing used for very long recordings."""
        if not self._supports_longform:
            return
        try:
            if enable and not self._longform_active:
                ctx = self.longform_context
                self.model.change_attention_model(
                    self_attention_model="rel_pos_local_attn", att_context_size=[ctx, ctx]
                )
                self._longform_active = True
            elif not enable and self._longform_active:
                sa, ctx = self._orig_attention  # type: ignore[misc]
                self.model.change_attention_model(self_attention_model=sa, att_context_size=ctx)
                self._longform_active = False
        except Exception as exc:  # pragma: no cover - defensive
            logfire.warning(
                "worker {index}: long-form attention switch unsupported ({error}); continuing",
                index=self.index,
                error=str(exc),
            )
            self._supports_longform = False

    def transcribe(
        self,
        paths: list[str],
        timestamps: bool,
        longform: bool,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> list[Any]:
        """Run transcription. Executes in the worker thread (one at a time)."""
        self._set_longform(longform)
        kwargs: dict[str, Any] = {"timestamps": timestamps, "batch_size": len(paths)}
        if source_lang is not None:
            kwargs["source_lang"] = source_lang
        if target_lang is not None:
            kwargs["target_lang"] = target_lang
        with torch.inference_mode():
            return self.model.transcribe(paths, **kwargs)


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
            Worker(i, dev, settings.model_name, settings.longform_context)
            for i, dev in enumerate(devices)
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
        longform: bool = False,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> list[Any]:
        if not self._ready:
            raise RuntimeError("model pool is not ready")

        with logfire.span(
            "transcribe",
            files=len(paths),
            timestamps=timestamps,
            longform=longform,
            source_lang=source_lang,
            target_lang=target_lang,
        ):
            worker = await self._idle.get()
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    worker.executor,
                    lambda: worker.transcribe(
                        paths, timestamps, longform, source_lang, target_lang
                    ),
                )
            finally:
                self._idle.put_nowait(worker)


pool = ModelPool()
