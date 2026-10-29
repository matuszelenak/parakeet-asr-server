"""A pool of Parakeet model replicas with a batching scheduler and admission control.

NeMo ASR models are not safe to call concurrently on the same instance, and a
single GPU serialises its own work anyway. The pool therefore loads one model
replica per GPU (configurable), pins each to a device, gives each its own
single-thread executor, and feeds them from a scheduling queue:

* **Batching.** Requests that arrive within ``BATCH_WINDOW_MS`` of each other
  and that agree on decoding options (timestamps / long-form / languages) are
  decoded in a *single* ``model.transcribe(paths, batch_size=N)`` call. The GPU
  is idle for most of a short request (fixed overhead dominates: a 20 s clip
  costs about the same as a 5 s clip), so this multiplies throughput for the
  many-short-requests pattern that HTTP clients produce. Requests are executed
  shortest-first inside a batch, which minimises padding.
* **Admission control.** The queue is bounded (``MAX_QUEUE_SIZE``) and every
  request has a deadline (``QUEUE_TIMEOUT_S``). Both are surfaced to clients as
  a 503 with ``Retry-After`` instead of letting latency grow without bound.
* **Fairness.** Queues are per decoding-option "key" and the dispatcher always
  serves the key with the oldest waiting request, so a stream of batch
  transcriptions cannot starve a live session (they use different keys).
* **Failure isolation.** If a batched call raises, every request in it is
  retried on its own, so a single undecodable file cannot fail its neighbours.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import logfire
import torch

from . import metrics
from .config import settings

log = logging.getLogger(__name__)

# Decoding options that must match for two requests to share a model call.
# ``kind`` is "file" (path on disk, enables NeMo chunking) or "array"
# (in-memory float32 samples); NeMo cannot mix the two in one call.
JobKey = tuple[str, bool, bool, str | None, str | None]


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _resolve_dtype(device: str) -> torch.dtype:
    """Map ``TORCH_DTYPE`` to a torch dtype (``auto`` = bf16 on Ampere+)."""
    name = settings.torch_dtype
    if name in ("", "auto"):
        if device.startswith("cuda"):
            try:
                major = torch.cuda.get_device_capability(device)[0]
            except Exception:  # pragma: no cover - defensive
                return torch.float32
            return torch.bfloat16 if major >= 8 else torch.float32
        return torch.float32
    dtype = _DTYPES.get(name)
    if dtype is None:
        log.warning("unknown TORCH_DTYPE=%r; using float32", name)
        return torch.float32
    return dtype


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


class PoolError(RuntimeError):
    """Base class for pool-level failures."""


class QueueFullError(PoolError):
    """Raised when the admission queue is full."""


class QueueTimeoutError(PoolError):
    """Raised when a request waited longer than ``QUEUE_TIMEOUT_S``."""


class PoolNotReadyError(PoolError):
    """Raised when the pool is shutting down or has no replicas."""


@dataclass
class Job:
    """One unit of work: a single audio input (temp WAV path or sample array)."""

    input: Any
    kind: str  # "file" | "array"
    duration: float
    timestamps: bool
    longform: bool
    source_lang: str | None
    target_lang: str | None
    future: asyncio.Future = field(repr=False)
    enqueued_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None

    @property
    def key(self) -> JobKey:
        return (self.kind, self.timestamps, self.longform, self.source_lang, self.target_lang)

    @property
    def deadline(self) -> float:
        return self.enqueued_at + settings.queue_timeout_s


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
        self.dtype = _resolve_dtype(device)

    def load(self) -> None:
        """Load the model onto this worker's device. Runs in the worker thread."""
        import nemo.collections.asr as nemo_asr

        logfire.info(
            "worker {index}: loading {model} onto {device} ({dtype}, matmul={prec})",
            index=self.index,
            model=settings.model_path or self.model_name,
            device=self.device,
            dtype=str(self.dtype).replace("torch.", ""),
            prec=settings.matmul_precision,
        )
        torch.set_float32_matmul_precision(settings.matmul_precision)
        if settings.model_path:
            # A local .nemo checkpoint (e.g. a community fine-tune) instead of a
            # HuggingFace/NGC model id.
            model = nemo_asr.models.ASRModel.restore_from(
                restore_path=settings.model_path, map_location=self.device
            )
        else:
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
        self._apply_dtype()
        logfire.info(
            "worker {index}: ready (default attention={attention}, dtype={dtype})",
            index=self.index,
            attention=self._orig_attention,
            dtype=_dtype_name(next(self.model.parameters()).dtype),
        )

    def _apply_dtype(self) -> None:
        """Cast the replica to the configured inference dtype (no-op for fp32)."""
        if self.dtype == torch.float32:
            return
        self.model.to(self.dtype)

    def _set_longform(self, enable: bool) -> bool:
        """Toggle local-attention windowing used for very long recordings.

        Returns True when the encoder's attention modules were rebuilt. NeMo
        builds the replacement modules with the checkpoint's dtype (fp32), so a
        reduced-precision replica has to be re-cast afterwards.
        """
        if not self._supports_longform:
            return False
        try:
            if enable and not self._longform_active:
                ctx = self.longform_context
                self.model.change_attention_model(
                    self_attention_model="rel_pos_local_attn", att_context_size=[ctx, ctx]
                )
                self._longform_active = True
                return True
            if not enable and self._longform_active:
                sa, ctx = self._orig_attention  # type: ignore[misc]
                self.model.change_attention_model(self_attention_model=sa, att_context_size=ctx)
                self._longform_active = False
                return True
        except Exception as exc:  # pragma: no cover - defensive
            logfire.warning(
                "worker {index}: long-form attention switch unsupported ({error}); continuing",
                index=self.index,
                error=str(exc),
            )
            self._supports_longform = False
        return False

    def transcribe(
        self,
        inputs: list[Any],
        timestamps: bool,
        longform: bool,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> list[Any]:
        """Run transcription. Executes in the worker thread (one call at a time).

        ``inputs`` is either a list of file paths or a list of float32 16 kHz
        mono numpy arrays (NeMo accepts both; they cannot be mixed).
        """
        if self._set_longform(longform):
            # Newly created attention modules are fp32; restore the replica dtype.
            self._apply_dtype()
        kwargs: dict[str, Any] = {"timestamps": timestamps, "batch_size": len(inputs)}
        if source_lang is not None:
            kwargs["source_lang"] = source_lang
        if target_lang is not None:
            kwargs["target_lang"] = target_lang
        with torch.inference_mode():
            return self.model.transcribe(inputs, **kwargs)


class ModelPool:
    """Bounded, batching, fair scheduler over one model replica per device."""

    def __init__(self) -> None:
        self._workers: list[Worker] = []
        self._loops: list[asyncio.Task] = []
        self._queues: dict[JobKey, deque[Job]] = {}
        self._pending = 0
        self._inflight = 0
        self._wakeup: asyncio.Event | None = None
        self._ready = False
        self._closing = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._workers)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def pending(self) -> int:
        return self._pending

    @property
    def inflight(self) -> int:
        return self._inflight

    def stats(self) -> dict[str, Any]:
        return {
            "workers": self.size,
            "pending": self._pending,
            "inflight": self._inflight,
            "max_queue_size": settings.max_queue_size,
            "batch_window_ms": settings.batch_window_ms,
            "max_batch_size": settings.max_batch_size,
            "dtype": _dtype_name(self._workers[0].dtype) if self._workers else "-",
            "matmul_precision": settings.matmul_precision,
        }

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

        self._ready = True
        self._closing = False
        self._wakeup = asyncio.Event()
        self._loops = [
            asyncio.create_task(self._dispatch(worker), name=f"asr-dispatch-{worker.index}")
            for worker in self._workers
        ]

        metrics.workers.set(len(self._workers))
        metrics.model_ready.set(1)
        metrics.queue_capacity.set(settings.max_queue_size)
        logfire.info(
            "model pool ready: {count} worker(s), batch window {window} ms, "
            "max batch {size}, queue limit {limit}",
            count=len(self._workers),
            window=settings.batch_window_ms,
            size=settings.max_batch_size,
            limit=settings.max_queue_size or "unlimited",
        )

    async def shutdown(self) -> None:
        self._closing = True
        metrics.model_ready.set(0)
        for task in self._loops:
            task.cancel()
        for task in self._loops:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # pragma: no cover - shutdown
                pass
        self._loops = []
        self._fail_all(PoolNotReadyError("server is shutting down"))
        for worker in self._workers:
            worker.executor.shutdown(wait=True, cancel_futures=False)
        self._workers = []
        self._ready = False

    # -- scheduling --------------------------------------------------------
    def _fail_all(self, exc: Exception) -> None:
        for queue in self._queues.values():
            while queue:
                job = queue.popleft()
                self._pending = max(0, self._pending - 1)
                if not job.future.done():
                    job.future.set_exception(exc)
        self._queues.clear()
        metrics.queue_depth.set(0)

    def _queue_for(self, key: JobKey) -> deque[Job]:
        queue = self._queues.get(key)
        if queue is None:
            queue = deque()
            self._queues[key] = queue
        return queue

    def _drop(self, job: Job) -> None:
        """Remove a still-queued job (used when its deadline passes)."""
        queue = self._queues.get(job.key)
        if queue is not None and job in queue:
            queue.remove(job)
            self._pending = max(0, self._pending - 1)
            metrics.queue_depth.set(self._pending)

    async def _next_batch(self) -> list[Job] | None:
        """Wait for work, let the batch window elapse, and cut a batch.

        Returns ``None`` while shutting down.
        """
        wakeup = self._wakeup
        assert wakeup is not None

        while True:
            if self._closing:
                return None
            # Clear before the check so a job that arrives in between is not lost.
            wakeup.clear()
            self._expire_queued()
            if self._pending > 0:
                break
            await wakeup.wait()

        window = settings.batch_window_ms
        if window > 0 and settings.max_batch_size > 1:
            await asyncio.sleep(window / 1000.0)

        # Serve the decoding-option group with the oldest waiting request so no
        # group can starve another.
        best: JobKey | None = None
        best_age = float("inf")
        for key, queue in self._queues.items():
            if not queue:
                continue
            head = queue[0]
            if head.enqueued_at < best_age:
                best, best_age = key, head.enqueued_at
        if best is None:
            return []

        queue = self._queues[best]
        batch: list[Job] = []
        audio = 0.0
        shortest = float("inf")
        ratio = settings.batch_duration_ratio
        while queue:
            head = queue[0]
            if batch and len(batch) >= settings.max_batch_size:
                break
            if batch and audio + head.duration > settings.max_batch_audio_seconds:
                break
            if head.duration > settings.max_batch_item_seconds:
                # Long request: decode it on its own.
                if not batch:
                    batch.append(queue.popleft())
                    audio += head.duration
                    shortest = min(shortest, head.duration)
                break
            if batch and ratio > 0 and head.duration > shortest * ratio:
                # Too different in length: padding one to the other's length would
                # change the shorter clip's transcript, so run it in its own batch.
                break
            batch.append(queue.popleft())
            audio += head.duration
            shortest = min(shortest, head.duration)

        self._pending = max(0, self._pending - len(batch))
        metrics.queue_depth.set(self._pending)
        return batch

    def _expire_queued(self) -> None:
        """Fail requests whose deadline passed while they were waiting."""
        now = time.monotonic()
        for key, queue in list(self._queues.items()):
            expired = [job for job in queue if job.deadline <= now]
            if not expired:
                continue
            for job in expired:
                queue.remove(job)
                self._pending = max(0, self._pending - 1)
                metrics.rejected_total.labels(reason="queue_timeout").inc()
                if not job.future.done():
                    job.future.set_exception(
                        QueueTimeoutError(
                            f"request waited longer than {settings.queue_timeout_s:.0f}s"
                        )
                    )
            metrics.queue_depth.set(self._pending)
            if not queue:
                del self._queues[key]

    async def _dispatch(self, worker: Worker) -> None:
        """Per-replica loop: cut a batch, decode it, resolve the futures."""
        while not self._closing:
            try:
                batch = await self._next_batch()
                if batch is None:
                    return
                if not batch:
                    continue
                await self._execute(worker, batch)
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                return
            except Exception:  # pragma: no cover - never kill the scheduler
                log.exception("scheduler loop error")
                await asyncio.sleep(0.05)

    async def _execute(self, worker: Worker, batch: list[Job]) -> None:
        # Shortest first: less padding, same set of results.
        ordered = sorted(batch, key=lambda j: j.duration)
        now = time.monotonic()
        for job in ordered:
            job.started_at = now
            metrics.queue_wait_seconds.observe(now - job.enqueued_at)

        self._inflight += len(ordered)
        metrics.inflight_requests.set(self._inflight)
        mode = metrics.mode_label(ordered[0].longform)
        audio_s = sum(j.duration for j in ordered)
        loop = asyncio.get_running_loop()
        started = time.monotonic()
        try:
            inputs = [j.input for j in ordered]
            head = ordered[0]
            results = await loop.run_in_executor(
                worker.executor,
                lambda: worker.transcribe(
                    inputs,
                    timestamps=head.timestamps,
                    longform=head.longform,
                    source_lang=head.source_lang,
                    target_lang=head.target_lang,
                ),
            )
            results = list(results)
            if len(results) != len(ordered):
                raise RuntimeError(
                    f"model returned {len(results)} results for {len(ordered)} inputs"
                )
            for job, result in zip(ordered, results):
                if not job.future.done():
                    job.future.set_result(result)
            metrics.inference_batches_total.labels(mode=mode).inc()
            metrics.inference_batch_size.labels(mode=mode).observe(len(ordered))
            metrics.inference_audio_seconds.labels(mode=mode).inc(audio_s)
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        except Exception as exc:
            if len(ordered) == 1:
                metrics.inference_errors_total.inc()
                logfire.exception("inference failed: {error}", error=str(exc))
                if not ordered[0].future.done():
                    ordered[0].future.set_exception(exc)
            else:
                # Isolate the failure: retry every request of the batch alone.
                metrics.split_batches_total.inc()
                logfire.warning(
                    "batched inference failed ({error}); retrying {count} request(s) "
                    "individually",
                    error=str(exc),
                    count=len(ordered),
                )
                await self._retry_individually(worker, ordered)
        finally:
            elapsed = time.monotonic() - started
            metrics.inference_seconds.labels(mode=mode).observe(elapsed)
            self._inflight = max(0, self._inflight - len(ordered))
            metrics.inflight_requests.set(self._inflight)

    async def _retry_individually(self, worker: Worker, batch: Sequence[Job]) -> None:
        loop = asyncio.get_running_loop()
        for job in batch:
            try:
                results = await loop.run_in_executor(
                    worker.executor,
                    lambda job=job: worker.transcribe(
                        [job.input],
                        timestamps=job.timestamps,
                        longform=job.longform,
                        source_lang=job.source_lang,
                        target_lang=job.target_lang,
                    ),
                )
                if not job.future.done():
                    job.future.set_result(list(results)[0])
            except Exception as exc:
                metrics.inference_errors_total.inc()
                logfire.exception("inference failed: {error}", error=str(exc))
                if not job.future.done():
                    job.future.set_exception(exc)

    # -- public API --------------------------------------------------------
    async def submit(
        self,
        audio: Any,
        *,
        duration: float,
        timestamps: bool = False,
        longform: bool = False,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> Any:
        """Queue one input (path or float32 16 kHz sample array) and await its result."""
        if not self._ready:
            raise PoolNotReadyError("model pool is not ready")

        limit = settings.max_queue_size
        if limit and self._pending >= limit:
            metrics.rejected_total.labels(reason="queue_full").inc()
            raise QueueFullError(
                f"queue is full ({self._pending} requests waiting); retry shortly"
            )

        kind = "array" if _is_array(audio) else "file"
        loop = asyncio.get_running_loop()
        job = Job(
            input=audio,
            kind=kind,
            duration=duration,
            timestamps=timestamps,
            longform=longform,
            source_lang=source_lang,
            target_lang=target_lang,
            future=loop.create_future(),
        )
        self._queue_for(job.key).append(job)
        self._pending += 1
        metrics.queue_depth.set(self._pending)
        if self._wakeup is not None:
            self._wakeup.set()

        try:
            # The deadline bites while the request is queued; once decoding has
            # started the request is guaranteed a result by the dispatcher, so a
            # very generous bound is only a safety net against scheduler bugs.
            try:
                return await asyncio.wait_for(
                    asyncio.shield(job.future), timeout=settings.queue_timeout_s
                )
            except asyncio.TimeoutError:
                if job.started_at is None:
                    metrics.rejected_total.labels(reason="queue_timeout").inc()
                    self._drop(job)
                    if job.future.done():
                        if not job.future.cancelled():
                            job.future.exception()  # mark as retrieved
                    else:
                        job.future.cancel()
                    raise QueueTimeoutError(
                        f"request waited longer than {settings.queue_timeout_s:.0f}s"
                    ) from None
                return await asyncio.wait_for(asyncio.shield(job.future), timeout=3600.0)
        finally:
            if not job.future.done():  # pragma: no cover - cancellation
                job.future.cancel()

    async def transcribe(
        self,
        audio: Sequence[Any],
        *,
        timestamps: bool = False,
        longform: bool = False,
        source_lang: str | None = None,
        target_lang: str | None = None,
        durations: Iterable[float] | None = None,
    ) -> list[Any]:
        """Queue several inputs (paths or sample arrays) and return results in order.

        Concurrent callers are batched together by the scheduler.
        """
        items = list(audio)
        durs = list(durations) if durations is not None else None
        if durs is None:
            loop = asyncio.get_running_loop()
            durs = await loop.run_in_executor(None, lambda: [_duration(i) for i in items])
        return list(
            await asyncio.gather(
                *(
                    self.submit(
                        item,
                        duration=dur,
                        timestamps=timestamps,
                        longform=longform,
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                    for item, dur in zip(items, durs)
                )
            )
        )


def _is_array(audio: Any) -> bool:
    """True for in-memory sample arrays (anything that is not a path)."""
    return not isinstance(audio, (str, bytes, os.PathLike))


def _duration(audio: Any) -> float:
    """Cheap duration lookup used when the caller does not provide one."""
    if _is_array(audio):
        try:
            return len(audio) / 16_000.0
        except TypeError:  # pragma: no cover - defensive
            return 0.0
    try:
        import soundfile as sf

        info = sf.info(str(audio))
        return float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    except Exception:  # pragma: no cover - defensive
        return 0.0


pool = ModelPool()
