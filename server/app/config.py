"""Runtime configuration, sourced from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _csv(name: str) -> list[str] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # HuggingFace / NeMo model id to load.
    model_name: str = os.environ.get("MODEL_NAME", "nvidia/canary-1b-v2")

    # Id advertised by ``/v1/models`` (incoming ``model`` fields are ignored).
    served_model_name: str = os.environ.get("SERVED_MODEL_NAME", "") or os.environ.get(
        "MODEL_NAME", "nvidia/canary-1b-v2"
    )

    # Model replicas; 0 = one per visible GPU (resolved at startup).
    num_workers: int = _int("NUM_WORKERS", 0)

    # Explicit devices, e.g. "cuda:0,cuda:1"; unset = round-robin over GPUs.
    devices: list[str] | None = field(default_factory=lambda: _csv("DEVICES"))

    # auto = bf16 on Ampere+ (compute capability >= 8.0), else fp32.
    # bf16 is ~3x faster but degrades transducer models - see docs/benchmarks.md.
    torch_dtype: str = os.environ.get("TORCH_DTYPE", "auto").strip().lower()

    # high = TF32 tensor cores (byte-identical, ~1.9x); medium = bf16 matmuls.
    matmul_precision: str = os.environ.get("MATMUL_PRECISION", "high").strip().lower()

    # Largest upload accepted, in megabytes.
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 200)

    # Defaults for models that take language arguments (Canary). Empty = the
    # model default en->en, which translates rather than transcribes.
    default_source_lang: str = os.environ.get("DEFAULT_SOURCE_LANG", "").strip().lower()
    default_target_lang: str = os.environ.get("DEFAULT_TARGET_LANG", "").strip().lower()

    # Local ``.nemo`` to load instead of MODEL_NAME (which becomes a label).
    model_path: str = os.environ.get("MODEL_PATH", "").strip()

    # Local-attention context window used for long-form transcription.
    longform_context: int = _int("LONGFORM_CONTEXT", 256)
    # Audio longer than this is transcribed with local-attention windowing
    # automatically. Whole-file full-context attention is quadratic and, past
    # ~30 s, its CUDA allocations are retained for the life of the process (a
    # 9-minute upload can pin >9 GiB); chunked local attention is bounded and
    # released. 0 disables the auto switch.
    longform_auto_seconds: float = _float("LONGFORM_AUTO_SECONDS", 30.0)

    # Built frontend to serve; empty = no static serving (Vite handles the UI).
    static_dir: str = os.environ.get("STATIC_DIR", "")

    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = _int("PORT", 8000)

    # --- Batching scheduler -------------------------------------------------
    # How long to wait for more requests before decoding a batch (0 = no batch).
    batch_window_ms: int = _int("BATCH_WINDOW_MS", 20)
    # Max requests decoded in one model call.
    max_batch_size: int = _int("MAX_BATCH_SIZE", 8)
    # Max total audio (s) per batch; keeps VRAM bounded.
    max_batch_audio_seconds: float = _float("MAX_BATCH_AUDIO_SECONDS", 240.0)
    # Longer requests run alone (NeMo chunks them and they would pad neighbours).
    max_batch_item_seconds: float = _float("MAX_BATCH_ITEM_SECONDS", 30.0)
    # Up to this length, audio goes to the model as an array instead of a temp
    # WAV; longer audio needs the file path for NeMo's chunked decoding.
    inline_audio_max_seconds: float = _float("INLINE_AUDIO_MAX_SECONDS", 30.0)
    # Max longest/shortest ratio in a batch: padding to a much longer neighbour
    # can change a clip's tail. 0 disables the check.
    batch_duration_ratio: float = _float("BATCH_DURATION_RATIO", 2.0)

    # --- Admission control --------------------------------------------------
    # Queued requests before 503; 0 = unlimited.
    max_queue_size: int = _int("MAX_QUEUE_SIZE", 32)
    # Max time a request may wait in the queue.
    queue_timeout_s: float = _float("QUEUE_TIMEOUT_S", 120.0)
    # ``Retry-After`` value sent with 503.
    retry_after_s: int = _int("RETRY_AFTER_S", 2)

    # --- Live streaming limits ---------------------------------------------
    # Concurrent WebSocket sessions; 0 = 8 in bf16/fp16, 4 in fp32.
    max_concurrent_streams: int = _int("MAX_CONCURRENT_STREAMS", 0)
    # Unprocessed audio a client may queue before the session is closed.
    stream_max_queue_seconds: float = _float("STREAM_MAX_QUEUE_SECONDS", 120.0)
    # Hard limit on one WebSocket session.
    stream_max_session_seconds: float = _float("STREAM_MAX_SESSION_SECONDS", 3600.0)

    # --- Observability ------------------------------------------------------
    # Expose Prometheus metrics on GET /metrics.
    metrics_enabled: bool = _bool("METRICS_ENABLED", True)

    # --- Streaming transcription ---
    # Audio needed before the first inference.
    stream_min_duration: float = _float("STREAM_MIN_DURATION", 1.0)
    # New audio needed between inference runs.
    stream_retranscribe_interval: float = _float("STREAM_RETRANSCRIBE_INTERVAL", 0.5)
    # Agreed words required for a commit.
    stream_stable_words: int = _int("STREAM_STABLE_WORDS", 1)
    # Consecutive agreeing hypotheses required for a commit.
    stream_stable_iters: int = _int("STREAM_STABLE_ITERS", 2)
    # Buffer cap: forces a trim, and a commit if nothing agreed.
    stream_max_duration: float = _float("STREAM_MAX_DURATION", 30.0)
    # Committed audio kept before a trim point as acoustic context.
    stream_context_duration: float = _float("STREAM_CONTEXT_DURATION", 1.5)
    # Buffer length before trimming at the last committed sentence end.
    stream_trim_duration: float = _float("STREAM_TRIM_DURATION", 10.0)
    # Words ending this close to the buffer end stay uncommitted.
    stream_holdback: float = _float("STREAM_HOLDBACK", 0.0)

    # --- SSE streaming (POST /v1/audio/transcriptions with stream=true) ---
    # Max audio (s) per SSE chunk/delta.
    sse_chunk_duration: float = _float("SSE_CHUNK_DURATION", 20.0)
    # Window before the chunk limit searched for a quiet cut point.
    sse_split_search: float = _float("SSE_SPLIT_SEARCH", 4.0)


settings = Settings()
