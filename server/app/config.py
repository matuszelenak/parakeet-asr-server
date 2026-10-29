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

    # Model id reported by the OpenAI-compatible ``/v1/models`` endpoint and in
    # responses. The ``model`` field of incoming requests is accepted as-is, so
    # this only matters for clients/proxies (e.g. LiteLLM) that list models.
    served_model_name: str = os.environ.get("SERVED_MODEL_NAME", "") or os.environ.get(
        "MODEL_NAME", "nvidia/canary-1b-v2"
    )

    # Number of model replicas. Defaults to one per visible GPU (resolved at
    # startup when this is 0).
    num_workers: int = _int("NUM_WORKERS", 0)

    # Explicit device list, e.g. "cuda:0,cuda:1". When unset, devices are
    # assigned round-robin across the visible GPUs.
    devices: list[str] | None = field(default_factory=lambda: _csv("DEVICES"))

    # Inference precision for the model weights. ``auto`` picks bfloat16 on
    # Ampere-and-newer GPUs (compute capability >= 8.0) and float32 elsewhere.
    # bf16 is ~3x faster than fp32 for this model at batch/stream sizes we use
    # and left the transcript quality unchanged in our Slovak evaluation
    # (WER 11.45 % -> 11.26 % long-form, 10.97 % -> 10.79 % plain).
    torch_dtype: str = os.environ.get("TORCH_DTYPE", "auto").strip().lower()

    # fp32 matmul/conv precision: ``high`` enables TF32 tensor cores (Ampere+),
    # which is byte-identical output for this model and ~1.9x faster than the
    # default ``highest``. ``medium`` (bf16 matmuls in fp32 tensors) is faster
    # still but changes the numerics.
    matmul_precision: str = os.environ.get("MATMUL_PRECISION", "high").strip().lower()

    # Largest upload accepted, in megabytes.
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 200)

    # Default source/target language for models that take language arguments
    # (Canary). Canary defaults to en->en, which *translates* non-English audio
    # into English; a Slovak deployment wants e.g. DEFAULT_SOURCE_LANG=sk so a
    # request without language fields transcribes instead of translating.
    # Leave empty to keep the model default (en).
    default_source_lang: str = os.environ.get("DEFAULT_SOURCE_LANG", "").strip().lower()
    default_target_lang: str = os.environ.get("DEFAULT_TARGET_LANG", "").strip().lower()

    # Local ``.nemo`` checkpoint to load instead of a HuggingFace/NGC model id
    # (e.g. /cache/hf/parakeet-ultra/ultra.nemo). When set, MODEL_NAME is
    # ignored for loading and only used as the model label.
    model_path: str = os.environ.get("MODEL_PATH", "").strip()

    # Local-attention context window used for long-form transcription.
    longform_context: int = _int("LONGFORM_CONTEXT", 256)

    # Directory of a built frontend to serve as static files. Empty (the dev
    # default) disables static serving so the Vite dev server handles the UI.
    # The production image sets this to the baked-in build (e.g. /app/static).
    static_dir: str = os.environ.get("STATIC_DIR", "")

    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = _int("PORT", 8000)

    # --- Batching scheduler -------------------------------------------------
    # How long the scheduler waits for more requests before starting a batch.
    # 0 disables batching (one request per model call, the pre-scheduler
    # behaviour); larger values improve throughput at the cost of latency.
    batch_window_ms: int = _int("BATCH_WINDOW_MS", 20)
    # Maximum number of requests decoded together in one model call.
    max_batch_size: int = _int("MAX_BATCH_SIZE", 8)
    # Maximum total audio (seconds) in one batch. Keeps VRAM bounded.
    max_batch_audio_seconds: float = _float("MAX_BATCH_AUDIO_SECONDS", 240.0)
    # Requests longer than this always run on their own (long audio is chunked
    # internally by NeMo and would otherwise pad every batch neighbour).
    max_batch_item_seconds: float = _float("MAX_BATCH_ITEM_SECONDS", 30.0)
    # Audio up to this many seconds is handed to the model as an in-memory
    # float32 array instead of a temporary WAV file (saves a write + read per
    # inference; measured ~16 % of a streaming inference). Longer audio still
    # goes through a temp file so NeMo's chunked long-form decoding applies.
    inline_audio_max_seconds: float = _float("INLINE_AUDIO_MAX_SECONDS", 30.0)
    # Max duration ratio inside a batch (longest / shortest). Padding a short clip
    # up to a much longer neighbour changes its tail transcript, so mismatched
    # lengths are decoded in separate batches. 0 disables the check.
    batch_duration_ratio: float = _float("BATCH_DURATION_RATIO", 2.0)

    # --- Admission control --------------------------------------------------
    # Maximum number of queued (not yet running) requests. 0 disables the limit.
    max_queue_size: int = _int("MAX_QUEUE_SIZE", 32)
    # How long a request may wait for a free model before it is rejected.
    queue_timeout_s: float = _float("QUEUE_TIMEOUT_S", 120.0)
    # Value of the ``Retry-After`` header sent with 503 responses.
    retry_after_s: int = _int("RETRY_AFTER_S", 2)

    # --- Live streaming limits ---------------------------------------------
    # Maximum number of concurrent WebSocket sessions per replica. ``0`` means
    # auto: 8 when the replica runs in bf16/fp16, 4 in fp32 (measured on an
    # RTX 3090 with batching: 12 sessions keep real time in bf16).
    max_concurrent_streams: int = _int("MAX_CONCURRENT_STREAMS", 0)
    # Maximum audio a client may have queued ahead of the model (seconds).
    stream_max_queue_seconds: float = _float("STREAM_MAX_QUEUE_SECONDS", 120.0)
    # Hard limit on a single WebSocket session (seconds).
    stream_max_session_seconds: float = _float("STREAM_MAX_SESSION_SECONDS", 3600.0)

    # --- Observability ------------------------------------------------------
    # Expose Prometheus metrics on GET /metrics.
    metrics_enabled: bool = _bool("METRICS_ENABLED", True)

    # --- Streaming transcription ---
    # Minimum audio duration (seconds) before the first inference run.
    stream_min_duration: float = _float("STREAM_MIN_DURATION", 1.0)
    # Minimum new audio (seconds) that must arrive before re-running inference.
    stream_retranscribe_interval: float = _float("STREAM_RETRANSCRIBE_INTERVAL", 0.5)
    # Minimum number of agreed words needed to commit.
    stream_stable_words: int = _int("STREAM_STABLE_WORDS", 1)
    # How many consecutive hypotheses must agree on a word prefix to commit it.
    stream_stable_iters: int = _int("STREAM_STABLE_ITERS", 2)
    # Hard buffer cap (seconds): forces a trim, and a commit if nothing agreed.
    stream_max_duration: float = _float("STREAM_MAX_DURATION", 30.0)
    # Seconds of already-committed audio kept before a trim point, so the model
    # has acoustic context at the start of the buffer.
    stream_context_duration: float = _float("STREAM_CONTEXT_DURATION", 1.5)
    # Buffer length (seconds) after which it is trimmed at the end of the last
    # committed sentence. Longer = more context per decode but more GPU time.
    stream_trim_duration: float = _float("STREAM_TRIM_DURATION", 10.0)
    # Words ending within this many seconds of the buffer end are not committed.
    stream_holdback: float = _float("STREAM_HOLDBACK", 0.0)

    # --- SSE streaming (POST /v1/audio/transcriptions with stream=true) ---
    # Uploaded audio is split into chunks of at most this many seconds, each
    # transcribed in turn and emitted as a ``transcript.text.delta`` event.
    sse_chunk_duration: float = _float("SSE_CHUNK_DURATION", 20.0)
    # Chunk boundaries are placed at the quietest point within this many seconds
    # before the chunk limit, so words are not cut in half.
    sse_split_search: float = _float("SSE_SPLIT_SEARCH", 4.0)


settings = Settings()
