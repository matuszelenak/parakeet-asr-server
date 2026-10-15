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

    # Largest upload accepted, in megabytes.
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 200)

    # Local-attention context window used for long-form transcription.
    longform_context: int = _int("LONGFORM_CONTEXT", 256)

    # Directory of a built frontend to serve as static files. Empty (the dev
    # default) disables static serving so the Vite dev server handles the UI.
    # The production image sets this to the baked-in build (e.g. /app/static).
    static_dir: str = os.environ.get("STATIC_DIR", "")

    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = _int("PORT", 8000)

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
