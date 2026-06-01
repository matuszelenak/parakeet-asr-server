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
    # Minimum word-prefix length that must match across iterations to count as stable.
    stream_stable_words: int = _int("STREAM_STABLE_WORDS", 4)
    # How many consecutive matching iterations are required to commit a segment.
    stream_stable_iters: int = _int("STREAM_STABLE_ITERS", 2)
    # Maximum buffer duration (seconds) before a forced commit, regardless of stability.
    stream_max_duration: float = _float("STREAM_MAX_DURATION", 30.0)
    # Seconds of already-committed audio to prepend as context on each inference
    # call.  Gives the model acoustic context so it can correctly transcribe the
    # start of each new segment instead of starting cold at a buffer boundary.
    stream_context_duration: float = _float("STREAM_CONTEXT_DURATION", 3.0)


settings = Settings()
