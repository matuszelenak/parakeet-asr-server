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


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _csv(name: str) -> list[str] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _int_csv(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # HuggingFace / NeMo model id to load. Defaults to NVIDIA's cache-aware
    # streaming Nemotron model, which transcribes via native streaming.
    model_name: str = os.environ.get(
        "MODEL_NAME", "nvidia/nemotron-3.5-asr-streaming-0.6b"
    )

    # Number of model replicas. Defaults to one per visible GPU (resolved at
    # startup when this is 0). Each replica serves one streaming session or one
    # offline request at a time.
    num_workers: int = _int("NUM_WORKERS", 0)

    # Explicit device list, e.g. "cuda:0,cuda:1". When unset, devices are
    # assigned round-robin across the visible GPUs.
    devices: list[str] | None = field(default_factory=lambda: _csv("DEVICES"))

    # Largest upload accepted, in megabytes.
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 200)

    # Directory of a built frontend to serve as static files. Empty (the dev
    # default) disables static serving so the Vite dev server handles the UI.
    # The production image sets this to the baked-in build (e.g. /app/static).
    static_dir: str = os.environ.get("STATIC_DIR", "")

    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = _int("PORT", 8000)

    # --- Cache-aware streaming ---
    # Attention context size [left, right] in 80 ms encoder frames, selecting
    # one of the model's trained look-aheads. The right context sets the
    # streaming latency: [56,0]=80ms, [56,1]=160ms, [56,3]=320ms, [56,6]=560ms,
    # [56,13]=1.12s (more look-ahead → higher accuracy, higher latency).
    att_context_size: list[int] = field(
        default_factory=lambda: _int_csv("ATT_CONTEXT_SIZE", [56, 6])
    )

    # Default transcription language as a BCP-47 locale (e.g. "en-US", "de-DE")
    # or "auto" to let the model detect and tag the spoken language. Clients may
    # override this per request/session.
    target_lang: str = os.environ.get("TARGET_LANG", "auto")

    # Strip the trailing language tag (e.g. "<en-US>") the model appends after
    # the transcript's terminal punctuation.
    strip_lang_tags: bool = _bool("STRIP_LANG_TAGS", True)

    # Normalise input features per chunk during streaming (recommended; only
    # applied when the model uses per-feature/all-feature input normalization).
    online_normalization: bool = _bool("ONLINE_NORMALIZATION", True)


settings = Settings()
