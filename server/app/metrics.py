"""Prometheus metrics for the ASR server.

Everything lives in a module-level registry so the values are visible from
``GET /metrics``. Importing this module is cheap and side-effect free apart from
creating the metric objects; nothing is exported unless the endpoint is enabled
(``METRICS_ENABLED``).

Naming: ``asr_<subsystem>_<unit>``, counters end in ``_total``, and every
metric carries only low-cardinality labels (route templates, model modes) so it
is safe to scrape at scale.
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from .config import settings

# A dedicated registry keeps the exposition clean (no default python/process
# metrics that would otherwise need a version-specific collector) and lets
# tests build isolated instances.
REGISTRY: CollectorRegistry = CollectorRegistry(auto_describe=True)

# --- HTTP layer ---------------------------------------------------------------
http_requests_total = Counter(
    "asr_http_requests_total",
    "HTTP requests handled, by route template and status code.",
    ["route", "method", "status"],
    registry=REGISTRY,
)
http_request_duration_seconds = Histogram(
    "asr_http_request_duration_seconds",
    "End-to-end HTTP request duration.",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
    registry=REGISTRY,
)
rejected_total = Counter(
    "asr_rejected_total",
    "Requests/sessions rejected by admission control, by reason.",
    ["reason"],
    registry=REGISTRY,
)
upload_bytes = Histogram(
    "asr_upload_bytes",
    "Size of accepted audio uploads.",
    buckets=(1e5, 5e5, 1e6, 5e6, 2e7, 5e7, 1e8, 2e8),
    registry=REGISTRY,
)
audio_seconds_total = Counter(
    "asr_audio_seconds_total",
    "Audio decoded and submitted for transcription.",
    registry=REGISTRY,
)

# --- Model / scheduler --------------------------------------------------------
model_ready = Gauge(
    "asr_model_ready", "1 when at least one model replica is loaded.", registry=REGISTRY
)
workers = Gauge(
    "asr_workers", "Number of model replicas (one per GPU by default).", registry=REGISTRY
)
queue_depth = Gauge(
    "asr_queue_depth", "Requests waiting for a model replica.", registry=REGISTRY
)
queue_capacity = Gauge(
    "asr_queue_capacity",
    "Admission limit for queued requests (0 = unlimited).",
    registry=REGISTRY,
)
inflight_requests = Gauge(
    "asr_inflight_requests", "Requests currently being decoded.", registry=REGISTRY
)
queue_wait_seconds = Histogram(
    "asr_queue_wait_seconds",
    "Time a request spent queued before decoding started.",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    registry=REGISTRY,
)
inference_batches_total = Counter(
    "asr_inference_batches_total",
    "Model calls executed, by decoding mode.",
    ["mode"],
    registry=REGISTRY,
)
inference_batch_size = Histogram(
    "asr_inference_batch_size",
    "Requests decoded per model call.",
    ["mode"],
    buckets=(1, 2, 3, 4, 6, 8, 12, 16, 32),
    registry=REGISTRY,
)
inference_seconds = Histogram(
    "asr_inference_seconds",
    "Wall time of a model call.",
    ["mode"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    registry=REGISTRY,
)
inference_audio_seconds = Counter(
    "asr_inference_audio_seconds_total",
    "Audio duration decoded, by mode.",
    ["mode"],
    registry=REGISTRY,
)
inference_errors_total = Counter(
    "asr_inference_errors_total",
    "Failed model calls (all per-item retries exhausted).",
    registry=REGISTRY,
)
split_batches_total = Counter(
    "asr_split_batches_total",
    "Batches that failed as a whole and were retried request-by-request.",
    registry=REGISTRY,
)

# --- Live streaming -----------------------------------------------------------
stream_sessions_active = Gauge(
    "asr_stream_sessions_active", "Open WebSocket transcription sessions.", registry=REGISTRY
)
stream_sessions_total = Counter(
    "asr_stream_sessions_total",
    "WebSocket sessions that ended, by outcome.",
    ["outcome"],
    registry=REGISTRY,
)
stream_session_seconds = Histogram(
    "asr_stream_session_seconds",
    "Duration of a WebSocket session.",
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600),
    registry=REGISTRY,
)
stream_audio_seconds = Counter(
    "asr_stream_audio_seconds_total",
    "Audio received over WebSocket sessions.",
    registry=REGISTRY,
)
stream_commits_total = Counter(
    "asr_stream_commits_total", "Committed stream segments.", registry=REGISTRY
)
stream_partials_total = Counter(
    "asr_stream_partials_total", "Partial (unstable) hypotheses emitted.", registry=REGISTRY
)
stream_commit_lag_seconds = Histogram(
    "asr_stream_commit_lag_seconds",
    "How much received audio is still uncommitted when a segment is committed.",
    buckets=(0.1, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 30),
    registry=REGISTRY,
)

# The module-level metrics above are bound to our own registry, so the default
# registry stays empty and nothing else leaks into /metrics.


def mode_label(longform: bool) -> str:
    """Decoding mode used as the ``mode`` label."""
    return "longform" if longform else "plain"


def metrics_enabled() -> bool:
    return settings.metrics_enabled
