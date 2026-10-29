"""FastAPI application exposing Parakeet/Canary."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

import logfire
import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
)
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from .audio import (
    TARGET_SR,
    InvalidAudioError,
    decode_16k_mono,
    float32_to_wav_path,
    split_on_silence,
)
from . import metrics
from .config import settings
from .languages import (
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    Language,
    model_supports_languages,
)
from .model_pool import PoolError, pool
from .schemas import (
    BatchStats,
    CharTimestamp,
    HealthResponse,
    LanguageInfo,
    ModelCard,
    ModelList,
    OpenAIError,
    OpenAIErrorBody,
    QueueStats,
    SegmentTimestamp,
    StreamCapacity,
    StreamConfig,
    StreamEvent,
    TimestampedResponse,
    TranscriptionResponse,
    TranscriptTextDeltaEvent,
    TranscriptTextDoneEvent,
    VerboseSegment,
    VerboseTranscriptionResponse,
    WordTimestamp,
)
from .streaming import continuous_transcriber

# 16 kHz mono everywhere (see audio.TARGET_SR); local alias for readability in
# the WebSocket handler's byte accounting.
SAMPLING_RATE = TARGET_SR

# Live WebSocket sessions currently open in this process (admission control).
_active_streams = 0


def _stream_limit() -> int:
    """Concurrent live-session limit, resolved from ``MAX_CONCURRENT_STREAMS``.

    ``0`` means auto: a bf16/fp16 replica decodes roughly 2.5x faster than fp32,
    so it can hold about twice as many real-time sessions. Measured on one RTX
    3090 (Slovak, 500 ms frames): 12 sessions in bf16, 8 comfortably; 4 in fp32.
    """
    if settings.max_concurrent_streams > 0:
        return settings.max_concurrent_streams
    reduced = pool.stats().get("dtype") in {"bfloat16", "float16"}
    return 8 if reduced else 4

logfire.configure(
    service_name="parakeet-asr-server",
    send_to_logfire="if-token-present",
)
logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()], level=logging.INFO)
# NeMo emits these on every model.transcribe() call; they are not actionable.
logging.getLogger("nemo.collections.asr.data.dataloader").setLevel(logging.ERROR)
logging.getLogger("lhotse").setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await pool.startup()
    yield
    await pool.shutdown()


app = FastAPI(
    title="Parakeet ASR Server",
    version="0.1.0",
    description="Transcribe audio with NVIDIA Parakeet/Canary",
    lifespan=lifespan,
)

# Instrument FastAPI: emit a span per request with route, status, and timing.
logfire.instrument_fastapi(app, capture_headers=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Metrics ---------------------------------------------------------------

def _route_label(request: Request) -> str:
    """Route *template* (e.g. ``/v1/models/{model_id}``) for low cardinality."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or "unmatched"


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """Count requests and observe latency by route template.

    For streaming responses (``stream=true``) the histogram records the time
    until the response *starts*, not until the last event is sent; the SSE work
    itself shows up in the ``asr_inference_*`` metrics.
    """
    if not settings.metrics_enabled or request.url.path == "/metrics":
        return await call_next(request)

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metrics.http_requests_total.labels(
            route=_route_label(request), method=request.method, status="500"
        ).inc()
        raise
    route = _route_label(request)
    metrics.http_requests_total.labels(
        route=route, method=request.method, status=str(response.status_code)
    ).inc()
    metrics.http_request_duration_seconds.labels(route=route).observe(
        time.perf_counter() - started
    )
    return response


if settings.metrics_enabled:

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        """Prometheus exposition (text format 0.0.4)."""
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(
            content=generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST
        )


# --- OpenAI-style errors ------------------------------------------------------
# The OpenAI-compatible routes return ``{"error": {...}}`` bodies (and 400
# instead of 422 for validation errors) so OpenAI SDKs and LiteLLM surface the
# real message. The native routes keep FastAPI's ``{"detail": ...}`` shape,
# which the frontend relies on.
_OPENAI_PATH_PREFIXES = ("/v1/audio/", "/v1/models")


def _is_openai_path(request: Request) -> bool:
    return request.url.path.startswith(_OPENAI_PATH_PREFIXES)


def _openai_error(
    status_code: int,
    message: str,
    *,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    error = OpenAIError(
        error=OpenAIErrorBody(
            message=message,
            type="invalid_request_error" if status_code < 500 else "server_error",
            param=param,
        )
    )
    return JSONResponse(error.model_dump(), status_code=status_code, headers=headers)


@app.exception_handler(StarletteHTTPException)
async def _http_exception(request: Request, exc: StarletteHTTPException) -> Response:
    if _is_openai_path(request):
        return _openai_error(exc.status_code, str(exc.detail), headers=exc.headers)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def _validation_exception(request: Request, exc: RequestValidationError) -> Response:
    if _is_openai_path(request):
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = [str(part) for part in first.get("loc", ())[1:]]
        return _openai_error(
            400, first.get("msg", "invalid request"), param=".".join(loc) or None
        )
    return await request_validation_exception_handler(request, exc)


# --- Upload handling ---------------------------------------------------------
async def _read_upload(upload: UploadFile) -> bytes:
    """Read an upload, enforcing the size limit."""
    raw = await upload.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        metrics.rejected_total.labels(reason="upload_too_large").inc()
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {settings.max_upload_mb} MB limit",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    metrics.upload_bytes.observe(len(raw))
    return raw


async def _read_and_prepare(upload: UploadFile) -> tuple[Any, float, str]:
    """Decode an upload and return ``(input, duration_s, kind)``.

    Short audio is returned as an in-memory float32 16 kHz array (no temp file);
    anything longer than ``INLINE_AUDIO_MAX_SECONDS`` is written to a temporary
    16 kHz mono WAV so NeMo's chunked long-form decoding still applies. The
    caller must ``_cleanup`` the input when ``kind == "file"``.
    """
    raw = await _read_upload(upload)
    try:
        # Decoding/resampling is CPU-bound; keep it off the event loop.
        samples = await asyncio.to_thread(decode_16k_mono, raw)
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    duration = len(samples) / TARGET_SR
    if duration <= settings.inline_audio_max_seconds:
        return samples, duration, "array"
    path = await asyncio.to_thread(float32_to_wav_path, samples)
    return path, duration, "file"


def _cleanup(audio: Any, kind: str) -> None:
    """Delete a temporary WAV unless the input was an in-memory array."""
    if kind != "file":
        return
    try:
        os.remove(audio)
    except OSError:
        pass


def _result_text(result: Any) -> str:
    return getattr(result, "text", "") or ""


def _admission_http_error(exc: PoolError) -> HTTPException:
    """Map scheduler admission failures onto HTTP 503 + ``Retry-After``."""
    return HTTPException(
        status_code=503,
        detail=str(exc),
        headers={"Retry-After": str(settings.retry_after_s)},
    )


async def _run_pool(audio: Any, duration: float, **kwargs: Any) -> Any:
    """Submit one input to the scheduler, translating admission errors to 503.

    ``duration`` is passed along so the scheduler can batch and bound VRAM
    without re-reading the input.
    """
    metrics.audio_seconds_total.inc(duration)
    try:
        results = await pool.transcribe([audio], durations=[duration], **kwargs)
    except PoolError as exc:
        logfire.warning("request rejected: {error}", error=str(exc))
        raise _admission_http_error(exc) from exc
    return results[0]


def _env_language(code: str) -> Language | None:
    """Parse DEFAULT_SOURCE_LANG / DEFAULT_TARGET_LANG; ignore invalid values."""
    if not code:
        return None
    try:
        return Language(code)
    except ValueError:
        logfire.warning("ignoring unknown language code in env: {code}", code=code)
        return None


def _resolve_languages(
    source_lang: Language | None,
    target_lang: Language | None,
    *,
    strict: bool = False,
) -> dict[str, str]:
    """Validate language args against the configured model's capabilities.

    Returns kwargs (``source_lang`` / ``target_lang``) to pass to the model, or
    an empty dict when the model does not use language selection. Languages are
    ignored (``strict=False``, the default) for such models: clients — including
    this project's own UI, scripts and LiteLLM configs — routinely send a
    language hint, and a model swap to a transducer (e.g. Parakeet, which
    auto-detects) must not start failing those requests. Pass ``strict=True`` to
    reject them with 400 instead.
    """
    if not model_supports_languages(settings.model_name):
        if strict and (source_lang is not None or target_lang is not None):
            raise HTTPException(
                status_code=400,
                detail="the configured model does not support language selection",
            )
        return {}

    # Canary-style models require both. Fall back to the deployment default
    # (DEFAULT_SOURCE_LANG/DEFAULT_TARGET_LANG), then to English.
    default_src = _env_language(settings.default_source_lang) or DEFAULT_LANGUAGE
    default_tgt = _env_language(settings.default_target_lang) or default_src
    src = source_lang or default_src
    tgt = target_lang or source_lang or default_tgt
    return {"source_lang": src.value, "target_lang": tgt.value}


def _parse_language(code: str | None) -> Language | None:
    """Map an OpenAI ``language`` string (ISO-639-1) to our Language enum.

    Empty/None yields None (use the model default). An unknown code is a 400.
    """
    if code is None or code == "":
        return None
    try:
        # Accept BCP-47 style tags such as "en-US" by keeping the language part.
        return Language(code.split("-")[0].split("_")[0].lower())
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"unsupported language code: {code}"
        ) from exc


def _as_text(value: Any) -> str:
    """NeMo returns BPE token timestamps as lists (``['Po']``); flatten to str."""
    if isinstance(value, (list, tuple)):
        return "".join(str(part) for part in value)
    return "" if value is None else str(value)


def _extract_words(result: Any) -> list[WordTimestamp]:
    stamps = getattr(result, "timestamp", None) or {}
    return [
        WordTimestamp(
            word=_as_text(w.get("word", "")), start=w["start"], end=w["end"]
        )
        for w in stamps.get("word", [])
    ]


def _extract_verbose_segments(result: Any) -> list[VerboseSegment]:
    stamps = getattr(result, "timestamp", None) or {}
    # Transducer models report word timestamps but no segments; fall back to words
    # so SRT/VTT output is still populated.
    entries = stamps.get("segment") or stamps.get("word") or []
    return [
        VerboseSegment(
            id=i,
            start=s["start"],
            end=s["end"],
            text=_as_text(s.get("segment") or s.get("word") or ""),
        )
        for i, s in enumerate(entries)
    ]


def _format_ts(seconds: float, sep: str) -> str:
    """Format ``seconds`` as ``HH:MM:SS<sep>mmm`` (sep ``,`` for SRT, ``.`` for VTT)."""
    ms_total = int(round(max(seconds, 0.0) * 1000))
    hours, ms_total = divmod(ms_total, 3_600_000)
    minutes, ms_total = divmod(ms_total, 60_000)
    secs, millis = divmod(ms_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _render_srt(segments: list[VerboseSegment]) -> str:
    blocks = [
        f"{i}\n{_format_ts(s.start, ',')} --> {_format_ts(s.end, ',')}\n{s.text.strip()}"
        for i, s in enumerate(segments, start=1)
    ]
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _render_vtt(segments: list[VerboseSegment]) -> str:
    blocks = [
        f"{_format_ts(s.start, '.')} --> {_format_ts(s.end, '.')}\n{s.text.strip()}"
        for s in segments
    ]
    return "WEBVTT\n\n" + "\n\n".join(blocks) + ("\n" if blocks else "")


_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}


def _check_response_format(response_format: str) -> None:
    if response_format not in _RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported response_format '{response_format}'; expected one of "
                f"{', '.join(sorted(_RESPONSE_FORMATS))}"
            ),
        )


def _sse(event: BaseModel) -> str:
    return f"data: {event.model_dump_json()}\n\n"


async def _openai_transcribe_stream(
    *, file: UploadFile, languages: dict[str, str], longform: bool
) -> StreamingResponse:
    """Server-sent-events variant of the OpenAI transcription endpoint.

    NeMo models are not incremental, so the audio is split into chunks at quiet
    points (``SSE_CHUNK_DURATION``) and each chunk is transcribed in turn; every
    chunk's text is emitted as a ``transcript.text.delta`` event, followed by a
    single ``transcript.text.done`` event with the full transcript. Errors that
    happen after the stream has started are sent as an ``{"error": {...}}``
    event, which the OpenAI SDKs raise as an exception.
    """
    raw = await _read_upload(file)
    try:
        samples = await asyncio.to_thread(decode_16k_mono, raw)
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    chunks = split_on_silence(
        samples, settings.sse_chunk_duration, settings.sse_split_search
    )

    async def events() -> AsyncGenerator[str, None]:
        parts: list[str] = []
        with logfire.span(
            "transcribe_sse",
            duration=len(samples) / TARGET_SR,
            chunks=len(chunks),
            longform=longform,
            **languages,
        ):
            try:
                for chunk in chunks:
                    # Skip slivers too short to contain speech.
                    if len(chunk) < TARGET_SR // 10:
                        continue
                    # NB: chunks go through a temp WAV rather than the in-memory
                    # array path. NeMo's tensor dataset frames a given utterance
                    # slightly differently from its lhotse path (4 of 31 chunks on
                    # our Slovak sample differed, mixed quality), so this bulk path
                    # keeps the lhotse behaviour that the accuracy runs validated;
                    # the ~15 ms/chunk saved is not worth the wobble here.
                    path = float32_to_wav_path(chunk)
                    try:
                        result = await _run_pool(
                            path, len(chunk) / TARGET_SR, longform=longform, **languages
                        )
                    finally:
                        _cleanup(path, "file")
                    text = _result_text(result).strip()
                    if not text:
                        continue
                    yield _sse(
                        TranscriptTextDeltaEvent(delta=f" {text}" if parts else text)
                    )
                    parts.append(text)
            except Exception as exc:
                logfire.exception("sse transcription failed: {exc}", exc=str(exc))
                yield _sse(
                    OpenAIError(
                        error=OpenAIErrorBody(message=str(exc), type="server_error")
                    )
                )
                return
            yield _sse(TranscriptTextDoneEvent(text=" ".join(parts)))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _openai_transcribe(
    *,
    file: UploadFile,
    source: Language | None,
    languages: dict[str, str],
    response_format: str,
    timestamp_granularities: list[str],
    longform: bool,
    task: Literal["transcribe", "translate"],
) -> Response:
    """Shared core for the OpenAI-compatible transcription/translation endpoints."""
    # Word/segment timestamps are needed for verbose_json (and word granularity)
    # and for the subtitle formats, which are built from segment timing.
    want_word = "word" in timestamp_granularities
    want_timestamps = response_format in {"verbose_json", "srt", "vtt"} or want_word

    audio, duration, kind = await _read_and_prepare(file)
    try:
        result = await _run_pool(
            audio, duration, timestamps=want_timestamps, longform=longform, **languages
        )
    finally:
        _cleanup(audio, kind)

    text = _result_text(result)

    if response_format == "text":
        return PlainTextResponse(text + "\n")

    if response_format in {"srt", "vtt"}:
        segments = _extract_verbose_segments(result)
        body = _render_srt(segments) if response_format == "srt" else _render_vtt(segments)
        return PlainTextResponse(body)

    if response_format == "verbose_json":
        language = (source or DEFAULT_LANGUAGE).value if languages else "en"
        payload = VerboseTranscriptionResponse(
            task=task,
            language=language,
            duration=duration,
            text=text,
            words=_extract_words(result) if want_word else [],
            segments=_extract_verbose_segments(result),
        )
        return Response(
            content=payload.model_dump_json(), media_type="application/json"
        )

    # response_format == "json" (the default)
    return Response(
        content=TranscriptionResponse(text=text).model_dump_json(),
        media_type="application/json",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    supports = model_supports_languages(settings.model_name)
    languages = (
        [LanguageInfo(code=lang.value, name=LANGUAGE_NAMES[lang]) for lang in Language]
        if supports
        else []
    )
    return HealthResponse(
        status="ok" if pool.ready else "loading",
        model=settings.model_path or settings.model_name,
        workers=pool.size,
        ready=pool.ready,
        supports_languages=supports,
        languages=languages,
        queue=QueueStats(
            pending=pool.pending,
            inflight=pool.inflight,
            capacity=settings.max_queue_size,
        ),
        batching=BatchStats(
            window_ms=settings.batch_window_ms,
            max_batch_size=settings.max_batch_size,
            max_batch_audio_seconds=settings.max_batch_audio_seconds,
        ),
        stream_capacity=StreamCapacity(
            active=_active_streams, limit=_stream_limit()
        ),
        dtype=pool.stats().get("dtype"),
        matmul_precision=settings.matmul_precision,
    )


@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    source_lang: Language | None = Form(None),
    target_lang: Language | None = Form(None),
) -> TranscriptionResponse:
    """Plain-text transcription (or translation when target differs from source)."""
    languages = _resolve_languages(source_lang, target_lang)
    audio, duration, kind = await _read_and_prepare(file)
    try:
        result = await _run_pool(audio, duration, **languages)
    finally:
        _cleanup(audio, kind)
    return TranscriptionResponse(text=_result_text(result))


@app.post("/v1/transcribe/timestamps", response_model=TimestampedResponse)
async def transcribe_timestamps(
    file: UploadFile = File(...),
    source_lang: Language | None = Form(None),
    target_lang: Language | None = Form(None),
) -> TimestampedResponse:
    """Transcription with word, segment, and char level timestamps."""
    languages = _resolve_languages(source_lang, target_lang)
    audio, duration, kind = await _read_and_prepare(file)
    try:
        result = await _run_pool(audio, duration, timestamps=True, **languages)
    finally:
        _cleanup(audio, kind)

    stamps = getattr(result, "timestamp", None) or {}

    words = [
        WordTimestamp(
            word=_as_text(w.get("word", "")), start=w["start"], end=w["end"]
        )
        for w in stamps.get("word", [])
    ]
    segments = [
        SegmentTimestamp(
            segment=_as_text(s.get("segment") or s.get("word") or ""),
            start=s["start"],
            end=s["end"],
        )
        for s in (stamps.get("segment") or stamps.get("word") or [])
    ]
    chars = [
        CharTimestamp(char=_as_text(c.get("char", "")), start=c["start"], end=c["end"])
        for c in stamps.get("char", [])
    ]
    return TimestampedResponse(
        text=_result_text(result), words=words, segments=segments, chars=chars
    )


@app.post("/v1/transcribe/longform", response_model=TranscriptionResponse)
async def transcribe_longform(
    file: UploadFile = File(...),
    source_lang: Language | None = Form(None),
    target_lang: Language | None = Form(None),
) -> TranscriptionResponse:
    """Long-form transcription using local-attention windowing.

    Switches the encoder to limited-context self-attention so very long
    recordings can be transcribed without exhausting GPU memory.
    """
    languages = _resolve_languages(source_lang, target_lang)
    audio, duration, kind = await _read_and_prepare(file)
    try:
        result = await _run_pool(audio, duration, longform=True, **languages)
    finally:
        _cleanup(audio, kind)
    return TranscriptionResponse(text=_result_text(result))


_MODEL_CREATED = int(time.time())


def _model_card() -> ModelCard:
    return ModelCard(id=settings.served_model_name, created=_MODEL_CREATED)


@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """OpenAI-compatible model listing (the single configured model)."""
    return ModelList(data=[_model_card()])


@app.get("/v1/models/{model_id:path}", response_model=ModelCard)
async def retrieve_model(model_id: str) -> ModelCard:
    if model_id not in {settings.served_model_name, settings.model_name}:
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return _model_card()


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
    timestamp_granularities: list[str] | None = Form(
        None, alias="timestamp_granularities[]"
    ),
    timestamp_granularities_alt: list[str] | None = Form(
        None, alias="timestamp_granularities"
    ),
    stream: bool = Form(False),
    # --- Non-standard extensions (ignored by standard OpenAI clients) ---
    # ``target_language`` keeps Canary's general source->target translation
    # available through the transcriptions endpoint; ``longform`` enables
    # local-attention windowing for very long recordings.
    target_language: str | None = Form(None),
    longform: bool = Form(False),
) -> Response:
    """OpenAI-compatible transcription endpoint.

    Mirrors ``POST /v1/audio/transcriptions``. The ``model``, ``prompt``, and
    ``temperature`` fields are accepted for compatibility but not used by the
    NeMo backend. ``response_format`` selects the body shape (``json``,
    ``text``, ``verbose_json``, ``srt``, ``vtt``); ``timestamp_granularities[]``
    (``word`` / ``segment``) controls which timestamps are populated in
    ``verbose_json``.

    ``stream=true`` returns server-sent events (``transcript.text.delta`` /
    ``transcript.text.done``) like OpenAI's ``gpt-4o-transcribe``. As with
    OpenAI, streaming applies to the ``json`` and ``text`` formats and is
    ignored for the timestamped ones.

    ``language`` is ignored (rather than rejected) when the configured model
    does not support language selection.
    """
    _check_response_format(response_format)
    source = _parse_language(language)
    target = _parse_language(target_language) or source
    languages = _resolve_languages(source, target, strict=False)
    if stream and response_format in {"json", "text"}:
        return await _openai_transcribe_stream(
            file=file, languages=languages, longform=longform
        )
    granularities = timestamp_granularities or timestamp_granularities_alt or []
    return await _openai_transcribe(
        file=file,
        source=source,
        languages=languages,
        response_format=response_format,
        timestamp_granularities=granularities,
        longform=longform,
        task="transcribe",
    )


@app.post("/v1/audio/translations")
async def audio_translations(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float | None = Form(None),
    # --- Non-standard extension ---
    # OpenAI's translations endpoint always targets English and auto-detects the
    # source. Canary does not auto-detect, so ``language`` names the source
    # (default English); the target is always English.
    language: str | None = Form(None),
    longform: bool = Form(False),
) -> Response:
    """OpenAI-compatible translation endpoint (translates audio into English).

    Mirrors ``POST /v1/audio/translations``. The source language defaults to
    English and may be set via the non-standard ``language`` field; the target
    is always English.
    """
    _check_response_format(response_format)
    source = _parse_language(language) or DEFAULT_LANGUAGE
    return await _openai_transcribe(
        file=file,
        source=source,
        languages=_resolve_languages(source, Language.en),
        response_format=response_format,
        timestamp_granularities=[],
        longform=longform,
        task="translate",
    )


@app.websocket("/v1/transcribe/stream")
async def transcribe_stream(
    websocket: WebSocket,
    source_lang: Language | None = None,
    target_lang: Language | None = None,
) -> None:
    """Continuous streaming transcription over WebSocket.

    Optional query params ``source_lang`` and ``target_lang`` select the
    language (e.g. ``?source_lang=sk&target_lang=en`` for Slovak→English
    translation).  When omitted, English is used.  Invalid values cause the
    WebSocket handshake to be rejected with HTTP 400.

    The client sends raw PCM-16 mono 16 kHz audio as binary frames and signals
    end-of-stream with a text frame containing ``{"type": "end"}``.

    The server responds with JSON StreamEvent text frames::

        {"type": "partial",   "text": "...", "start": 0.0, "id": 0}
        {"type": "committed", "text": "...", "start": 0.0, "id": 0}
        {"type": "final",     "text": "...", "start": 1.4, "id": 1}

    ``partial``  – in-progress transcription of the current segment (may change).
    ``committed``– confirmed segment; words will not be revised.
    ``final``    – last segment emitted after end-of-stream.

    Admission control: at most ``MAX_CONCURRENT_STREAMS`` sessions may be open
    (further connections are accepted and closed with 1013), a client may not
    queue more than ``STREAM_MAX_QUEUE_SECONDS`` of audio ahead of the model
    (faster-than-real-time feeds are cut off 1013), and a session is closed
    after ``STREAM_MAX_SESSION_SECONDS``. The model replica is checked out per
    inference, so batch requests are not blocked by a live session.
    """
    global _active_streams

    # Lenient: the browser always sends source_lang/target_lang, and a client
    # cannot know whether the loaded model takes them — ignore them instead of
    # failing the upgrade (e.g. on a Parakeet/transducer deployment).
    languages = _resolve_languages(source_lang, target_lang, strict=False)
    await websocket.accept()

    if not pool.ready:
        await websocket.close(code=1013, reason="model not ready")
        return

    if _active_streams >= _stream_limit():
        limit = _stream_limit()
        metrics.rejected_total.labels(reason="stream_limit").inc()
        metrics.stream_sessions_total.labels(outcome="rejected").inc()
        await websocket.close(
            code=1013,
            reason=f"server at capacity ({limit} streams); retry",
        )
        return

    _active_streams += 1
    metrics.stream_sessions_active.set(_active_streams)
    session_started = time.monotonic()
    session_outcome = "error"
    received_s = 0.0
    # Backpressure: a client may queue at most this much *unprocessed* audio
    # (``STREAM_MAX_QUEUE_SECONDS``). Real-time clients never get near it; a
    # fast-feed client is closed with 1013 and pointed at the SSE endpoint,
    # because pausing the ASGI receive path would also stall WebSocket control
    # frames (pings) and make the session look dead.
    max_queued_samples = int(settings.stream_max_queue_seconds * SAMPLING_RATE)
    queued_samples = 0

    audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    # --- Read optional per-session configure message ----------------------------
    # The client MAY send {"type": "configure", ...overrides} as its very first
    # frame.  We wait up to 2 s; if no text frame arrives (or it isn't a
    # configure message) we fall through using server defaults.  A binary audio
    # frame that arrives before any configure message is decoded and queued so
    # no audio is lost.
    cfg = StreamConfig()
    ended_upfront = False
    try:
        first = await asyncio.wait_for(websocket.receive(), timeout=2.0)
        if first.get("text"):
            try:
                msg = json.loads(first["text"])
                if msg.get("type") == "configure":
                    cfg = StreamConfig.model_validate(
                        {k: v for k, v in msg.items() if k != "type"}
                    )
                elif msg.get("type") == "end":
                    # A client may end the stream before sending any audio; the
                    # probe above already consumed that frame.
                    ended_upfront = True
            except (json.JSONDecodeError, AttributeError, ValueError):
                pass
        elif first.get("bytes"):
            samples = (
                np.frombuffer(first["bytes"], dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            await audio_queue.put(samples)
    except asyncio.TimeoutError:
        pass

    # Resolve each field: client value → server default.
    min_duration        = cfg.min_duration        if cfg.min_duration        is not None else settings.stream_min_duration
    retranscribe_interval = cfg.retranscribe_interval if cfg.retranscribe_interval is not None else settings.stream_retranscribe_interval
    stable_words        = cfg.stable_words        if cfg.stable_words        is not None else settings.stream_stable_words
    stable_iters        = cfg.stable_iters        if cfg.stable_iters        is not None else settings.stream_stable_iters
    max_duration        = cfg.max_duration        if cfg.max_duration        is not None else settings.stream_max_duration
    context_duration    = cfg.context_duration    if cfg.context_duration    is not None else settings.stream_context_duration
    trim_duration       = cfg.trim_duration       if cfg.trim_duration       is not None else settings.stream_trim_duration
    holdback            = cfg.holdback            if cfg.holdback            is not None else settings.stream_holdback
    # ----------------------------------------------------------------------------

    async def _reader() -> None:
        nonlocal received_s, queued_samples, session_outcome
        if ended_upfront:
            session_outcome = "ended_by_client"
            await audio_queue.put(None)
            return
        try:
            while True:
                data = await websocket.receive()
                if data["type"] == "websocket.disconnect":
                    session_outcome = "client_disconnect"
                    break
                raw_bytes = data.get("bytes")
                if raw_bytes:
                    samples = (
                        np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    received_s += len(samples) / SAMPLING_RATE
                    metrics.stream_audio_seconds.inc(len(samples) / SAMPLING_RATE)
                    queued_samples += len(samples)
                    if queued_samples > max_queued_samples:
                        session_outcome = "backpressure"
                        metrics.rejected_total.labels(reason="stream_backpressure").inc()
                        await websocket.close(
                            code=1013,
                            reason=(
                                f"client queued more than "
                                f"{settings.stream_max_queue_seconds:.0f}s of "
                                "unprocessed audio (faster than the model can keep "
                                "up); use POST /v1/audio/transcriptions with "
                                "stream=true for bulk audio"
                            ),
                        )
                        break
                    await audio_queue.put(samples)
                    if time.monotonic() - session_started > settings.stream_max_session_seconds:
                        session_outcome = "max_duration"
                        break
                    continue
                text = data.get("text")
                if text:
                    try:
                        if json.loads(text).get("type") == "end":
                            session_outcome = "ended_by_client"
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass
        except WebSocketDisconnect:
            session_outcome = "client_disconnect"
        finally:
            # Aborted sessions (client gone / overloaded / too long) should not keep
            # the GPU busy decoding audio nobody will hear; an explicit ``end``
            # still drains the remainder so the tail of the utterance is kept.
            if session_outcome in {"backpressure", "client_disconnect", "max_duration"}:
                while not audio_queue.empty():
                    audio_queue.get_nowait()
                queued_samples = 0
            await audio_queue.put(None)

    async def _audio_gen() -> AsyncGenerator[np.ndarray, None]:
        nonlocal queued_samples
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                return
            queued_samples = max(0, queued_samples - len(chunk))
            yield chunk

    async def _transcribe(samples: np.ndarray) -> Any:
        try:
            # Streaming buffers are passed in memory; no temp file per inference.
            return await _run_pool(
                samples, len(samples) / SAMPLING_RATE, timestamps=True, **languages
            )
        except PoolError as exc:
            logfire.warning("streaming inference rejected: {error}", error=str(exc))
            return None
        except Exception as exc:
            logfire.exception("streaming inference error: {exc}", exc=str(exc))
            return None

    stats: dict[str, Any] = {}
    reader_task = asyncio.create_task(_reader())
    try:
        with logfire.span("transcribe_stream"):
            async for event in continuous_transcriber(
                _transcribe,
                _audio_gen(),
                min_duration=min_duration,
                retranscribe_interval=retranscribe_interval,
                stable_words=stable_words,
                stable_iters=stable_iters,
                max_duration=max_duration,
                context_duration=context_duration,
                trim_duration=trim_duration,
                holdback=holdback,
                stats=stats,
            ):
                if event.type == "committed":
                    metrics.stream_commits_total.inc()
                    commit_end = stats.get("commit_end")
                    if commit_end is not None:
                        metrics.stream_commit_lag_seconds.observe(
                            max(0.0, received_s - commit_end)
                        )
                elif event.type == "partial":
                    metrics.stream_partials_total.inc()
                await websocket.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        session_outcome = "client_disconnect"
    except Exception:
        session_outcome = "error"
        raise
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass
        except Exception as exc:  # e.g. starlette's WebSocketDisconnected
            logfire.debug("stream reader ended: {error}", error=str(exc))
        _active_streams = max(0, _active_streams - 1)
        metrics.stream_sessions_active.set(_active_streams)
        metrics.stream_sessions_total.labels(outcome=session_outcome).inc()
        metrics.stream_session_seconds.observe(time.monotonic() - session_started)
        # Closing again is a no-op when the reader already closed the socket.
        try:
            await websocket.close()
        except Exception:
            pass


# ── Static frontend ──────────────────────────────────────────────────────────
# Cache lifetimes by asset class. The Vite bundle has content-hashed names, so
# it is immutable; the VAD model/runtime are version-named but not hashed, so
# they get a week; the document itself must stay revalidated so a redeploy is
# picked up.
_IMMUTABLE_SUFFIXES = (".js", ".css", ".woff2", ".svg", ".png", ".ico")


def _cache_control(path: str) -> str:
    if path.startswith("assets/") and path.endswith(_IMMUTABLE_SUFFIXES):
        return "public, max-age=31536000, immutable"
    if path.endswith((".onnx", ".wasm", ".data")) or path.startswith("vad.worklet"):
        return "public, max-age=604800"
    if path.endswith(".html") or path == "":
        return "no-cache"
    return "public, max-age=3600"


class _FrontendStaticFiles(StaticFiles):
    """StaticFiles that ships precompressed siblings and cache headers.

    The VAD runtime is a ~14 MB wasm plus a 2 MB onnx model; both are fetched by
    every client the first time it opens a live session. Compressing them on the
    fly would burn ~1 s of the single event loop per cold request (and stall the
    WebSocket streams), so the image build writes ``<asset>.gz`` next to them and
    this class serves that file when the client accepts gzip.
    """

    async def get_response(self, path: str, scope: Any):
        response = await super().get_response(path, scope)
        origin_path = scope.get("path", "").lstrip("/") or "index.html"
        headers = {"cache-control": _cache_control(origin_path)}
        if response.status_code == 200 and not origin_path.endswith(".html"):
            headers["vary"] = "accept-encoding"

        accept = dict(scope["headers"]).get(b"accept-encoding", b"")
        if b"gzip" in accept and response.status_code == 200:
            gzipped = Path(self.directory) / f"{path}.gz"
            if gzipped.is_file():
                return FileResponse(
                    gzipped,
                    media_type=response.media_type,
                    headers={**headers, "content-encoding": "gzip"},
                )
        response.headers.update(headers)
        return response


# Serve the built frontend as static files when configured (production image).
# Mounted last so API routes (/v1, /health, /docs, ...) keep precedence; the
# catch-all only handles the SPA assets and index.html. Disabled in dev (empty
# STATIC_DIR), where the Vite dev server serves the UI instead.
if settings.static_dir:
    _static_path = Path(settings.static_dir)
    if _static_path.is_dir():
        app.mount(
            "/",
            _FrontendStaticFiles(directory=_static_path, html=True),
            name="frontend",
        )
        logfire.info("serving static frontend from {path}", path=str(_static_path))
    else:
        logfire.warning(
            "STATIC_DIR={path} is set but not a directory; static serving disabled",
            path=settings.static_dir,
        )
