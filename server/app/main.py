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
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
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
    to_wav16k_mono,
    wav_duration_seconds,
)
from .config import settings
from .languages import (
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    Language,
    model_supports_languages,
)
from .model_pool import pool
from .schemas import (
    CharTimestamp,
    HealthResponse,
    LanguageInfo,
    ModelCard,
    ModelList,
    OpenAIError,
    OpenAIErrorBody,
    SegmentTimestamp,
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
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {settings.max_upload_mb} MB limit",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    return raw


async def _read_and_prepare(upload: UploadFile) -> str:
    """Validate the upload size, decode it, and return a temp 16k mono WAV path."""
    raw = await _read_upload(upload)
    try:
        # Decoding/resampling is CPU-bound; keep it off the event loop.
        return await asyncio.to_thread(to_wav16k_mono, raw)
    except InvalidAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _cleanup(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _result_text(result: Any) -> str:
    return getattr(result, "text", "") or ""


def _resolve_languages(
    source_lang: Language | None,
    target_lang: Language | None,
    *,
    strict: bool = True,
) -> dict[str, str]:
    """Validate language args against the configured model's capabilities.

    Returns kwargs (``source_lang`` / ``target_lang``) to pass to the model, or
    an empty dict when the model does not use language selection. With
    ``strict=False`` languages are silently ignored for such models (OpenAI
    clients routinely send ``language`` as a hint).
    """
    if not model_supports_languages(settings.model_name):
        if strict and (source_lang is not None or target_lang is not None):
            raise HTTPException(
                status_code=400,
                detail="the configured model does not support language selection",
            )
        return {}

    # Canary-style models require both; default to English when omitted.
    src = source_lang or DEFAULT_LANGUAGE
    tgt = target_lang or source_lang or DEFAULT_LANGUAGE
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


def _extract_words(result: Any) -> list[WordTimestamp]:
    stamps = getattr(result, "timestamp", None) or {}
    return [
        WordTimestamp(word=w.get("word", ""), start=w["start"], end=w["end"])
        for w in stamps.get("word", [])
    ]


def _extract_verbose_segments(result: Any) -> list[VerboseSegment]:
    stamps = getattr(result, "timestamp", None) or {}
    return [
        VerboseSegment(
            id=i,
            start=s["start"],
            end=s["end"],
            text=s.get("segment", ""),
        )
        for i, s in enumerate(stamps.get("segment", []))
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
                    path = float32_to_wav_path(chunk)
                    try:
                        results = await pool.transcribe(
                            [path], longform=longform, **languages
                        )
                    finally:
                        _cleanup(path)
                    text = _result_text(results[0]).strip()
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

    path = await _read_and_prepare(file)
    try:
        duration = wav_duration_seconds(path)
        results = await pool.transcribe(
            [path], timestamps=want_timestamps, longform=longform, **languages
        )
    finally:
        _cleanup(path)

    result = results[0]
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
        model=settings.model_name,
        workers=pool.size,
        ready=pool.ready,
        supports_languages=supports,
        languages=languages,
    )


@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    source_lang: Language | None = Form(None),
    target_lang: Language | None = Form(None),
) -> TranscriptionResponse:
    """Plain-text transcription (or translation when target differs from source)."""
    languages = _resolve_languages(source_lang, target_lang)
    path = await _read_and_prepare(file)
    try:
        results = await pool.transcribe([path], **languages)
    finally:
        _cleanup(path)
    return TranscriptionResponse(text=_result_text(results[0]))


@app.post("/v1/transcribe/timestamps", response_model=TimestampedResponse)
async def transcribe_timestamps(
    file: UploadFile = File(...),
    source_lang: Language | None = Form(None),
    target_lang: Language | None = Form(None),
) -> TimestampedResponse:
    """Transcription with word, segment, and char level timestamps."""
    languages = _resolve_languages(source_lang, target_lang)
    path = await _read_and_prepare(file)
    try:
        results = await pool.transcribe([path], timestamps=True, **languages)
    finally:
        _cleanup(path)

    result = results[0]
    stamps = getattr(result, "timestamp", None) or {}

    words = [
        WordTimestamp(word=w.get("word", ""), start=w["start"], end=w["end"])
        for w in stamps.get("word", [])
    ]
    segments = [
        SegmentTimestamp(segment=s.get("segment", ""), start=s["start"], end=s["end"])
        for s in stamps.get("segment", [])
    ]
    chars = [
        CharTimestamp(char=c.get("char", ""), start=c["start"], end=c["end"])
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
    path = await _read_and_prepare(file)
    try:
        results = await pool.transcribe([path], longform=True, **languages)
    finally:
        _cleanup(path)
    return TranscriptionResponse(text=_result_text(results[0]))


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

    One pool worker is occupied for the duration of the session, so the maximum
    number of concurrent streaming sessions equals the pool size.
    """
    languages = _resolve_languages(source_lang, target_lang)
    await websocket.accept()

    if not pool.ready:
        await websocket.close(code=1013, reason="model not ready")
        return

    audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    # --- Read optional per-session configure message ----------------------------
    # The client MAY send {"type": "configure", ...overrides} as its very first
    # frame.  We wait up to 2 s; if no text frame arrives (or it isn't a
    # configure message) we fall through using server defaults.  A binary audio
    # frame that arrives before any configure message is decoded and queued so
    # no audio is lost.
    cfg = StreamConfig()
    try:
        first = await asyncio.wait_for(websocket.receive(), timeout=2.0)
        if first.get("text"):
            try:
                msg = json.loads(first["text"])
                if msg.get("type") == "configure":
                    cfg = StreamConfig.model_validate(
                        {k: v for k, v in msg.items() if k != "type"}
                    )
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
        try:
            while True:
                data = await websocket.receive()
                if data["type"] == "websocket.disconnect":
                    break
                raw_bytes = data.get("bytes")
                if raw_bytes:
                    samples = (
                        np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    await audio_queue.put(samples)
                    continue
                text = data.get("text")
                if text:
                    try:
                        if json.loads(text).get("type") == "end":
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass
        except WebSocketDisconnect:
            pass
        finally:
            await audio_queue.put(None)

    async def _audio_gen() -> AsyncGenerator[np.ndarray, None]:
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                return
            yield chunk

    async def _transcribe(samples: np.ndarray) -> Any:
        path = float32_to_wav_path(samples)
        try:
            results = await pool.transcribe([path], timestamps=True, **languages)
            return results[0]
        except Exception as exc:
            logfire.exception("streaming inference error: {exc}", exc=str(exc))
            return None
        finally:
            _cleanup(path)

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
            ):
                await websocket.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass


# Serve the built frontend as static files when configured (production image).
# Mounted last so API routes (/v1, /health, /docs, ...) keep precedence; the
# catch-all only handles the SPA assets and index.html. Disabled in dev (empty
# STATIC_DIR), where the Vite dev server serves the UI instead.
if settings.static_dir:
    _static_path = Path(settings.static_dir)
    if _static_path.is_dir():
        app.mount(
            "/", StaticFiles(directory=_static_path, html=True), name="frontend"
        )
        logfire.info("serving static frontend from {path}", path=str(_static_path))
    else:
        logfire.warning(
            "STATIC_DIR={path} is set but not a directory; static serving disabled",
            path=settings.static_dir,
        )
