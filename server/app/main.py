"""FastAPI application exposing Parakeet/Canary."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

import logfire
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect

from .audio import (
    InvalidAudioError,
    float32_to_wav_path,
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
    SegmentTimestamp,
    StreamConfig,
    StreamEvent,
    TimestampedResponse,
    TranscriptionResponse,
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


async def _read_and_prepare(upload: UploadFile) -> str:
    """Validate the upload size, decode it, and return a temp 16k mono WAV path."""
    raw = await upload.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds {settings.max_upload_mb} MB limit",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        return to_wav16k_mono(raw)
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
    source_lang: Language | None, target_lang: Language | None
) -> dict[str, str]:
    """Validate language args against the configured model's capabilities.

    Returns kwargs (``source_lang`` / ``target_lang``) to pass to the model, or
    an empty dict when the model does not use language selection.
    """
    if not model_supports_languages(settings.model_name):
        if source_lang is not None or target_lang is not None:
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
        return Language(code.lower())
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


async def _openai_transcribe(
    *,
    file: UploadFile,
    source: Language | None,
    target: Language | None,
    response_format: str,
    timestamp_granularities: list[str],
    longform: bool,
    task: Literal["transcribe", "translate"],
) -> Response:
    """Shared core for the OpenAI-compatible transcription/translation endpoints."""
    if response_format not in _RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported response_format '{response_format}'; expected one of "
                f"{', '.join(sorted(_RESPONSE_FORMATS))}"
            ),
        )

    languages = _resolve_languages(source, target)
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
    """
    source = _parse_language(language)
    target = _parse_language(target_language) or source
    granularities = timestamp_granularities or timestamp_granularities_alt or []
    return await _openai_transcribe(
        file=file,
        source=source,
        target=target,
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
    source = _parse_language(language) or DEFAULT_LANGUAGE
    return await _openai_transcribe(
        file=file,
        source=source,
        target=Language.en,
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
