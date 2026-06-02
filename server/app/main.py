"""FastAPI application exposing Parakeet/Canary."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import logfire
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect

from .audio import InvalidAudioError, float32_to_wav_path, to_wav16k_mono
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
