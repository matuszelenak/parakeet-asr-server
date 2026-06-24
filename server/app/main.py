"""FastAPI application exposing NVIDIA Nemotron streaming ASR."""
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

from .audio import InvalidAudioError, to_wav16k_mono
from .config import settings
from .languages import LANGUAGES, is_supported
from .model_pool import pool
from .schemas import (
    CharTimestamp,
    HealthResponse,
    LanguageInfo,
    SegmentTimestamp,
    TimestampedResponse,
    TranscriptionResponse,
    WordTimestamp,
)
from .streaming import stream_transcribe

logfire.configure(
    service_name="nemotron-asr-server",
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
    title="Nemotron ASR Server",
    version="0.2.0",
    description="Streaming speech recognition with NVIDIA Nemotron",
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


def _resolve_lang(target_lang: str | None) -> str:
    """Validate the requested language and resolve to a model prompt value."""
    if target_lang is not None and not is_supported(target_lang):
        raise HTTPException(
            status_code=400, detail=f"unsupported language: {target_lang}"
        )
    return target_lang or settings.target_lang


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if pool.ready else "loading",
        model=settings.model_name,
        workers=pool.size,
        ready=pool.ready,
        supports_languages=True,
        languages=[LanguageInfo(code=code, name=name) for code, name in LANGUAGES],
    )


@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    target_lang: str | None = Form(None),
) -> TranscriptionResponse:
    """Plain-text transcription of an uploaded audio file."""
    lang = _resolve_lang(target_lang)
    path = await _read_and_prepare(file)
    try:
        results = await pool.transcribe([path], target_lang=lang)
    finally:
        _cleanup(path)
    return TranscriptionResponse(text=_result_text(results[0]))


@app.post("/v1/transcribe/timestamps", response_model=TimestampedResponse)
async def transcribe_timestamps(
    file: UploadFile = File(...),
    target_lang: str | None = Form(None),
) -> TimestampedResponse:
    """Transcription with word, segment, and char level timestamps."""
    lang = _resolve_lang(target_lang)
    path = await _read_and_prepare(file)
    try:
        results = await pool.transcribe([path], timestamps=True, target_lang=lang)
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
    target_lang: str | None = Form(None),
) -> TranscriptionResponse:
    """Transcription of a long recording.

    The cache-aware streaming encoder handles arbitrarily long audio with a
    bounded memory footprint, so this is the same offline path as
    ``/v1/transcribe``; it is kept as a distinct endpoint for API compatibility.
    """
    lang = _resolve_lang(target_lang)
    path = await _read_and_prepare(file)
    try:
        results = await pool.transcribe([path], target_lang=lang)
    finally:
        _cleanup(path)
    return TranscriptionResponse(text=_result_text(results[0]))


@app.websocket("/v1/transcribe/stream")
async def transcribe_stream(
    websocket: WebSocket,
    target_lang: str | None = None,
) -> None:
    """Continuous streaming transcription over WebSocket (native cache-aware).

    Optional query param ``target_lang`` selects the language as a BCP-47 locale
    (e.g. ``?target_lang=de-DE``) or ``auto`` to auto-detect.  When omitted, the
    server default is used.

    The client sends raw PCM-16 mono 16 kHz audio as binary frames and signals
    end-of-stream with a text frame containing ``{"type": "end"}``.

    The server responds with JSON StreamEvent text frames::

        {"type": "partial", "text": "...", "start": 0.0, "id": 0}
        {"type": "final",   "text": "...", "start": 0.0, "id": 0}

    ``partial`` – the running transcript of the session (replaces as it grows).
    ``final``   – the complete transcript emitted after end-of-stream.

    One pool worker is held for the duration of the session, so the maximum
    number of concurrent streaming sessions equals the pool size.
    """
    lang = target_lang if (target_lang and is_supported(target_lang)) else settings.target_lang
    await websocket.accept()

    if not pool.ready:
        await websocket.close(code=1013, reason="model not ready")
        return

    audio_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

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

    reader_task = asyncio.create_task(_reader())
    try:
        with logfire.span("transcribe_stream", target_lang=lang):
            async with pool.stream_session(target_lang=lang) as session:
                async for event in stream_transcribe(session, _audio_gen()):
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
