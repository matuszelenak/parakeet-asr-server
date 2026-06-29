"""Response models for the transcription API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    text: str
    duration_seconds: float | None = None


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float


class SegmentTimestamp(BaseModel):
    segment: str
    start: float
    end: float


class CharTimestamp(BaseModel):
    char: str
    start: float
    end: float


class TimestampedResponse(BaseModel):
    text: str
    words: list[WordTimestamp] = []
    segments: list[SegmentTimestamp] = []
    chars: list[CharTimestamp] = []


class VerboseSegment(BaseModel):
    """A segment in an OpenAI ``verbose_json`` transcription response.

    Only ``id``/``start``/``end``/``text`` carry real data here; the remaining
    fields (``tokens``, ``avg_logprob``, ...) are part of OpenAI's schema but are
    not produced by NeMo, so they are emitted with neutral defaults so that
    standard OpenAI clients can parse the response without error.
    """

    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class VerboseTranscriptionResponse(BaseModel):
    """OpenAI ``verbose_json`` response shape for transcriptions/translations."""

    task: Literal["transcribe", "translate"] = "transcribe"
    language: str
    duration: float
    text: str
    words: list[WordTimestamp] = []
    segments: list[VerboseSegment] = []


class LanguageInfo(BaseModel):
    code: str
    name: str


class StreamEvent(BaseModel):
    """A single event emitted by the streaming transcription endpoint."""

    type: Literal["partial", "committed", "final"]
    text: str
    start: float
    id: int


class StreamConfig(BaseModel):
    """Optional per-session overrides sent by the client as the first WebSocket
    message: ``{"type": "configure", ...}``.  Any omitted field falls back to
    the server-side default from ``Settings``."""

    min_duration: float | None = None
    retranscribe_interval: float | None = None
    stable_words: int | None = None
    stable_iters: int | None = None
    max_duration: float | None = None
    context_duration: float | None = None


class HealthResponse(BaseModel):
    status: str
    model: str
    workers: int
    ready: bool
    # Whether the configured model accepts source/target language selection,
    # and the list of supported languages (empty when unsupported).
    supports_languages: bool = False
    languages: list[LanguageInfo] = []
