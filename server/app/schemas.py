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


class LanguageInfo(BaseModel):
    code: str
    name: str


class StreamEvent(BaseModel):
    """A single event emitted by the streaming transcription endpoint."""

    type: Literal["partial", "committed", "final"]
    text: str
    start: float
    id: int


class HealthResponse(BaseModel):
    status: str
    model: str
    workers: int
    ready: bool
    # Whether the configured model accepts source/target language selection,
    # and the list of supported languages (empty when unsupported).
    supports_languages: bool = False
    languages: list[LanguageInfo] = []
