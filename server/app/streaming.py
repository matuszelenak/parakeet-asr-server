"""Continuous transcription over a live async audio stream.

Parakeet/Canary do not support streaming inference, so this module
approximates it with *local agreement*: the audio buffer is re-transcribed as
it grows, and the leading words on which the last ``stable_iters`` hypotheses
agree are committed.

Key design choices:
- The whole buffer is re-transcribed on every run (not just the audio after
  the last commit), so every word is decoded with the full current sentence
  as context. The buffer is only trimmed once it exceeds ``trim_duration``,
  and then at the end of the last committed *sentence* (a natural pause),
  keeping ``context_duration`` seconds before the cut as acoustic context.
  ``max_duration`` is a hard cap that trims at the last committed word.
- Already-committed words are removed from each hypothesis by time (words
  starting before the last commit) and by n-gram overlap with the committed
  tail, including re-decoded variants of the same word ("andrej"/"andreja").
- Agreement is checked on normalised words (case- and punctuation-insensitive)
  so "veľa," vs "veľa." does not hold back a commit.
- Words ending within ``holdback`` seconds of the buffer end are never
  committed: they may be only partially spoken, and committing them early is
  the main source of errors at short re-transcription intervals.
- Re-transcription is rate-limited by ``retranscribe_interval``.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, AsyncGenerator, Awaitable, Callable

import numpy as np

from .schemas import StreamEvent

SAMPLING_RATE = 16_000

Word = dict  # {"word": str, "n": str, "start": float, "end": float}, stream time

_NON_WORD = re.compile(r"[^\w]", re.UNICODE)
_SENTENCE_END = re.compile(r"[.?!…]['\"»”)]*$")

# A word starting this much before the last commit's end is still treated as
# already committed (word timestamps are imprecise).
_DEDUP_TOLERANCE = 0.2
# Longest committed-tail / hypothesis-head overlap removed, in words.
_MAX_OVERLAP = 5


def _norm(word: str) -> str:
    return _NON_WORD.sub("", unicodedata.normalize("NFC", word).lower())


def _same_word(a: str, b: str) -> bool:
    """Same normalised word, or a re-decoded variant sharing its stem."""
    if a == b:
        return True
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return prefix >= 3 and prefix >= 0.6 * min(len(a), len(b))


def _agreed_prefix(hypotheses: list[list[Word]]) -> int:
    """Length of the common (normalised) prefix of all hypotheses."""
    n = min(len(h) for h in hypotheses)
    i = 0
    while i < n and all(h[i]["n"] == hypotheses[0][i]["n"] for h in hypotheses[1:]):
        i += 1
    return i


def _text(words: list[Word]) -> str:
    return " ".join(w["word"] for w in words)


async def continuous_transcriber(
    transcribe_fn: Callable[[np.ndarray], Awaitable[Any]],
    audio_chunks: AsyncGenerator[np.ndarray, None],
    *,
    min_duration: float,
    retranscribe_interval: float,
    stable_words: int,
    stable_iters: int,
    max_duration: float,
    context_duration: float = 1.5,
    trim_duration: float = 10.0,
    holdback: float = 0.0,
) -> AsyncGenerator[StreamEvent, None]:
    """Yield StreamEvent objects as speech is transcribed incrementally.

    Args:
        transcribe_fn: async callable that accepts a float32 16 kHz mono array
            and returns a NeMo Hypothesis with word timestamps (or None on
            error).
        audio_chunks: async generator of float32 sample arrays.
        min_duration: seconds of audio required before the first inference.
        retranscribe_interval: minimum seconds of *new* audio required between
            consecutive inference runs.
        stable_words: minimum number of agreed words needed to commit.
        stable_iters: number of consecutive hypotheses that must agree.
        max_duration: hard buffer cap in seconds; forces a trim (and a commit,
            if nothing has been committed in the buffer).
        context_duration: seconds of already-committed audio kept before a
            trim point as acoustic context.
        trim_duration: buffer length after which it is trimmed at the end of
            the last committed sentence.
        holdback: words ending within this many seconds of the buffer end are
            not committed yet.
    """
    buffer = np.empty(0, dtype=np.float32)
    buf_offset = 0.0  # stream time of buffer[0]
    committed: list[Word] = []
    commit_end = 0.0  # stream time at which the last committed word ends
    history: list[list[Word]] = []
    tail: list[Word] = []  # current uncommitted words
    segment_id = 0
    samples_at_last_run = 0
    stable_iters = max(1, stable_iters)
    stable_words = max(1, stable_words)
    min_samples = int(min_duration * SAMPLING_RATE)
    interval_samples = int(retranscribe_interval * SAMPLING_RATE)

    async def hypothesis() -> list[Word]:
        """Transcribe the buffer; return the uncommitted words (stream time)."""
        result = await transcribe_fn(buffer)
        stamps = (getattr(result, "timestamp", None) or {}).get("word", []) if result is not None else []
        words = [
            {"word": w["word"], "n": _norm(w["word"]),
             "start": w["start"] + buf_offset, "end": w["end"] + buf_offset}
            for w in stamps
        ]
        words = [w for w in words if w["start"] >= commit_end - _DEDUP_TOLERANCE]
        if committed and words:
            done = [w["n"] for w in committed[-_MAX_OVERLAP:]]
            head = [w["n"] for w in words]
            for k in range(min(len(done), len(head)), 0, -1):
                if all(_same_word(a, b) for a, b in zip(done[-k:], head[:k])):
                    words = words[k:]
                    break
        return words

    def commit(words: list[Word]) -> StreamEvent:
        nonlocal commit_end, segment_id
        committed.extend(words)
        commit_end = words[-1]["end"]
        event = StreamEvent(type="committed", text=_text(words), start=words[0]["start"], id=segment_id)
        segment_id += 1
        return event

    def trim(force: bool) -> None:
        """Drop audio before the last committed sentence end (or word, if forced)."""
        nonlocal buffer, buf_offset, samples_at_last_run
        cut = None
        for w in reversed(committed):
            if w["end"] <= buf_offset + context_duration:
                break
            if _SENTENCE_END.search(w["word"]):
                cut = w["end"]
                break
        if cut is None and force and committed and committed[-1]["end"] > buf_offset:
            cut = committed[-1]["end"]
        if cut is None:
            return
        new_offset = max(buf_offset, cut - context_duration)
        drop = int(round((new_offset - buf_offset) * SAMPLING_RATE))
        if drop <= 0:
            return
        buffer = buffer[drop:]
        buf_offset += drop / SAMPLING_RATE
        samples_at_last_run = max(0, samples_at_last_run - drop)

    async for chunk in audio_chunks:
        buffer = np.concatenate((buffer, chunk))
        duration = len(buffer) / SAMPLING_RATE
        over_limit = duration > max_duration
        if not over_limit and (
            len(buffer) < min_samples or len(buffer) - samples_at_last_run < interval_samples
        ):
            continue

        # ── Run inference ────────────────────────────────────────────────────
        samples_at_last_run = len(buffer)
        words = await hypothesis()
        history = (history + [words])[-stable_iters:]

        # ── Commit the agreed prefix ─────────────────────────────────────────
        if len(history) == stable_iters:
            n = _agreed_prefix(history)
            edge = buf_offset + duration - holdback
            while n > 0 and words[n - 1]["end"] > edge:
                n -= 1
            if n >= stable_words:
                yield commit(words[:n])
                history = [h[n:] for h in history]
            words = history[-1]

        # ── Trim the buffer ──────────────────────────────────────────────────
        if over_limit:
            if words and (not committed or committed[-1]["end"] <= buf_offset + context_duration):
                # Nothing committed in this buffer: force-commit all but the last word.
                forced = words[:-1] if len(words) > 1 else words
                yield commit(forced)
                history = []
                words = words[len(forced):]
            trim(force=True)
        elif duration > trim_duration:
            trim(force=False)

        # ── Partial update ───────────────────────────────────────────────────
        tail = words
        if tail:
            yield StreamEvent(type="partial", text=_text(tail), start=tail[0]["start"], id=segment_id)

    # ── Stream ended: re-transcribe any unseen audio, emit the rest ──────────
    if len(buffer) >= int(0.1 * SAMPLING_RATE) and (
        not history or len(buffer) - samples_at_last_run >= int(0.1 * SAMPLING_RATE)
    ):
        tail = await hypothesis()

    # Always yield a final event to signal stream completion
    yield StreamEvent(
        type="final",
        text=_text(tail),
        start=tail[0]["start"] if tail else commit_end,
        id=segment_id,
    )
