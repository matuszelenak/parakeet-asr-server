"""Continuous transcription over a live async audio stream.

Parakeet/Canary do not support streaming inference, so this module
approximates it by repeatedly re-transcribing an accumulating buffer and
committing the leading word-prefix once it has appeared identically across
`stable_iters` consecutive inference runs.

Key design choices:
- A look-back context window (``context_duration`` seconds of already-committed
  audio) is prepended to every inference call so the model never starts cold at
  a buffer boundary.  This eliminates the class of errors where the first few
  words of a new segment are mis-transcribed due to missing acoustic context.
- Re-transcription is rate-limited by ``retranscribe_interval`` to avoid GPU
  overload on every 250 ms chunk.
- A hard ``max_duration`` cap forces a commit when the buffer grows too large.
- Stability detection uses a bounded deque and takes the minimum matching-
  prefix length across all entries in the window.
"""
from __future__ import annotations

from collections import deque
from typing import Any, AsyncGenerator, Awaitable, Callable, Deque, List

import numpy as np

from .schemas import StreamEvent

SAMPLING_RATE = 16_000

Word = dict  # {"word": str, "start": float, "end": float}


def _strip_context(
    all_words: List[Word], ctx_secs: float
) -> List[Word]:
    """Remove words that fall entirely inside the context window and shift
    timestamps so they are relative to the current buffer start (t=0)."""
    return [
        {**w, "start": w["start"] - ctx_secs, "end": w["end"] - ctx_secs}
        for w in all_words
        if w["end"] > ctx_secs
    ]


async def continuous_transcriber(
    transcribe_fn: Callable[[np.ndarray], Awaitable[Any]],
    audio_chunks: AsyncGenerator[np.ndarray, None],
    *,
    min_duration: float,
    retranscribe_interval: float,
    stable_words: int,
    stable_iters: int,
    max_duration: float,
    context_duration: float = 2.0,
) -> AsyncGenerator[StreamEvent, None]:
    """Yield StreamEvent objects as speech is transcribed incrementally.

    Args:
        transcribe_fn: async callable that accepts a float32 16 kHz mono array
            and returns a NeMo Hypothesis (or None on error).
        audio_chunks: async generator of float32 sample arrays.
        min_duration: seconds of audio required before the first inference.
        retranscribe_interval: minimum seconds of *new* audio required between
            consecutive inference runs.
        stable_words: minimum prefix length (in words) that must match across
            ``stable_iters`` runs to trigger a commit.
        stable_iters: number of consecutive identical-prefix runs needed to
            commit a segment.
        max_duration: buffer cap in seconds; triggers a forced commit.
        context_duration: seconds of previously committed audio to prepend as
            acoustic context on each inference call (0 disables the feature).
    """
    buffer = np.empty(0, dtype=np.float32)
    # Rolling window of recently committed audio used as look-back context.
    context_audio = np.empty(0, dtype=np.float32)
    segment_id = 0
    ts_offset = 0.0
    last_committed_word = ""

    history: Deque[List[Word]] = deque(maxlen=stable_iters + 1)
    last_words: List[Word] = []

    samples_at_last_run = 0
    min_samples = int(min_duration * SAMPLING_RATE)
    interval_samples = int(retranscribe_interval * SAMPLING_RATE)
    max_samples = int(max_duration * SAMPLING_RATE)
    context_samples = int(context_duration * SAMPLING_RATE)

    def _run_inference_input() -> tuple[np.ndarray, float]:
        """Return (inference_array, context_seconds) for the current state."""
        if context_samples > 0 and len(context_audio) > 0:
            return np.concatenate((context_audio, buffer)), len(context_audio) / SAMPLING_RATE
        return buffer, 0.0

    def _advance_context(advance: int) -> None:
        """Roll the context window forward after committing ``advance`` samples."""
        nonlocal context_audio
        if context_samples <= 0:
            return
        committed_chunk = buffer[:advance]
        combined = np.concatenate((context_audio, committed_chunk))
        context_audio = combined[-context_samples:] if len(combined) >= context_samples else combined.copy()

    async for chunk in audio_chunks:
        buffer = np.concatenate((buffer, chunk))

        over_limit = len(buffer) > max_samples
        enough_new = (len(buffer) - samples_at_last_run) >= interval_samples
        long_enough = len(buffer) >= min_samples

        if not over_limit and not (long_enough and enough_new):
            continue

        # ── Run inference ────────────────────────────────────────────────────
        inference_input, ctx_secs = _run_inference_input()
        samples_at_last_run = len(buffer)
        result = await transcribe_fn(inference_input)
        all_words: List[Word] = (
            (result.timestamp or {}).get("word", []) if result is not None else []
        )

        # Remove context-window words and re-anchor timestamps to buffer start.
        words = _strip_context(all_words, ctx_secs)

        # Drop the leading boundary duplicate (last word of previous segment
        # occasionally re-appears as the first word when context is thin).
        if (
            words
            and last_committed_word
            and words[0]["word"].strip().lower() == last_committed_word.strip().lower()
        ):
            words = words[1:]

        last_words = words

        # ── Forced commit (buffer overflow) ──────────────────────────────────
        if over_limit:
            if words:
                yield StreamEvent(
                    type="committed",
                    text=" ".join(w["word"] for w in words),
                    start=ts_offset,
                    id=segment_id,
                )
                advance = min(int(words[-1]["end"] * SAMPLING_RATE), len(buffer))
                last_committed_word = words[-1]["word"]
            else:
                advance = min(max_samples // 2, len(buffer))

            _advance_context(advance)
            buffer = buffer[advance:]
            ts_offset += advance / SAMPLING_RATE
            segment_id += 1
            history.clear()
            last_words = []
            samples_at_last_run = len(buffer)
            continue

        if not words:
            continue

        history.append(words)

        # ── Stability check ──────────────────────────────────────────────────
        if len(history) >= stable_iters:
            recent = list(history)[-stable_iters:]
            reference = recent[-1]

            max_match = len(reference)
            for older in recent[:-1]:
                i = 0
                while (
                    i < min(max_match, len(older))
                    and reference[i]["word"].strip().lower()
                    == older[i]["word"].strip().lower()
                ):
                    i += 1
                max_match = i
                if max_match < stable_words:
                    break

            if max_match >= stable_words:
                committed = reference[:max_match]
                yield StreamEvent(
                    type="committed",
                    text=" ".join(w["word"] for w in committed),
                    start=ts_offset,
                    id=segment_id,
                )
                advance = min(int(committed[-1]["end"] * SAMPLING_RATE), len(buffer))
                _advance_context(advance)
                ts_offset += advance / SAMPLING_RATE
                buffer = buffer[advance:]
                segment_id += 1
                last_committed_word = committed[-1]["word"]
                residual = reference[max_match:]
                history.clear()
                if residual:
                    history.append(residual)
                last_words = residual
                samples_at_last_run = len(buffer)
                continue

        # ── Partial update ───────────────────────────────────────────────────
        yield StreamEvent(
            type="partial",
            text=" ".join(w["word"] for w in words),
            start=ts_offset,
            id=segment_id,
        )

    # ── Stream ended: emit final ─────────────────────────────────────────────
    if len(buffer) > 0 and (len(buffer) - samples_at_last_run) >= int(
        0.1 * SAMPLING_RATE
    ):
        inference_input, ctx_secs = _run_inference_input()
        result = await transcribe_fn(inference_input)
        final_words: List[Word] = (
            (result.timestamp or {}).get("word", []) if result is not None else []
        )
        final_words = _strip_context(final_words, ctx_secs)
        if (
            final_words
            and last_committed_word
            and final_words[0]["word"].strip().lower() == last_committed_word.strip().lower()
        ):
            final_words = final_words[1:]
        last_words = final_words

    if last_words:
        yield StreamEvent(
            type="final",
            text=" ".join(w["word"] for w in last_words),
            start=ts_offset,
            id=segment_id,
        )
