"""Native cache-aware streaming transcription over a live async audio stream.

The Nemotron model streams natively: audio is fed chunk by chunk through the
encoder while the encoder cache and the running RNN-T hypothesis are carried
forward between steps (see :class:`app.model_pool.StreamingSession`).  Each step
yields the full running transcript, which only ever grows — already-emitted
tokens are not revised.

This module adapts that to the WebSocket event protocol:

* ``partial`` – the current running transcript of the session (replaces the
  previously shown partial text as it grows).
* ``final``   – the complete transcript, emitted once the stream ends.

Incoming audio frames (which can be very short, e.g. a single VAD frame) are
coalesced to at least ``MIN_FLUSH_SECONDS`` before being handed to the model, so
the feature preprocessor always has enough samples and per-append boundary
effects stay infrequent.
"""
from __future__ import annotations

from typing import AsyncGenerator

import numpy as np

from .model_pool import StreamingSession
from .schemas import StreamEvent

SAMPLING_RATE = 16_000

# Minimum amount of audio to accumulate before running a streaming step.
MIN_FLUSH_SECONDS = 0.32
_MIN_FLUSH_SAMPLES = int(MIN_FLUSH_SECONDS * SAMPLING_RATE)


async def stream_transcribe(
    session: StreamingSession,
    audio_chunks: AsyncGenerator[np.ndarray, None],
    *,
    segment_id: int = 0,
) -> AsyncGenerator[StreamEvent, None]:
    """Drive native streaming for one session, yielding StreamEvents.

    Args:
        session: a started :class:`StreamingSession` bound to a pool worker.
        audio_chunks: async generator of float32 16 kHz mono sample arrays.
        segment_id: the ``id`` stamped on emitted events.
    """
    pending = np.empty(0, dtype=np.float32)
    last_emitted = ""

    async for chunk in audio_chunks:
        pending = np.concatenate((pending, chunk))
        if len(pending) < _MIN_FLUSH_SAMPLES:
            continue

        text = await session.add_audio(pending)
        pending = np.empty(0, dtype=np.float32)

        if text and text != last_emitted:
            last_emitted = text
            yield StreamEvent(type="partial", text=text, start=0.0, id=segment_id)

    # Flush any audio held back below the coalescing threshold.
    if len(pending) > 0:
        text = await session.add_audio(pending)
        if text and text != last_emitted:
            last_emitted = text
            yield StreamEvent(type="partial", text=text, start=0.0, id=segment_id)

    # End of stream: flush the encoder tail and commit the final transcript.
    final_text = await session.finalize()
    yield StreamEvent(type="final", text=final_text, start=0.0, id=segment_id)
