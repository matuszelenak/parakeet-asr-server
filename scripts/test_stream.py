#!/usr/bin/env python3
"""Test the streaming WebSocket transcription endpoint.

Usage:
    python scripts/test_stream.py <audio.wav> [options]

Options:
    --host WS_URL       WebSocket base URL (default: ws://localhost:9000)
    --chunk-ms MS       Audio chunk size in milliseconds (default: 500)
    --realtime          Simulate real-time playback (sleep between chunks)
    --target-lang LANG  BCP-47 locale to transcribe/translate into, e.g.
                        de-DE, sk-SK, or "auto" (default: server default)
    --raw               Print every event on its own line instead of updating
                        the running partial in place

Requires:
    pip install websockets soundfile numpy
    # torchaudio is only needed when the WAV sample rate is not already 16 kHz
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import websockets
except ImportError:
    sys.exit("error: websockets not installed — run: pip install websockets")

SAMPLING_RATE = 16_000

# The streaming endpoint emits two event types:
#   partial - the running transcript of the session; it *replaces* the previous
#             partial as it grows (it is not an incremental delta).
#   final   - the complete transcript, sent once after end-of-stream.
EVENT_MARKER = {
    "partial": "~",
    "final": "!",
}

_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
_CLEAR_EOL = "\033[K"

EVENT_COLOR = {
    "partial": _YELLOW,
    "final": _CYAN,
}


def _load_audio(wav_path: Path) -> np.ndarray:
    """Read a WAV file and return a float32 16 kHz mono array."""
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)

    if sr != SAMPLING_RATE:
        print(
            f"warning: resampling from {sr} Hz to {SAMPLING_RATE} Hz (requires torchaudio)",
            file=sys.stderr,
        )
        try:
            import torch
            import torchaudio

            tensor = torch.from_numpy(mono).unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, sr, SAMPLING_RATE)
            mono = tensor.squeeze(0).numpy()
        except ImportError:
            sys.exit(
                "error: torchaudio is required to resample this file — pip install torchaudio"
            )

    # Guard against values outside [-1, 1] after potential downmix clipping.
    peak = float(np.max(np.abs(mono)))
    if peak > 1.0:
        mono = mono / peak

    return mono


async def run(
    wav_path: Path,
    host: str,
    chunk_ms: int,
    realtime: bool,
    target_lang: str | None,
    raw: bool,
) -> int:
    mono = _load_audio(wav_path)
    duration = len(mono) / SAMPLING_RATE

    pcm16 = (mono * 32767).clip(-32768, 32767).astype(np.int16)
    chunk_samples = int(SAMPLING_RATE * chunk_ms / 1000)

    qs = f"?target_lang={target_lang}" if target_lang else ""

    uri = f"{host.rstrip('/')}/v1/transcribe/stream{qs}"
    print(f"Connecting to {uri}")
    lang_info = f"  |  target_lang: {target_lang}" if target_lang else ""
    print(
        f"Audio: {duration:.1f}s  |  chunk size: {chunk_ms} ms  |  realtime: {realtime}{lang_info}"
    )
    print()

    # Render mode: with `--raw` each event is logged on its own line; otherwise
    # the running partial is rewritten in place (it replaces, not appends).
    live = not raw and sys.stdout.isatty()

    try:
        async with websockets.connect(uri) as ws:

            async def _send() -> None:
                for start in range(0, len(pcm16), chunk_samples):
                    chunk = pcm16[start : start + chunk_samples]
                    await ws.send(chunk.tobytes())
                    if realtime:
                        await asyncio.sleep(chunk_ms / 1000)
                await ws.send(json.dumps({"type": "end"}))

            sender = asyncio.create_task(_send())

            try:
                while True:
                    try:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    except asyncio.TimeoutError:
                        print(
                            "error: timed out waiting for a server response",
                            file=sys.stderr,
                        )
                        return 1

                    event = json.loads(raw_msg)
                    ev_type = event.get("type", "?")
                    text = event.get("text", "")
                    seg_id = event.get("id", 0)
                    marker = EVENT_MARKER.get(ev_type, "?")
                    color = EVENT_COLOR.get(ev_type, "")

                    # print(event)

                    if live and ev_type == "partial":
                        # Overwrite the current line with the latest running text.
                        print(
                            f"\r{color}[~] {text}{_RESET}{_CLEAR_EOL}",
                            end="",
                            flush=True,
                        )
                    elif live and ev_type == "final":
                        # Clear the live partial line, then print the final result.
                        print(f"\r{_CLEAR_EOL}", end="")
                        print(f"{color}[final] {text!r}{_RESET}")
                    else:
                        print(
                            f"{color}[{ev_type:8s}] [{marker}] id={seg_id}  {text!r}{_RESET}"
                        )

                    if ev_type == "final":
                        break
            finally:
                sender.cancel()

    except OSError as exc:
        print(f"error: could not connect to {uri}: {exc}", file=sys.stderr)
        return 1

    print()
    print("Stream complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream a WAV file to the ASR WebSocket endpoint and print events."
    )
    parser.add_argument("file", type=Path, help="WAV file to stream")
    parser.add_argument(
        "--host",
        default="ws://localhost:9000",
        help="WebSocket base URL (default: ws://localhost:9000)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=500,
        metavar="MS",
        help="Audio chunk size in milliseconds (default: 500)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep between chunks to simulate real-time playback",
    )
    parser.add_argument(
        "--target-lang",
        metavar="LANG",
        default=None,
        help="BCP-47 locale to transcribe/translate into, e.g. de-DE, sk-SK, "
        "or 'auto' (default: server default)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print every event on its own line instead of updating the partial in place",
    )
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        return 1

    return asyncio.run(
        run(
            args.file,
            args.host,
            args.chunk_ms,
            args.realtime,
            args.target_lang,
            args.raw,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
