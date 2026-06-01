#!/usr/bin/env python3
"""Test the streaming WebSocket transcription endpoint.

Usage:
    python scripts/test_stream.py <audio.wav> [options]

Options:
    --host WS_URL     WebSocket base URL (default: ws://localhost:9000)
    --chunk-ms MS     Audio chunk size in milliseconds (default: 500)
    --realtime        Simulate real-time playback (sleep between chunks)

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

EVENT_MARKER = {
    "partial": "~",
    "committed": "+",
    "final": "!",
}

_YELLOW = "\033[33m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

EVENT_COLOR = {
    "partial": _YELLOW,
    "committed": _GREEN,
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
            sys.exit("error: torchaudio is required to resample this file — pip install torchaudio")

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
    source_lang: str | None,
    target_lang: str | None,
) -> int:
    mono = _load_audio(wav_path)
    duration = len(mono) / SAMPLING_RATE

    pcm16 = (mono * 32767).clip(-32768, 32767).astype(np.int16)
    chunk_samples = int(SAMPLING_RATE * chunk_ms / 1000)

    params: list[str] = []
    if source_lang:
        params.append(f"source_lang={source_lang}")
    if target_lang:
        params.append(f"target_lang={target_lang}")
    qs = ("?" + "&".join(params)) if params else ""

    uri = f"{host.rstrip('/')}/v1/transcribe/stream{qs}"
    print(f"Connecting to {uri}")
    lang_info = f"  |  {source_lang or 'en'} → {target_lang or source_lang or 'en'}" if (source_lang or target_lang) else ""
    print(f"Audio: {duration:.1f}s  |  chunk size: {chunk_ms} ms  |  realtime: {realtime}{lang_info}")
    print()

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
                        raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    except asyncio.TimeoutError:
                        print("error: timed out waiting for a server response", file=sys.stderr)
                        return 1

                    event = json.loads(raw)
                    ev_type = event.get("type", "?")
                    text = event.get("text", "")
                    start_s = event.get("start", 0.0)
                    seg_id = event.get("id", 0)
                    marker = EVENT_MARKER.get(ev_type, "?")
                    color = EVENT_COLOR.get(ev_type, "")

                    # if ev_type != 'committed':
                    #     continue

                    print(
                        f"{color}[{ev_type:9s}] [{marker}] id={seg_id}  start={start_s:.2f}s  {text!r}{_RESET}"
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
        "--source-lang",
        metavar="LANG",
        default=None,
        help="Source language code, e.g. sk, de, fr (default: en)",
    )
    parser.add_argument(
        "--target-lang",
        metavar="LANG",
        default=None,
        help="Target language code for translation (default: same as source)",
    )
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        return 1

    return asyncio.run(run(args.file, args.host, args.chunk_ms, args.realtime, args.source_lang, args.target_lang))


if __name__ == "__main__":
    raise SystemExit(main())
