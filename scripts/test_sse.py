#!/usr/bin/env python3
"""Test SSE streaming on the OpenAI-compatible transcription endpoint.

Uses the official OpenAI SDK, so it exercises the same wire format that OpenAI
clients (and a LiteLLM pass-through route) see.

Usage:
    python scripts/test_sse.py <audio file> [--base-url URL] [--api-key KEY]
                               [--model NAME] [--language CODE] [--no-stream]

Examples:
    # straight to the ASR server
    python scripts/test_sse.py sample.wav
    # through a LiteLLM proxy pass-through route (see docs/api.md)
    python scripts/test_sse.py sample.wav --base-url http://localhost:4000/parakeet/v1 --api-key sk-...

Requires:
    pip install openai
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    sys.exit("error: openai not installed — run: pip install openai")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream a transcription over SSE.")
    parser.add_argument("file", type=Path, help="audio file to transcribe")
    parser.add_argument("--base-url", default="http://localhost:9000/v1")
    parser.add_argument("--api-key", default="unused")
    parser.add_argument("--model", default="nvidia/canary-1b-v2")
    parser.add_argument("--language", default=None, help="source language, e.g. en, sk")
    parser.add_argument("--no-stream", action="store_true", help="plain JSON request")
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        return 1

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    extra = {"language": args.language} if args.language else {}
    started = time.monotonic()

    with args.file.open("rb") as fh:
        if args.no_stream:
            result = client.audio.transcriptions.create(model=args.model, file=fh, **extra)
            print(result.text)
            print(f"\n[{time.monotonic() - started:.2f}s]", file=sys.stderr)
            return 0

        stream = client.audio.transcriptions.create(
            model=args.model, file=fh, stream=True, **extra
        )
        for event in stream:
            elapsed = time.monotonic() - started
            if event.type == "transcript.text.delta":
                print(f"[{elapsed:6.2f}s delta] {event.delta!r}", file=sys.stderr)
            elif event.type == "transcript.text.done":
                print(f"[{elapsed:6.2f}s done ]", file=sys.stderr)
                print(event.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
