#!/usr/bin/env python3
"""Test the plain-text transcription endpoint.

Usage:
    python scripts/test_transcribe.py <audio.wav> [--host http://localhost:9000]

Uses only the standard library, so no extra dependencies are required.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def build_multipart(file_path: Path, fields: dict[str, str]) -> tuple[bytes, str]:
    """Encode a file (field 'file') plus extra text fields as multipart/form-data."""
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(file_path.name)[0] or "audio/wav"

    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )

    file_head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()

    body = b"".join(parts) + file_head + file_path.read_bytes() + tail
    return body, boundary


def main() -> int:
    parser = argparse.ArgumentParser(description="Test plain-text ASR transcription.")
    parser.add_argument("file", type=Path, help="WAV file to transcribe")
    parser.add_argument(
        "--host",
        default="http://localhost:9000",
        help="API base URL (default: http://localhost:9000)",
    )
    parser.add_argument(
        "--target-lang",
        metavar="LANG",
        default=None,
        help="BCP-47 locale to transcribe/translate into, e.g. de-DE, sk-SK, "
        "or 'auto' (default: server default)",
    )
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        return 1

    fields: dict[str, str] = {}
    if args.target_lang:
        fields["target_lang"] = args.target_lang

    url = f"{args.host.rstrip('/')}/v1/transcribe"
    body, boundary = build_multipart(args.file, fields)

    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    print(f"POST {url}  (file: {args.file})")
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"request failed ({exc.code}): {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"could not reach {url}: {exc.reason}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
