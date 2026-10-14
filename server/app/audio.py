"""Audio ingestion: accept arbitrary audio uploads, normalise to 16 kHz mono.

NeMo's Parakeet models expect 16 kHz mono audio. Uploaded files may use any
container, sample rate, bit depth, or channel count, so we decode with
libsndfile (WAV/FLAC/OGG/MP3), falling back to ffmpeg for everything else
(M4A/MP4/WebM/...), downmix to mono, resample, and write a temporary 16-bit PCM
WAV that the model can read.
"""
from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000

_FFMPEG = shutil.which("ffmpeg")


class InvalidAudioError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as audio."""


def float32_to_wav_path(samples: np.ndarray) -> str:
    """Write a float32 16 kHz mono array to a temp PCM WAV file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="asr_stream_")
    os.close(fd)
    sf.write(path, samples, TARGET_SR, subtype="PCM_16")
    return path


def wav_duration_seconds(path: str) -> float:
    """Return the duration in seconds of a WAV file written by this module."""
    try:
        info = sf.info(path)
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:  # pragma: no cover - defensive
        pass
    return 0.0


def _decode_ffmpeg(raw: bytes) -> np.ndarray:
    """Decode any ffmpeg-supported container to float32 16 kHz mono.

    The input goes through a temp file rather than stdin: MP4/M4A files with
    the ``moov`` atom at the end need a seekable input.
    """
    fd, src = tempfile.mkstemp(prefix="asr_src_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        proc = subprocess.run(
            [
                _FFMPEG or "ffmpeg",
                "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", src,
                "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR),
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip().splitlines()
        raise InvalidAudioError(
            f"could not decode audio: {err[-1] if err else 'ffmpeg failed'}"
        )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def decode_16k_mono(raw: bytes) -> np.ndarray:
    """Decode `raw` audio bytes to a float32 16 kHz mono array."""
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        # Downmix to mono: (frames, channels) -> (frames,)
        mono = data.mean(axis=1)
        if sr != TARGET_SR:
            g = math.gcd(int(sr), TARGET_SR)
            mono = resample_poly(mono, TARGET_SR // g, int(sr) // g).astype(np.float32)
    except Exception as exc:  # libsndfile raises a variety of error types
        if _FFMPEG is None:
            raise InvalidAudioError(f"could not decode audio: {exc}") from exc
        mono = _decode_ffmpeg(raw)

    if mono.size == 0:
        raise InvalidAudioError("audio contains no samples")

    # Guard against clipping introduced by downmixing / resampling.
    peak = float(np.max(np.abs(mono)))
    if peak > 1.0:
        mono = mono / peak
    return mono


def to_wav16k_mono(raw: bytes) -> str:
    """Decode `raw` audio bytes and write a 16 kHz mono PCM WAV to a temp file.

    Returns the path to the temporary file; the caller is responsible for
    deleting it.
    """
    mono = decode_16k_mono(raw)
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="asr_")
    os.close(fd)
    sf.write(path, mono, TARGET_SR, subtype="PCM_16")
    return path


def split_on_silence(
    samples: np.ndarray, max_duration: float, search_duration: float
) -> list[np.ndarray]:
    """Split audio into chunks of at most ``max_duration`` seconds.

    Each cut is placed at the quietest 20 ms frame within the last
    ``search_duration`` seconds of the window, so words are rarely split.
    """
    max_len = int(max_duration * TARGET_SR)
    search_len = min(int(search_duration * TARGET_SR), max_len // 2)
    frame = TARGET_SR // 50

    chunks: list[np.ndarray] = []
    start = 0
    while len(samples) - start > max_len:
        win_start = start + max_len - search_len
        window = samples[win_start : start + max_len]
        n_frames = len(window) // frame
        if n_frames == 0:
            cut = start + max_len
        else:
            energy = np.square(window[: n_frames * frame]).reshape(n_frames, frame).mean(axis=1)
            cut = win_start + int(np.argmin(energy)) * frame + frame // 2
        chunks.append(samples[start:cut])
        start = cut
    chunks.append(samples[start:])
    return chunks
