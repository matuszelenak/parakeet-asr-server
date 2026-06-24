"""Audio ingestion: accept arbitrary WAV uploads, normalise to 16 kHz mono.

The ASR model expects 16 kHz mono audio. Uploaded WAV files may use any sample
rate, bit depth, or channel count, so we decode with libsndfile, downmix to
mono, resample with torchaudio, and write a temporary 16-bit PCM WAV that the
model can read. (Live streaming feeds float32 samples to the model directly and
does not go through this module.)
"""
from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

TARGET_SR = 16_000


class InvalidAudioError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as audio."""


def to_wav16k_mono(raw: bytes) -> str:
    """Decode `raw` audio bytes and write a 16 kHz mono PCM WAV to a temp file.

    Returns the path to the temporary file; the caller is responsible for
    deleting it.
    """
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:  # libsndfile raises a variety of error types
        raise InvalidAudioError(f"could not decode audio: {exc}") from exc

    if data.size == 0:
        raise InvalidAudioError("audio contains no samples")

    # Downmix to mono: (frames, channels) -> (frames,)
    mono = data.mean(axis=1)

    if sr != TARGET_SR:
        tensor = torch.from_numpy(mono).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, sr, TARGET_SR)
        mono = tensor.squeeze(0).contiguous().numpy()

    # Guard against clipping introduced by downmixing.
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="asr_")
    os.close(fd)
    sf.write(path, mono, TARGET_SR, subtype="PCM_16")
    return path
