# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv@sha256:929560df6a6231d0509739319bc214aaa1e78838ce1f779b1faeb877c23a50d8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System libs: libsndfile for soundfile, ffmpeg for broad audio decoding,
# build-essential/git for any source builds pulled in by nemo_toolkit[asr].
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        ffmpeg \
        libsndfile1 \
        build-essential \
        wget \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=managed \
    UV_CONCURRENT_INSTALLS=8 \
    HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf/hub \
    TORCH_HOME=/cache/torch

# Pin the interpreter to 3.12. On the base image's Python 3.13, serving a
# transducer model (Parakeet-TDT family) leaves uvicorn unable to complete any
# later WebSocket upgrade — every live session works once, then all further
# upgrades hang (HTTP routes keep working). The same code, model and deps on
# 3.12 (host venv) do not show it, and Canary on 3.13 does not either, so this
# keeps the live path working for every supported model.
ENV UV_PYTHON=3.12

WORKDIR /app

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

COPY app ./app

EXPOSE 9000

# The app only starts serving once the model is loaded (lifespan startup), so a
# successful /health probe means the server is actually ready. Allow a long
# start period for the model download/load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD wget --no-verbose --tries=1 http://localhost:9000/health || exit 1

# ASGI server: granian (Rust/Hyper) instead of uvicorn. With uvicorn 0.54 every
# WebSocket implementation (websockets-sansio, legacy websockets, wsproto) stops
# accepting upgrades once a transducer model (Parakeet-TDT family) has run an
# inference in the process: the handshake request arrives truncated
# ("invalid HTTP request line: GET/v1/transcribe/stream HTTP/1.1") and the client
# hangs or gets a 400. Reproduced on this image, not with Canary and not on the
# host venv; the uvicorn issue tracker has related sansio handshake bugs
# (#3107, #3175, #2825). granian serves the same ASGI app with a healthy
# WebSocket path and matching throughput.
#
# One worker only: the model pool, the live-session counters and the metrics
# registry are per process.
CMD ["/app/.venv/bin/granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "9000", "--workers", "1", "--log-level", "info", "app.main:app"]
