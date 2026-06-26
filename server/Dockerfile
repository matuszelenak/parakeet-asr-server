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

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

COPY app ./app

EXPOSE 9000

# The app only starts serving once the model is loaded (lifespan startup), so a
# successful /health probe means the server is actually ready. Allow a long
# start period for the model download/load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD wget --no-verbose --tries=1 http://localhost:9000/health || exit 1

CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
