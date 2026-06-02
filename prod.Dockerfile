# syntax=docker/dockerfile:1
#
# Production image: builds the Svelte frontend and bakes it into the server
# image, which serves it as static files alongside the API on a single port.
#
# Build from the repository root:
#   docker build -f prod.Dockerfile -t parakeet-asr:prod .
# Run (needs the NVIDIA Container Toolkit):
#   docker run --gpus all -p 9000:9000 parakeet-asr:prod
# Then open http://<host>:9000 — UI and API are served from the same origin.

# ---- Stage 1: build the frontend with Deno ----
FROM denoland/deno:2.8.1 AS frontend
WORKDIR /frontend
# Install deps first for layer caching.
COPY frontend/deno.json frontend/package.json ./
RUN deno install
COPY frontend/ ./
# The bundle calls the API at its own origin (this same server), so there is no
# API base to bake in. Outputs to /frontend/dist.
RUN deno task build

# ---- Stage 2: assemble the server image ----
FROM ghcr.io/astral-sh/uv@sha256:929560df6a6231d0509739319bc214aaa1e78838ce1f779b1faeb877c23a50d8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System libs: libsndfile for soundfile, ffmpeg for broad audio decoding,
# build-essential/git for any source builds pulled in by nemo_toolkit[asr],
# wget for the healthcheck.
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
    TORCH_HOME=/cache/torch \
    STATIC_DIR=/app/static

WORKDIR /app

# Install dependencies first for layer caching (the slow, heavy layer). The uv
# cache lives on a BuildKit cache mount so it stays out of the image layer.
COPY server/pyproject.toml server/README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# Application code and the built frontend.
COPY server/app ./app
COPY --from=frontend /frontend/dist ./static

EXPOSE 9000

# The app only starts serving once the model is loaded (lifespan startup), so a
# successful /health probe means the server is actually ready. Allow a long
# start period for the model download/load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD wget --no-verbose --tries=1 http://localhost:9000/health || exit 1

CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
