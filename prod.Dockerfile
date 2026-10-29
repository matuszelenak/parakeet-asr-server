# syntax=docker/dockerfile:1
#
# Production image: Svelte frontend baked into the server image, which serves
# the UI and the API on port 9000. See docs/deployment.md.
#
#   docker build -f prod.Dockerfile -t parakeet-asr:prod .
#   docker run --gpus all -p 9000:9000 parakeet-asr:prod

# ---- Stage 1: build the frontend with Deno (build-time only) ----
FROM denoland/deno:2.9.7 AS frontend
WORKDIR /frontend
COPY frontend/deno.json frontend/package.json ./
RUN deno install
COPY frontend/ ./
RUN deno task build
# Precompress the VAD assets (16 MB -> 5 MB) so the server never gzips a
# multi-megabyte body on its event loop; app.main serves the .gz siblings.
RUN find /frontend/dist -type f \( -name '*.wasm' -o -name '*.onnx' -o -name '*.js' -o -name '*.css' \) \
        -size +256k -exec gzip -9 -k -f {} + \
    && ls -l /frontend/dist/*.gz

# ---- Stage 2: build the venv (build-time only) ----
FROM ghcr.io/astral-sh/uv@sha256:929560df6a6231d0509739319bc214aaa1e78838ce1f779b1faeb877c23a50d8 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=managed \
    UV_PYTHON=3.12 \
    UV_COMPILE_BYTECODE=1 \
    UV_CONCURRENT_INSTALLS=8

# Compiler/git only: needed for any source builds during `uv sync`. This stage
# is not shipped, so the toolchain never reaches the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/pyproject.toml server/README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# nemo-toolkit[asr] pulls telemetry/profiling tooling the inference path never
# imports: wandb, tensorboard's gRPC server, and triton (only used by
# torch.compile, which this server does not use). Dropping them is verified by
# loading and transcribing both Parakeet and Canary. Keep numba/llvmlite:
# AudioToMelSpectrogramPreprocessor imports numba at model load.
RUN uv pip uninstall -p /app/.venv \
      wandb triton grpcio tensorboard_data_server

# Smoke-test the import chain (no GPU or weights needed); this catches a bad
# trim before it ships.
RUN /app/.venv/bin/python -c "import nemo.collections.asr.models"

COPY server/app ./app

# ---- Stage 3: runtime (slim) ----
FROM debian:bookworm-slim@sha256:99c0b25c46d0214d5a2091092e7908907dcaecd6a962cde0ac700acee222b6f3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/hf \
    HF_HUB_CACHE=/cache/hf/hub \
    TORCH_HOME=/cache/torch \
    STATIC_DIR=/app/static

# Runtime libs only: libsndfile (soundfile), ffmpeg (audio decode fallback),
# libgomp (torch OpenMP), ca-certificates+wget (model download / healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        wget \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The uv-managed CPython the venv points at, plus the symlink the venv's
# pyvenv.cfg / scripts resolve through.
COPY --from=builder /root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu \
                    /root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu
RUN ln -s /root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu \
          /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app /app/app
COPY --from=frontend /frontend/dist /app/static

WORKDIR /app

EXPOSE 9000

# Serving starts only after the model loads (lifespan startup); the start period
# covers the download.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD wget --no-verbose --tries=1 http://localhost:9000/health || exit 1

# granian: uvicorn breaks WebSocket upgrades after transducer inference (see
# docs/pitfalls.md). One worker - pool, session counters and metrics are per
# process.
CMD ["/app/.venv/bin/granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "9000", "--workers", "1", "--log-level", "info", "app.main:app"]
