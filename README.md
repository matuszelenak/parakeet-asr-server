# Parakeet ASR Server

A FastAPI service that serves NVIDIA's **Parakeet/Canary** speech-to-text models,
plus a Svelte frontend for recording, uploading and live-transcribing audio in
the browser. Designed to run on a node with NVIDIA GPUs.

![Parakeet](screenshot.png)

```
.
├── server/               # FastAPI + NeMo API (managed with uv)
├── frontend/             # Svelte + TypeScript app (managed with Deno)
├── docs/                 # API, configuration, deployment, benchmarks, pitfalls
├── Makefile              # run/stop/status for one or two model instances
├── docker-compose.yml    # development stack (hot reload)
├── docker-compose.prod.yml
└── prod.Dockerfile       # production image: frontend baked into the API image
```

## Features

- **Three batch transcription modes** — plain text, word/segment/char
  timestamps, and long-form (local-attention windowing for very long recordings).
- **OpenAI-compatible API** — `/v1/audio/transcriptions`, `/v1/audio/translations`
  and `/v1/models`, including SSE streaming (`stream=true`), so the server can
  sit behind a [LiteLLM proxy](docs/api.md#litellm-proxy) or be used with any
  OpenAI SDK.
- **Live transcription** — real-time streaming over WebSocket, driven by
  client-side Silero VAD so only detected speech reaches the server.
- **Any audio in** — uploads are decoded (libsndfile, with an ffmpeg fallback for
  M4A/MP4/WebM/…), downmixed to mono and resampled to 16 kHz.
- **Parallel requests** — one model replica per GPU, and requests that arrive
  together are batched into a single model call, so throughput scales with
  concurrency instead of serialising.
- **Admission control** — bounded queue, per-request deadlines and per-session
  limits, surfaced as `503`/`Retry-After` (HTTP) and WebSocket close code 1013,
  so overload degrades into fast rejections instead of unbounded latency.
- **Prometheus metrics** — queue depth, batch sizes, inference time, commit lag
  and rejection reasons on `GET /metrics`.
- **Language selection & translation** — models that support it (e.g.
  `nvidia/canary-1b-v2`) accept `source_lang` / `target_lang`; different values
  perform speech translation.
- **Any NeMo checkpoint** — point `MODEL_PATH` at a local `.nemo` file to serve a
  community fine-tune.

## Quick start

Requires Docker with the NVIDIA Container Toolkit.

### Development stack (hot reload)

```bash
cp .env.example .env        # optional — tweak model, worker count, etc.
docker compose up --build
```

- API & docs: http://localhost:9000/docs
- Frontend:   http://localhost:5173

Changes to `frontend/src` hot-reload via Vite; changes to `server/app` sync and
restart the API (`docker compose watch` does the same with a watch loop). The
first boot downloads the model weights into the `model-cache` volume.

### Production image (single container)

`prod.Dockerfile` compiles the Svelte frontend and bakes it into the server
image, which then serves the UI **and** the API from one port.

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

Open http://localhost:9000 — UI and API on the same origin. Equivalent manual
build/run:

```bash
docker build -f prod.Dockerfile -t parakeet-asr:prod .
docker run --gpus all -p 9000:9000 \
  -v /models/huggingface:/cache/hf -v /cache/torch:/cache/torch \
  parakeet-asr:prod
```

### Two models side by side (Makefile)

One image, two independent instances, each with its own model, env profile and
port:

```bash
make build                     # production image, shared by both instances
make run-parakeet              # Parakeet Ultra, GPU 0 -> http://localhost:9000
make run-canary                # Canary 1B v2,   GPU 1 -> http://localhost:9001
make run-both                  # both at once
GPU=2 make run-canary          # explicit GPU always wins
make status                    # model, precision, language mode, stream slots, queue
make logs-parakeet | make stop-parakeet | make stop-all
make dev-up | make dev-down    # the hot-reload dev stack
```

`make` on its own lists every recipe. Details, VRAM budget and the profile
contents: **[docs/deployment.md](docs/deployment.md#running-two-model-instances)**.

## Running without Docker

### Server (uv)

```bash
cd server
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 9000
```

(`granian` is what the images use — see
[docs/deployment.md](docs/deployment.md#asgi-server-granian). For a host run,
`uv run granian --interface asgi --host 0.0.0.0 --port 9000 app.main:app` from
`server/` behaves the same.)

### Frontend (Deno)

```bash
cd frontend
deno install          # install npm deps via Deno's node_modules compat
deno task dev         # copies VAD assets then starts Vite — http://localhost:5173
# deno task build     # copies VAD assets then builds production bundle → dist/
# deno task check     # Svelte + TypeScript type-check
```

## Configuration

Everything is set through environment variables (see `.env.example` for a
commented list). The ones you normally touch:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `MODEL_NAME` | `nvidia/canary-1b-v2` | HuggingFace model ID to load |
| `MODEL_PATH` | _(unset)_ | Load a local `.nemo` checkpoint instead (takes precedence) |
| `SERVED_MODEL_NAME` | _(= `MODEL_NAME`)_ | Model id reported by `GET /v1/models` |
| `TORCH_DTYPE` | `auto` | `auto` (bf16 on Ampere+) / `float32` / `bfloat16` / `float16` |
| `MAX_CONCURRENT_STREAMS` | `0` (auto) | Concurrent live WebSocket sessions |
| `MAX_QUEUE_SIZE` | `32` | Queued batch requests before new ones get `503` |
| `HF_HOME` | `./server/models` | Host path mounted as the model cache |

Model- and deployment-specific tuning — batching, precision, streaming
parameters, admission limits, metrics, frontend proxy:

- **[docs/configuration.md](docs/configuration.md)** — full reference, every variable
- **[.env.parakeet](.env.parakeet) / [.env.canary](.env.canary)** — working
  profiles for the two instance recipes

## API at a glance

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/health` | Readiness, model, precision, queue/batch/stream state |
| `GET`  | `/metrics` | Prometheus metrics |
| `POST` | `/v1/transcribe` | Plain-text transcription |
| `POST` | `/v1/transcribe/timestamps` | Word / segment / char timestamps |
| `POST` | `/v1/transcribe/longform` | Long-form transcription |
| `WS`   | `/v1/transcribe/stream` | Real-time streaming transcription |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible transcription (+ SSE) |
| `POST` | `/v1/audio/translations` | OpenAI-compatible translation to English |
| `GET`  | `/v1/models` | OpenAI-compatible model list |

```bash
curl -F file=@sample.wav http://localhost:9000/v1/transcribe
curl -F file=@sample.wav http://localhost:9000/v1/transcribe/longform
```

Field-by-field reference, SSE format, WebSocket protocol, language handling and
the LiteLLM setup: **[docs/api.md](docs/api.md)**.

## UI

Four mode tabs plus a transcript panel:

| Tab | Description |
| --- | ----------- |
| **Live** | Real-time mic transcription with a Silero VAD activity meter |
| **Plain text** | Record or upload → plain transcript |
| **Timestamped** | Record or upload → transcript + word/segment timings |
| **Long form** | Record or upload → long-form transcript |

Audio can be recorded in the browser or dropped onto the drop zone (WAV, MP3,
M4A, WEBM, …) — everything is decoded, downmixed and resampled to 16 kHz
client-side before upload. Language selectors appear when the loaded model
supports them. The transcript panel offers copy, download (`.txt`) and clear.

How the live path works internally (VAD gating, commit flow, asset loading):
**[docs/frontend.md](docs/frontend.md)**.

## Documentation

| Document | Contents |
| -------- | -------- |
| [docs/api.md](docs/api.md) | Endpoints, request fields, SSE streaming, WebSocket protocol, LiteLLM proxy, languages |
| [docs/configuration.md](docs/configuration.md) | Every server and frontend environment variable |
| [docs/deployment.md](docs/deployment.md) | Production image, two-instance Makefile workflow, GPUs/VRAM, replicas, metrics, granian |
| [docs/benchmarks.md](docs/benchmarks.md) | Measured throughput, precision, model comparison (Parakeet / Canary / audio.cpp), methodology |
| [docs/pitfalls.md](docs/pitfalls.md) | Every non-obvious failure we hit and why the code looks the way it does |
| [docs/frontend.md](docs/frontend.md) | UI structure, live-transcription flow, VAD assets, dev workflow |

## Hardware notes

- One 24 GB GPU is enough for one replica; VRAM depends on the model and on
  whether long-form/full-context attention is in use (Canary bf16 ≈ 10 GB,
  Parakeet fp32 ≈ 5 GB, ≈ 13 GB whole-file full context). Numbers and limits in
  [docs/deployment.md](docs/deployment.md).
- The Silero VAD runs entirely in the browser (AudioWorklet, ONNX Runtime Web,
  single-threaded) — no Cross-Origin-Isolation headers required.
