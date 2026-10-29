# Deployment

## Production image

`prod.Dockerfile` is a two-stage build:

1. **Frontend** — a Deno image runs `deno install` and `deno task build`, which
   copies the VAD runtime (`silero_vad_v5.onnx`, `ort-wasm-simd-threaded.wasm`,
   the AudioWorklet bundle) into `public/` and produces `dist/`. Large assets are
   then gzipped (`gzip -9`) so the server can serve `.gz` siblings instead of
   compressing on the request path.
2. **Server** — the `uv` image installs system libs (ffmpeg, libsndfile) and the
   Python deps (`uv sync --no-dev`), copies `dist/` to `/app/static`, and sets
   `STATIC_DIR=/app/static`. One process serves UI and API on port 9000 with
   granian.

```bash
docker build -f prod.Dockerfile -t parakeet-asr:prod .
docker run --gpus all -p 9000:9000 \
  -v /models/huggingface:/cache/hf -v /cache/torch:/cache/torch \
  parakeet-asr:prod
```

or `docker compose -f docker-compose.prod.yml up --build -d`.

Cache mounts: `/cache/hf` (`HF_HOME`) holds the HuggingFace cache and any local
`.nemo` files, `/cache/torch` holds `TORCH_HOME`. Mount both, otherwise every
recreate re-downloads the weights.

The healthcheck is `GET /health` (30 s interval, 600 s start period for the
model load).

## Running two model instances

`Makefile` drives one image twice, with separate env profiles
([`.env.parakeet`](../.env.parakeet), [`.env.canary`](../.env.canary)) and
separate compose projects (`asr-parakeet`, `asr-canary`), so starting or
stopping one never touches the other:

```bash
make build                     # production image, shared by both
make run-parakeet              # Parakeet Ultra, GPU 0 -> http://localhost:9000
make run-canary                # Canary 1B v2,   GPU 1 -> http://localhost:9001
make run-both
GPU=2 make run-canary          # explicit GPU always wins
make status                    # model, precision, language mode, stream slots, queue
make ps | logs-parakeet | stop-parakeet | stop-all | restart-canary
make dev-up | make dev-down    # hot-reload dev stack (takes port 9000)
```

GPU selection: `GPU=<n>` if given, else `GPU_PARAKEET` / `GPU_CANARY`
(default 0 / 1). The per-instance defaults exist on purpose — a shared default
once made `make run-canary` recreate the canary container on the parakeet GPU
(compose recreates on that config change rather than failing).

Each instance serves its own bundled UI on its own port; the UI always talks to
the API on its own origin, so `:9000` drives Parakeet and `:9001` drives Canary
with no frontend switch to configure.

### VRAM budget

Measured on one RTX 3090 (24 GB):

| instance | precision | resident | notes |
| -------- | --------- | -------- | ----- |
| `canary-1b-v2` | bf16 | ~10 GB | one replica, any audio length |
| `parakeet-ultra` | fp32 | ~5 GB | long-form / chunked |
| `parakeet-ultra` | fp32 | ~13 GB | whole-file full-context attention on a 9-minute file |

Both instances fit on two 3090s with headroom; running them on the *same* GPU
also works if you keep to short audio, but there is no admission control across
processes.

## Scaling past one replica

Within one process, capacity is bounded by the GPU:
~12 req/s of 20 s clips (~235 audio-s/s), ~12 concurrent real-time streams in
bf16, one long upload at a time (files longer than `MAX_BATCH_ITEM_SECONDS`
never share a batch). See [benchmarks.md](benchmarks.md#concurrency--capacity).

For more, run more containers behind an L7 proxy that supports WebSocket upgrade
(LiteLLM for the OpenAI routes, plus a plain reverse proxy if you need the live
endpoint), and scale on queue pressure:

```bash
kubectl get --raw /api/v1/namespaces/asr/services/asr:9000/proxy/metrics \
  | grep asr_queue_depth
```

`/health` reports the same state without Prometheus:

```json
{
  "queue": {"pending": 0, "inflight": 1, "capacity": 32},
  "batching": {"window_ms": 20, "max_batch_size": 8, "max_batch_audio_seconds": 240},
  "stream_capacity": {"active": 2, "limit": 10},
  "dtype": "float32",
  "matmul_precision": "high"
}
```

Keep one process per container: the model pool, the live-session counters and
the metrics registry are all per process, so `--workers > 1` would multiply
loaded models and split the counters.

GPU assignment in compose uses `deploy.resources.reservations.devices`; with
multiple GPUs the default loads one replica per GPU, so `NUM_WORKERS` is a cap,
not a target. Each replica is pinned to its GPU and switches to local attention
only while serving a long-form request, then restores the default attention
config.

## Observability

Logging and tracing use [Logfire](https://logfire.pydantic.dev/). FastAPI is
instrumented (one span per request), each transcription and streaming session
runs inside its own span, and stdlib logging (granian, NeMo) is routed through
Logfire as well. Set `LOGFIRE_TOKEN` to export to the Logfire backend.

### Prometheus metrics (`GET /metrics`)

Scrapeable text format, one process (no multiprocess collector needed). Route
labels use templates (`/v1/models/{model_id}`), so cardinality stays bounded.

| Metric | Type | Use |
| ------ | ---- | --- |
| `asr_http_requests_total{route,method,status}` | counter | error rates per endpoint |
| `asr_http_request_duration_seconds{route}` | histogram | latency SLOs (time to response *start* for SSE) |
| `asr_rejected_total{reason}` | counter | `queue_full`, `queue_timeout`, `stream_limit`, `stream_backpressure`, `upload_too_large` |
| `asr_queue_depth`, `asr_queue_capacity`, `asr_inflight_requests` | gauge | **scale on these** (queue depth / capacity) |
| `asr_queue_wait_seconds` | histogram | time spent waiting for the model |
| `asr_inference_batches_total{mode}`, `asr_inference_batch_size{mode}` | counter/histogram | is batching actually merging requests? |
| `asr_inference_seconds{mode}`, `asr_inference_audio_seconds_total{mode}` | histogram/counter | GPU occupancy and real-time factor |
| `asr_audio_seconds_total`, `asr_upload_bytes` | counter/histogram | input volume |
| `asr_stream_sessions_active`, `asr_stream_sessions_total{outcome}` | gauge/counter | live capacity and how sessions end |
| `asr_stream_commits_total`, `asr_stream_partials_total` | counter | transcript churn |
| `asr_stream_commit_lag_seconds` | histogram | how much audio is still uncommitted at each commit |
| `asr_model_ready`, `asr_workers` | gauge | readiness / capacity |

## ASGI server (granian)

The images run `granian` (Rust/Hyper) rather than uvicorn, because with uvicorn
0.54 the WebSocket endpoint stops accepting upgrades once a transducer model has
run an inference in the process. The full story, the error string and what was
ruled out: [pitfalls.md](pitfalls.md#uvicorn-drops-websocket-upgrades-after-transducer-inference).

Operational consequences:

- `--workers 1` — pool, session counters and metrics are per process.
- No response compression in the server itself; the static assets are
  precompressed at build time (see above).
- Request logging and access formats come from granian; keep `--log-level info`
  unless you are debugging.

## Static frontend serving

`STATIC_DIR` mounts the built UI at `/` behind the API routes. Assets are served
with cache headers by class:

| Asset | `Cache-Control` |
| ----- | --------------- |
| `/assets/*` (content-hashed by Vite) | `public, max-age=31536000, immutable` |
| VAD runtime (`*.onnx`, `*.wasm`, worklet) | `public, max-age=604800` |
| `index.html` | `no-cache` |
| everything else | `public, max-age=3600` |

Large assets are shipped as `<name>.gz` and served to clients that accept gzip
(wasm 13.6 MB → 3.5 MB, onnx 2.2 MB → 1.85 MB). Compression is deliberately not
done per request: gzipping that wasm costs about a second of the single event
loop, which would stall running WebSocket streams. After upgrading the image,
clients pick the new bundle because `index.html` is revalidated.

## Development stack

```bash
docker compose up --build       # or: docker compose watch
```

- Changes to `frontend/src` hot-reload via Vite HMR.
- Changes to `server/app` sync and restart the API.
- Changes to `pyproject.toml` / `package.json` / `deno.json` trigger a rebuild.

> **VAD assets**: the first `deno task dev` (or `docker compose watch`) copies
> three files from `node_modules` into `frontend/public/`: the Silero VAD v5 ONNX
> model, the ONNX Runtime WASM binary, and the AudioWorklet bundle. They are
> gitignored and regenerated automatically; do not commit them.
