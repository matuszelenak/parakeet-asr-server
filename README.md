# Parakeet ASR Server

A FastAPI service that serves NVIDIA's **Parakeet/Canary** speech-to-text model, plus a Svelte frontend for
recording, uploading, and live-transcribing audio in the browser. Designed to
run on a node with NVIDIA GPUs.

![Parakeet](screenshot.png)

```
.
├── server/      # FastAPI + NeMo API (managed with uv)
├── frontend/    # Svelte + TypeScript app (managed with Deno)
└── docker-compose.yml
```

## Features

- **Three batch transcription modes** — plain text, word/segment/char
  timestamps, and long-form (local-attention windowing for very long recordings).
- **OpenAI-compatible API** — `/v1/audio/transcriptions`, `/v1/audio/translations`
  and `/v1/models`, including **SSE streaming** (`stream=true`), so the server
  can sit behind a [LiteLLM proxy](#litellm-proxy) or be used with any OpenAI SDK.
- **Live transcription** — real-time streaming transcription over WebSocket,
  driven by client-side Silero VAD so only detected speech reaches the server.
- **Any audio in** — uploads are decoded (libsndfile, with an ffmpeg fallback
  for M4A/MP4/WebM/...), downmixed to mono, and resampled to 16 kHz.
- **Parallel requests** — one model replica is loaded per GPU; requests that
  arrive together are **batched into a single model call** by a scheduler, so
  throughput scales with concurrency instead of serialising.
- **Admission control** — a bounded queue, per-request deadlines and per-session
  limits, surfaced as `503`/`Retry-After` (HTTP) and WebSocket close code 1013,
  so overload degrades into fast rejections instead of unbounded latency.
- **Prometheus metrics** — `GET /metrics` exposes queue depth, batch sizes,
  inference time, commit lag and rejection reasons for HPA/KEDA and dashboards.
- **Language selection & translation** — models that support it (e.g.
  `nvidia/canary-1b-v2`) accept `source_lang` / `target_lang`; setting them to
  different values performs speech translation.

## UI overview

The frontend is a control panel with four mode tabs, plus a transcript panel
underneath:

| Tab | Description |
| --- | ----------- |
| **Live** | Real-time mic transcription with Silero VAD activity meter |
| **Plain text** | Record or upload → plain transcript |
| **Timestamped** | Record or upload → transcript + word/segment timings |
| **Long form** | Record or upload → long-form transcript |

Language selectors (source → target, with a swap button) appear at the top when
the loaded model supports them; they are disabled while live transcription is
active. The transcript panel shows a word count and offers copy, download
(`.txt`) and clear actions; word/segment timings that a mode returns are
rendered below the transcript.

Audio can be recorded in the browser or dropped onto the drop zone (WAV, MP3,
M4A, WEBM, …) — everything is decoded, downmixed to mono and resampled to
16 kHz client-side before upload. If the backend cannot be reached, a notice
appears above the controls.

### Live transcription

The Live tab uses the [Silero VAD](https://github.com/ricky0123/vad) browser
library (ONNX Runtime Web, v5 model) to detect voice activity locally:

1. Click **Start** — the ONNX model is loaded (one-time, ~2 s) and a WebSocket
   connection is opened to `/v1/transcribe/stream`.
2. The VAD runs on every audio frame (512 samples / 32 ms) and shows a
   **probability meter** that fills green→red as speech likelihood rises. A
   pulsing dot indicates active speech.
3. Only frames where speech is detected are forwarded to the server as
   PCM-16 / 16 kHz binary WebSocket frames; silent intervals are dropped.
4. The server streams back `partial` events (shown in italic grey) and
   `committed` events (appended to the permanent transcript) in real time.
5. When the VAD detects that the user has stopped speaking it waits for the
   **commit delay** (configurable slider, 200 – 1 000 ms, default 500 ms) and
   then sends `{type: "end"}` to the server. The server finalises the current
   segment, emits a `final` event, and closes the connection. The client
   silently reopens the WebSocket for the next utterance.
6. The status label cycles through: **Listening…** → **Speaking…** →
   **Committing… (N ms)** → **Listening…**
7. Click **Stop** to end the session.

## API

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/health` | Readiness, worker count, language capabilities |
| `POST` | `/v1/transcribe` | Plain-text transcription |
| `POST` | `/v1/transcribe/timestamps` | Word / segment / char timestamps |
| `POST` | `/v1/transcribe/longform` | Long-form transcription |
| `WS`   | `/v1/transcribe/stream` | Real-time streaming transcription |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible transcription (+ SSE streaming) |
| `POST` | `/v1/audio/translations` | OpenAI-compatible translation to English |
| `GET`  | `/v1/models` | OpenAI-compatible model list |

`POST` endpoints take `multipart/form-data` with a `file` field (any common
audio format).

```bash
curl -F file=@sample.wav http://localhost:9000/v1/transcribe
curl -F file=@sample.wav http://localhost:9000/v1/transcribe/timestamps
curl -F file=@sample.wav http://localhost:9000/v1/transcribe/longform
```

### OpenAI-compatible endpoints

`POST /v1/audio/transcriptions` accepts the standard OpenAI fields:

| Field | Notes |
| ----- | ----- |
| `file` | Required. Any format libsndfile or ffmpeg can decode |
| `model` | Accepted and ignored (the server runs one model) |
| `language` | ISO-639-1 code (`en`, `sk`, `en-US` is fine). Selects the source language on Canary; ignored on models without language selection |
| `response_format` | `json` (default), `text`, `verbose_json`, `srt`, `vtt` |
| `timestamp_granularities[]` | `word` and/or `segment` (for `verbose_json`) |
| `stream` | `true` → server-sent events (see below) |
| `prompt`, `temperature` | Accepted and ignored |
| `target_language` | Extension: translate into this language (Canary) |
| `longform` | Extension: local-attention mode for very long recordings |

`POST /v1/audio/translations` is the same minus streaming/timestamps, always
targets English, and takes the source via `language` (default `en`).

Language hints are **ignored** for models that do not take them (transducer
models such as Parakeet auto-detect the language) so that the same client code
works across a model swap; pass `source_lang`/`target_lang` only to Canary-style
models.

Errors on these routes use the OpenAI envelope
(`{"error": {"message", "type", "param", "code"}}`) so OpenAI SDKs and LiteLLM
surface the real message.

```bash
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@sample.m4a -F model=parakeet -F language=sk
```

#### SSE streaming (`stream=true`)

With `stream=true` (and `response_format` `json`/`text`; it is ignored for the
timestamped formats, as on OpenAI) the response is `text/event-stream` in
OpenAI's `gpt-4o-transcribe` format:

```
data: {"type":"transcript.text.delta","delta":"Hello there."}

data: {"type":"transcript.text.delta","delta":" How are you?"}

data: {"type":"transcript.text.done","text":"Hello there. How are you?"}
```

Parakeet/Canary are not incremental models, so the audio is split into chunks
of at most `SSE_CHUNK_DURATION` seconds, cut at the quietest point near the
limit. Each chunk is transcribed in turn and emitted as one delta, so the first
text arrives after one chunk rather than after the whole file. An error after
the stream has started is sent as a `data: {"error": {...}}` event, which the
OpenAI SDKs raise as an exception.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="unused")
with open("sample.wav", "rb") as f:
    for event in client.audio.transcriptions.create(
        model="parakeet", file=f, stream=True
    ):
        if event.type == "transcript.text.delta":
            print(event.delta, end="", flush=True)
```

`scripts/test_sse.py` does the same from the command line.

### LiteLLM proxy

[`litellm/config.yaml`](litellm/config.yaml) is a ready-to-use example. It
exposes the server in two ways:

1. **As a regular model** (`model_name: parakeet`, provider `openai/` with
   `api_base` pointing at this server). Clients call LiteLLM's standard
   `POST /v1/audio/transcriptions` with `model=parakeet` and get keys,
   budgets, logging, load balancing and health checks (`mode:
   audio_transcription`). All response formats work. LiteLLM sends
   `response_format=verbose_json` by default so that it can read `duration`
   for cost tracking.
2. **As a pass-through route** (`/parakeet/...` → this server, with LiteLLM
   key auth). This is how SSE streaming and the native endpoints reach clients
   through LiteLLM, e.g. `base_url="http://litellm:4000/parakeet/v1"`.

```bash
litellm --config litellm/config.yaml --port 4000
curl http://localhost:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-change-me" -F model=parakeet -F file=@sample.wav
```

> **Streaming caveat (LiteLLM ≤ 1.105.1, tested):** LiteLLM's own
> `/v1/audio/transcriptions` route does not stream. It drops `stream` and
> returns the complete JSON response. Its pass-through route forwards the SSE
> events intact (the OpenAI SDK's `stream=True` works through it), but for
> `multipart/form-data` uploads it buffers the upstream response, so all the
> deltas arrive together when transcription finishes. For truly progressive
> streaming, call the server directly. The WebSocket live endpoint is not
> proxied by LiteLLM either; expose it directly or through a plain reverse
> proxy.

### WebSocket streaming protocol

Connect to `/v1/transcribe/stream` (optional `?source_lang=…&target_lang=…`).

**Client → server**

| Frame | Content |
| ----- | ------- |
| Binary | PCM-16 mono 16 kHz audio samples (`Int16Array`) |
| Text | `{"type": "end"}` — signals end of stream, triggers final commit |

**Server → client**

| Event type | When emitted |
| ---------- | ------------ |
| `partial` | In-progress hypothesis, may still change |
| `committed` | Words the last `STREAM_STABLE_ITERS` hypotheses agree on; never revised |
| `final` | Last segment after `{"type": "end"}` received |

All events carry `{type, text, start, id}`.

### Source / target language (model-dependent)

```bash
# transcribe German speech
curl -F file=@de.wav -F source_lang=de -F target_lang=de \
  http://localhost:9000/v1/transcribe
# translate German speech to English
curl -F file=@de.wav -F source_lang=de -F target_lang=en \
  http://localhost:9000/v1/transcribe
```

Whether language selection is available is reported by `GET /health`
(`supports_languages` + `languages` list); the UI shows the pickers only when
the model supports them. Supplying these fields to a model that does not support
them returns `400`.

## Quick start (Docker Compose)

Requires Docker with the NVIDIA Container Toolkit.

```bash
cp .env.example .env        # optional — tweak model, worker count, etc.
docker compose up --build
```

- API & docs: http://localhost:9000/docs
- Frontend:   http://localhost:5173

The first boot downloads the model weights (cached in the `model-cache` volume).

### Live-reload development

```bash
docker compose watch
```

- Changes to `frontend/src` hot-reload via Vite HMR.
- Changes to `server/app` sync and restart the API.
- Changes to `pyproject.toml` / `package.json` / `deno.json` trigger a full
  rebuild.

> **Note — VAD assets**: the first time you run `deno task dev` (or `docker
> compose watch`) the `copy-assets` step copies four files from `node_modules`
> into `frontend/public/`: the Silero VAD v5 ONNX model, the ONNX Runtime WASM
> binary, and the AudioWorklet bundle. These are listed in `.gitignore` and
> regenerated automatically; do not commit them.

## Production image (single container)

`prod.Dockerfile` is a multi-stage build that compiles the Svelte frontend
(including the VAD asset copy step) and bakes the result into the server image.
FastAPI then serves the static UI **and** the API from a single port.

```bash
# build from the repo root
docker build -f prod.Dockerfile -t parakeet-asr:prod .

# run (needs the NVIDIA Container Toolkit)
docker run --gpus all -p 9000:9000 \
  -v /models/huggingface:/cache/hf -v /cache/torch:/cache/torch \
  parakeet-asr:prod
```

Or use the production compose file:

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

Open http://localhost:9000 — UI and API on the same origin.

## Running without Docker

### Server (uv)

```bash
cd server
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 9000
```

### Frontend (Deno)

```bash
cd frontend
deno install          # install npm deps via Deno's node_modules compat
deno task dev         # copies VAD assets then starts Vite — http://localhost:5173
# deno task build     # copies VAD assets then builds production bundle → dist/
# deno task check     # Svelte + TypeScript type-check
```

## Configuration

### Server environment variables

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `MODEL_NAME` | `nvidia/canary-1b-v2` | HuggingFace model ID to load |
| `NUM_WORKERS` | `0` (one per GPU) | Number of model replicas |
| `DEVICES` | _(auto)_ | Explicit device list, e.g. `cuda:0,cuda:1` |
| `MAX_UPLOAD_MB` | `200` | Max batch upload size |
| `LONGFORM_CONTEXT` | `256` | Local-attention context for long-form mode |
| `MODEL_PATH` | _(unset)_ | Load a local `.nemo` checkpoint instead of `MODEL_NAME` (e.g. a community fine-tune) |
| `INLINE_AUDIO_MAX_SECONDS` | `30` | Audio up to this length goes to the model as an in-memory array instead of a temp WAV |
| `BATCH_DURATION_RATIO` | `2` | Max longest/shortest duration inside one batch; mismatched lengths are decoded separately |
| `STREAM_MIN_DURATION` | `1.0` | Seconds of audio before first streaming inference |
| `STREAM_RETRANSCRIBE_INTERVAL` | `0.5` | Min new audio (s) between inference runs |
| `STREAM_STABLE_WORDS` | `1` | Min agreed words required for a commit |
| `STREAM_STABLE_ITERS` | `2` | Consecutive hypotheses that must agree before committing |
| `STREAM_MAX_DURATION` | `30.0` | Hard buffer cap (s) that forces a trim/commit |
| `STREAM_CONTEXT_DURATION` | `1.5` | Committed audio (s) kept before a trim point as context |
| `STREAM_TRIM_DURATION` | `10.0` | Buffer length (s) after which it is trimmed at the last committed sentence end |
| `STREAM_HOLDBACK` | `0.0` | Words ending this close (s) to the buffer end are not committed |
| `SERVED_MODEL_NAME` | _(= `MODEL_NAME`)_ | Model id reported by `GET /v1/models` |
| `SSE_CHUNK_DURATION` | `20.0` | Max audio (s) per chunk/delta for `stream=true` |
| `SSE_SPLIT_SEARCH` | `4.0` | Window (s) before the chunk limit searched for a quiet cut point |
| `BATCH_WINDOW_MS` | `20` | How long the scheduler waits for more requests before decoding a batch (`0` disables batch wait) |
| `MAX_BATCH_SIZE` | `8` | Max requests decoded in one model call |
| `MAX_BATCH_AUDIO_SECONDS` | `240` | Max total audio (s) per batch |
| `MAX_BATCH_ITEM_SECONDS` | `30` | Requests longer than this run on their own |
| `MAX_QUEUE_SIZE` | `32` | Queued requests before new ones get `503` (`0` = unlimited) |
| `QUEUE_TIMEOUT_S` | `120` | Max time a request may wait in the queue |
| `RETRY_AFTER_S` | `2` | `Retry-After` value sent with `503` |
| `MAX_CONCURRENT_STREAMS` | `0` (auto) | Concurrent WebSocket sessions; `0` = 8 in bf16/fp16, 4 in fp32 |
| `STREAM_MAX_QUEUE_SECONDS` | `120` | Unprocessed audio a client may queue before the session is closed (1013) |
| `STREAM_MAX_SESSION_SECONDS` | `3600` | Hard limit on one WebSocket session |
| `TORCH_DTYPE` | `auto` | Model weight dtype: `auto` (bf16 on compute capability ≥ 8.0, else fp32), `float32`, `bfloat16`, `float16` |
| `MATMUL_PRECISION` | `high` | fp32 matmul precision: `high` = TF32 tensor cores, `medium` (bf16 matmuls), `highest` (strict fp32) |
| `METRICS_ENABLED` | `true` | Expose `GET /metrics` |
| `LOGFIRE_TOKEN` | _(unset)_ | Export traces to Logfire; console-only if unset |

### Frontend environment variables

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `API_PROXY_TARGET` | `http://localhost:9000` (host) / `http://server:9000` (compose) | Backend address the Vite dev server proxies `/v1` and `/health` to |
| `VITE_API_BASE` | _(unset)_ | Override to call a different API origin directly from the browser |

The browser always calls the API at the same origin the page was loaded from.
The Vite proxy makes this work seamlessly in development.

## Observability

Logging and tracing use [Logfire](https://logfire.pydantic.dev/). FastAPI is
instrumented (one span per request), each transcription and streaming session
runs inside its own span, and stdlib logging (uvicorn, NeMo) is routed through
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
| `asr_inference_seconds{mode}`, `asr_inference_audio_seconds_total{mode}` | histogram/counter | GPU occupancy and real-time factor (audio-s ÷ s of decode) |
| `asr_audio_seconds_total`, `asr_upload_bytes` | counter/histogram | input volume |
| `asr_stream_sessions_active`, `asr_stream_sessions_total{outcome}` | gauge/counter | live capacity and how sessions end |
| `asr_stream_commits_total`, `asr_stream_partials_total` | counter | transcript churn |
| `asr_stream_commit_lag_seconds` | histogram | how much audio is still uncommitted at each commit |
| `asr_model_ready`, `asr_workers` | gauge | readiness / capacity |

`GET /health` additionally reports `queue`, `batching` and `stream_capacity` so
a simple orchestrator can react to queue pressure without Prometheus.

## Concurrency & capacity

Requests to the batch endpoints are **batched** by a scheduler: everything that
arrives within `BATCH_WINDOW_MS` and shares decoding options (timestamps /
long-form / languages) is decoded in one `model.transcribe(paths, batch_size=N)`
call, shortest-first to minimise padding. Live-stream inferences are batched the
same way, which is what lets several sessions share one GPU in real time.

Measured on one RTX 3090 with `nvidia/canary-1b-v2` (559 s Slovak podcast):

| scenario | eager fp32, no batching | scheduler (fp32) | scheduler + bf16 |
| -------- | ----------------------- | ---------------- | ---------------- |
| 20 s clips, 4 concurrent | 2.5 req/s, 1.0 s latency | 5.1 req/s, 0.78 s | **7.6 req/s, 0.52 s** |
| 20 s clips, 8 concurrent | 2.5 req/s, 1.9 s latency | 6.4 req/s, 1.2 s | **11.8 req/s, 0.68 s** |
| 559 s file, `/v1/transcribe/longform` | 6.5 s (86× RT) | 6.5 s | **3.5 s (160× RT)** |
| 559 s file, GPU-only (no HTTP) | 110× RT | 133× RT | **336× RT** |
| live streams on one GPU | 2 keep up, 3rd falls behind | 4 keep up | **12 keep up** |
| overload (45 concurrent) | unbounded queueing | 12 fast `503`s + `Retry-After` | same |

Results are order-mapped 1:1 to the batch inputs and decoding is greedy, so a
request always gets its own transcript. One caveat is inherent to NeMo, not to the
scheduler: clips of *different* lengths that share a forward pass are padded to
the longest one, and that can change a word at the tail of the shorter clip (this
happens with the file/lhotse path too). `BATCH_DURATION_RATIO` bounds the
mismatch (default 2×) and identical-length inputs are unaffected.

Practical limits per worker (one 24 GB GPU, `canary-1b-v2`): ~12 req/s of 20 s
clips (~235 audio-s/s), ~12 concurrent real-time streams in bf16 (8 by default,
see `MAX_CONCURRENT_STREAMS`), one long upload at a time (files longer than
`MAX_BATCH_ITEM_SECONDS` never share a batch). For higher concurrency run more
containers/replicas behind a load balancer (LiteLLM, or any L7 proxy that
supports WebSocket upgrade).

### Precision / numerics

`TORCH_DTYPE=auto` casts the replica to **bfloat16** on Ampere and newer
(TF32 matmuls are on via `MATMUL_PRECISION=high`). Measured on the 559 s Slovak
podcast against a word-timed reference:

| setting | RTFx (GPU-only) | WER | CER | transcript |
| ------- | --------------- | --- | --- | ---------- |
| fp32 (strict) | 110× | 10.97 % | 4.26 % | baseline |
| fp32 + TF32 | 208× | 10.97 % | 4.26 % | **byte-identical** |
| bf16 | 336× | 10.79 % | 4.19 % | ~10 character spans differ |
| fp16 | 331× | 11.07 % | 4.29 % | ~2 character spans differ |

Quality differences are at reference-noise level (~1 word in 1 050), so bf16 is
the default; set `TORCH_DTYPE=float32` if you need bit-exact reproducibility.
Note that NeMo rebuilds the encoder's attention modules when long-form mode is
toggled, and they come back in fp32 — the worker re-casts after each switch,
otherwise bf16 + long-form raises a dtype mismatch.

> **Streaming note for bulk audio:** the WebSocket endpoint is for live audio. A
> client that pushes faster than the model can decode is cut off after
> `STREAM_MAX_QUEUE_SECONDS` of unprocessed audio (close code 1013). Use
> `POST /v1/audio/transcriptions` with `stream=true` (SSE) to feed a file as fast
> as the server can take it.

### Other models (`MODEL_PATH`)

Any NeMo ASR checkpoint can be served: point `MODEL_PATH` at a local `.nemo`
file (it takes precedence over `MODEL_NAME` for loading) and set `MODEL_NAME` to
a label. Language arguments are only passed to Canary-style models
(`model_supports_languages`); transducer models such as Parakeet auto-detect the
language, and the WebSocket endpoint ignores `source_lang`/`target_lang` for them
instead of failing the upgrade.

Measured with **Moondream Parakeet Ultra** (`moondream/parakeet-ultra`,
`ncannings/parakeet-ultra-nemo` — a post-trained `parakeet-tdt-0.6b-v3`, 25
European languages) on the same 559 s Slovak file:

| | canary-1b-v2 (bf16) | parakeet-ultra (fp32) | parakeet-ultra (bf16) |
| --- | --- | --- | --- |
| whole file, GPU-only | 336× RT | 226× RT | 364× RT |
| 8 × 20 s batch | 328× RT | **927× RT** | – |
| 20 s window + timestamps (streaming shape) | 276 ms | **42 ms** | – |
| WER vs reference | 11.26 % | 10.41 % | 14.47 % |
| CER vs reference | 4.10 % | **2.99 %** | 4.68 % |
| translation (AST) | yes | no | no |

With `longform=true` (bounded local attention) Parakeet gets both better and
cheaper on long audio: **WER 8.70 % / CER 3.0 % at 211× realtime** on the same
559 s file, and VRAM stays at ~4 GB instead of ~13 GB for whole-file
full-context attention. Use `longform=true` (or `/v1/transcribe/longform`) for
anything longer than a couple of minutes with this model.

Takeaways: the transducer is 2.5-6× faster per inference and its punctuation /
capitalisation is better, at comparable accuracy (CER lower, WER within reference
noise); it cannot translate. **bf16 hurts the transducer on long audio** (WER
10.4 % → 14.5 % whole-file), so use `TORCH_DTYPE=float32` for Parakeet or feed it
shorter chunks (30 s chunks in bf16 scored 9.7 %). To try it:

```bash
# once (~2.5 GB)
python -c "from huggingface_hub import hf_hub_download as d; d('ncannings/parakeet-ultra-nemo', 'ultra.nemo', local_dir='/models/parakeet-ultra')"

MODEL_PATH=/models/parakeet-ultra/ultra.nemo \
MODEL_NAME=parakeet-ultra \
TORCH_DTYPE=float32 \
docker compose up -d server
```

## Running two model instances (Makefile)

One production image, two independent instances — each with its own model, its
own env profile, its own port and its own bundled UI:

```bash
make build                     # production image (frontend + API), shared by both
make run-parakeet              # Parakeet Ultra, GPU 0 -> http://localhost:9000
make run-canary                # Canary 1B v2,   GPU 1 -> http://localhost:9001
make run-both                  # both at once
GPU=2 make run-canary          # explicit GPU always wins
make status                    # model, precision, language mode, stream slots, queue
make logs-parakeet | make stop-parakeet | make stop-all
```

The GPU is chosen per invocation (`GPU=<n>`, else `GPU_PARAKEET` / `GPU_CANARY`,
defaulting to 0 and 1, so a bare `make run-canary` never lands on the Parakeet
GPU); everything else comes from the profile file, which is passed as
`--env-file` to
`docker-compose.prod.yml` and therefore *replaces* `.env` for that instance:

| | `.env.parakeet` | `.env.canary` |
| --- | --- | --- |
| model | `MODEL_PATH=/cache/hf/parakeet-ultra/ultra.nemo` | `MODEL_NAME=nvidia/canary-1b-v2` (HF cache) |
| precision | `float32` (bf16 costs ~4 WER points on long audio) | `auto` → bf16 |
| languages | auto-detect, no hint needed | `DEFAULT_SOURCE_LANG=sk` so requests without language fields transcribe instead of translating |
| live tuning | `STREAM_CONTEXT_DURATION=20`, trim 30, max 60, 3 iters (~0.6 s lag, best WER) | 1.5 / 10 / 30 / 2 (~60 ms lag) |
| stream slots | 10 | 8 |
| port / name | 9000 / `parakeet-asr-parakeet` | 9001 / `parakeet-asr-canary` |

Notes:

* Each instance is its own compose project (`asr-parakeet`, `asr-canary`), so
  starting or stopping one never touches the other.
* Put them on **different GPUs**: canary bf16 holds ~10 GB, Parakeet fp32 ~5 GB
  (~13 GB worst case with full-context on long audio).
* `make dev-down` stops the hot-reload dev stack (which takes port 9000) before
  you start the production instances.
* Both instances are the same build, so the UI is identical; it talks to the API
  on its own origin, i.e. the UI on :9000 drives Parakeet and :9001 drives Canary.

## ASGI server (granian)

The image no longer runs uvicorn: `granian` (Rust/Hyper) serves the same ASGI app.
With uvicorn 0.54, **every** WebSocket implementation (`websockets-sansio`, the
legacy `websockets` one, `wsproto`) stopped accepting upgrades after a
transducer model (Parakeet-TDT family) had run one inference in the process — the
handshake request arrived truncated (`invalid HTTP request line:
GET/v1/transcribe/stream HTTP/1.1`), so clients hung or got HTTP 400 while HTTP
routes kept working. Reproduced only in the container image; not with Canary, not
on a host venv (related uvicorn WS bugs: #3107, #3175, #2825). With granian the
live path stays healthy (verified: upgrade → session with audio → upgrade →
batch → upgrade, plus 12 concurrent sessions keeping real time).

Keep `--workers 1`: the model pool, the live-session counters and the metrics
registry are all per process.

## Notes

- Each replica is pinned to a GPU and switches to local attention only while
  serving a long-form request, then restores the default attention config.
- GPU assignment in Compose uses `deploy.resources.reservations.devices`; with
  multiple GPUs the default loads one replica per GPU. Set `NUM_WORKERS` to cap
  that.
- The Silero VAD runs entirely in the browser (AudioWorklet thread) using ONNX
  Runtime Web in single-threaded mode — no Cross-Origin-Isolation headers
  required. Inference on 512-sample (32 ms) frames is well within real-time
  budget even on modest hardware.
