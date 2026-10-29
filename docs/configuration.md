# Configuration

Everything is configured through environment variables. `.env.example` (copied
to `.env` for the dev stack) covers the ones you normally set; everything below
is the full list. The two production profiles
([`.env.parakeet`](../.env.parakeet), [`.env.canary`](../.env.canary)) are
complete worked examples.

## Server

### Model

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `MODEL_NAME` | `nvidia/canary-1b-v2` | HuggingFace model ID to load (also the model label unless `SERVED_MODEL_NAME` is set) |
| `MODEL_PATH` | _(unset)_ | Local `.nemo` checkpoint to load instead; takes precedence over `MODEL_NAME` for loading |
| `SERVED_MODEL_NAME` | _(= `MODEL_NAME`)_ | Model id reported by `GET /v1/models` (e.g. for LiteLLM) |
| `NUM_WORKERS` | `0` (one per GPU) | Number of model replicas |
| `DEVICES` | _(auto)_ | Explicit device list, e.g. `cuda:0,cuda:1` |
| `LONGFORM_CONTEXT` | `256` | Local-attention context window for long-form mode |
| `LONGFORM_AUTO_SECONDS` | `30` | Auto-switch to long-form decoding for audio longer than this (0 disables); prevents full-context attention from pinning GPU memory |
| `DEFAULT_SOURCE_LANG` | _(unset, = `en`)_ | Default source language for models that take language arguments (Canary) |
| `DEFAULT_TARGET_LANG` | _(unset, = source)_ | Default target language; equal to source = transcribe, different = translate |

### Precision / numerics

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `TORCH_DTYPE` | `auto` | Model weight dtype: `auto` (bf16 on compute capability ≥ 8.0, else fp32), `float32`, `bfloat16`, `float16` |
| `MATMUL_PRECISION` | `high` | fp32 matmul precision: `high` = TF32 tensor cores, `medium` (bf16 matmuls), `highest` (strict fp32) |

Transducer models (Parakeet-TDT family) lose accuracy in bf16 on long audio —
use `float32` there. See [benchmarks.md](benchmarks.md#precision--numerics) and
[pitfalls.md](pitfalls.md#bf16-hurts-transducer-models-on-long-audio).

### Batching scheduler

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `BATCH_WINDOW_MS` | `20` | How long the scheduler waits for more requests before decoding a batch (`0` disables batching) |
| `MAX_BATCH_SIZE` | `8` | Max requests decoded in one model call |
| `MAX_BATCH_AUDIO_SECONDS` | `240` | Max total audio (s) per batch |
| `MAX_BATCH_ITEM_SECONDS` | `30` | Requests longer than this run on their own |
| `BATCH_DURATION_RATIO` | `2` | Max longest/shortest duration inside one batch; more mismatched lengths are decoded separately |
| `INLINE_AUDIO_MAX_SECONDS` | `30` | Audio up to this length is handed to the model as an in-memory array instead of a temp WAV |

### Admission control

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `MAX_QUEUE_SIZE` | `32` | Queued requests before new ones get `503` (`0` = unlimited) |
| `QUEUE_TIMEOUT_S` | `120` | Max time a request may wait in the queue |
| `RETRY_AFTER_S` | `2` | `Retry-After` value sent with `503` |
| `MAX_UPLOAD_MB` | `200` | Max batch upload size |
| `MAX_CONCURRENT_STREAMS` | `0` (auto) | Concurrent WebSocket sessions; `0` = 8 in bf16/fp16, 4 in fp32 |
| `STREAM_MAX_QUEUE_SECONDS` | `120` | Unprocessed audio a client may queue before the session is closed (1013) |
| `STREAM_MAX_SESSION_SECONDS` | `3600` | Hard limit on one WebSocket session |

### Live streaming tuning

Model-dependent; these defaults suit an AED model such as Canary, while a
transducer wants the wider settings in [`.env.parakeet`](../.env.parakeet). The
measured effect of each is in
[benchmarks.md](benchmarks.md#live-streaming-tuning).

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `STREAM_MIN_DURATION` | `1.0` | Seconds of audio before the first streaming inference |
| `STREAM_RETRANSCRIBE_INTERVAL` | `0.5` | Min new audio (s) between inference runs |
| `STREAM_STABLE_WORDS` | `1` | Min agreed words required for a commit |
| `STREAM_STABLE_ITERS` | `2` | Consecutive hypotheses that must agree before committing |
| `STREAM_MAX_DURATION` | `30.0` | Hard buffer cap (s) that forces a trim/commit |
| `STREAM_CONTEXT_DURATION` | `1.5` | Committed audio (s) kept before a trim point as context |
| `STREAM_TRIM_DURATION` | `10.0` | Buffer length (s) after which it is trimmed at the last committed sentence end |
| `STREAM_HOLDBACK` | `0.0` | Words ending this close (s) to the buffer end are not committed |

### SSE streaming

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `SSE_CHUNK_DURATION` | `20.0` | Max audio (s) per chunk/delta for `stream=true` |
| `SSE_SPLIT_SEARCH` | `4.0` | Window (s) before the chunk limit searched for a quiet cut point |

### Observability & serving

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `METRICS_ENABLED` | `true` | Expose `GET /metrics` |
| `LOGFIRE_TOKEN` | _(unset)_ | Export traces to Logfire; console-only if unset |
| `STATIC_DIR` | _(unset)_ | Directory of the built frontend to serve; set by the production image |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address when run directly (the images use granian on 9000) |

## Frontend

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `API_PROXY_TARGET` | `http://localhost:9000` (host) / `http://server:9000` (compose) | Backend address the Vite dev server proxies `/v1` and `/health` to |
| `VITE_API_BASE` | _(unset)_ | Override to call a different API origin directly from the browser |

The browser always calls the API at the same origin the page was loaded from;
the Vite proxy makes that work in development, and the production image serves
both from one port.

## Deployment profiles

The Makefile recipes pass a whole profile file with `--env-file`, which
**replaces** `.env` for that invocation (compose interpolation order: shell
environment > `--env-file` > no `.env`). Anything not in the profile falls back
to the defaults above.

| | [`.env.parakeet`](../.env.parakeet) | [`.env.canary`](../.env.canary) |
| --- | --- | --- |
| model | `MODEL_PATH=/cache/hf/parakeet-ultra/ultra.nemo` | `MODEL_NAME=nvidia/canary-1b-v2` (HF cache) |
| precision | `float32` | `auto` → bf16 |
| languages | auto-detect, hints ignored | `DEFAULT_SOURCE_LANG=sk` |
| streaming | 20 / 30 / 60 / 3 iterations | 1.5 / 10 / 30 / 2 iterations |
| stream slots | 10 | 8 |
| port / container | 9000 / `parakeet-asr-parakeet` | 9001 / `parakeet-asr-canary` |

## Language defaults

Canary-style models require a source and a target language; with nothing set they
use `en→en`, which means non-English audio gets **translated into English**
(observed as 113 % WER against a Slovak reference before this was fixed). For a
non-English deployment set:

```bash
DEFAULT_SOURCE_LANG=sk
DEFAULT_TARGET_LANG=sk      # same as source = transcribe
```

Per-request `language` / `source_lang` / `target_lang` still win. Models without
language selection (Parakeet) ignore all of it.
