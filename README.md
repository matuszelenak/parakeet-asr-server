# Parakeet ASR Server

A FastAPI service that serves NVIDIA's **Parakeet/Canary**
(`nvidia/parakeet-tdt-0.6b-v3`) speech-to-text model, plus a Svelte frontend for
recording and transcribing audio in the browser. Designed to run on a node with
H200 GPUs.

```
.
├── server/      # FastAPI + NeMo API (managed with uv)
├── frontend/    # Svelte + TypeScript app (managed with Deno)
└── docker-compose.yml
```

## Features

- **Three transcription endpoints** — plain text, word/segment/char timestamps,
  and long-form (local-attention windowing for very long recordings).
- **Any WAV in** — uploads are decoded, downmixed to mono, and resampled to
  16 kHz server-side (the frontend also normalises before upload).
- **Parallel requests** — one model replica is loaded per GPU; requests check
  out an idle replica from a queue, so N requests run concurrently across N GPUs
  while each replica processes one at a time (NeMo models aren't concurrency
  safe).
- **Browser recorder** — record from the microphone or upload a file, pick a
  mode, and view the transcript (with a segment table + word timings).

## API

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/health` | Readiness, worker count, language capabilities |
| `POST` | `/v1/transcribe` | Plain-text transcription |
| `POST` | `/v1/transcribe/timestamps` | Word / segment / char timestamps |
| `POST` | `/v1/transcribe/longform` | Long-form transcription |

`POST` endpoints take `multipart/form-data` with a `file` field (a WAV file).

```bash
curl -F file=@sample.wav http://localhost:9000/v1/transcribe
curl -F file=@sample.wav http://localhost:9000/v1/transcribe/timestamps
curl -F file=@sample.wav http://localhost:9000/v1/transcribe/longform
```

### Source / target language (model-dependent)

Models that support language selection (e.g. `nvidia/canary-1b-v2`) accept
optional `source_lang` and `target_lang` form fields on every `POST` endpoint.
Setting a `target_lang` different from `source_lang` performs speech
**translation**; matching them performs transcription.

```bash
# transcribe German speech
curl -F file=@de.wav -F source_lang=de -F target_lang=de \
  http://localhost:9000/v1/transcribe
# translate German speech to English
curl -F file=@de.wav -F source_lang=de -F target_lang=en \
  http://localhost:9000/v1/transcribe
```

Whether this is enabled is reported by `GET /health` (`supports_languages` plus
the supported `languages` list); the frontend shows the language pickers only
when the configured model supports them. Supplying these fields to a model that
does not support them returns `400`. Supported languages: Bulgarian, Croatian,
Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek,
Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish, Portuguese, Romanian,
Slovak, Slovenian, Spanish, Swedish, Russian, Ukrainian.

## Quick start (Docker Compose)

Requires Docker with the NVIDIA Container Toolkit (so containers can see the
GPUs).

```bash
cp .env.example .env        # optional, to tweak defaults
docker compose up --build
```

- API:      http://localhost:9000  (docs at `/docs`)
- Frontend: http://localhost:5173

The first boot downloads the model (cached in the `model-cache` volume for
subsequent runs).

### Live-reload development

```bash
docker compose watch
```

- Editing files in `frontend/src` hot-reloads via Vite HMR.
- Editing files in `server/app` syncs and restarts the API.
- Changing `pyproject.toml` / `package.json` / `deno.json` triggers a rebuild.

## Production image (single container)

`prod.Dockerfile` is a multi-stage build that compiles the Svelte frontend and
bakes it into the server image. FastAPI then serves the static UI **and** the
API from a single port — no separate frontend container or proxy needed (the
browser calls the API at its own origin).

```bash
# build from the repo root (note the build context is `.`)
docker build -f prod.Dockerfile -t parakeet-asr:prod .

# run (needs the NVIDIA Container Toolkit)
docker run --gpus all -p 9000:9000 \
  -v /models/huggingface:/cache/hf -v /cache/torch:/cache/torch \
  parakeet-asr:prod
```

Or use the production compose file (GPU reservation + model-cache volumes):

```bash
docker compose -f docker-compose.prod.yml up --build -d
```

Open <http://localhost:9000> for the UI; the API and `/docs` live on the same
origin. Static serving is controlled by `STATIC_DIR` (set to `/app/static` in
the image); it stays empty in dev, so the Vite dev server keeps serving the UI
there.

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
deno install
deno task dev        # http://localhost:5173
# deno task build    # production bundle into dist/
# deno task check    # type-check
```

## Configuration

Server environment variables:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `MODEL_NAME` | `nvidia/parakeet-tdt-0.6b-v3` | Model id to load |
| `NUM_WORKERS` | `0` (one per GPU) | Number of model replicas |
| `DEVICES` | _(auto)_ | Explicit device list, e.g. `cuda:0,cuda:1` |
| `MAX_UPLOAD_MB` | `200` | Max upload size |
| `LONGFORM_CONTEXT` | `256` | Local-attention context for long-form |
| `LOGFIRE_TOKEN` | _(unset)_ | Ship logs/traces to Logfire; console-only when unset |

Frontend variables:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `API_PROXY_TARGET` | `http://localhost:9000` (host) / `http://server:9000` (compose) | Backend address the dev server proxies API calls to |
| `VITE_API_BASE` | _(unset)_ | Optional override to call a different API origin directly, bypassing the same-origin proxy |

The browser always calls the API at the **same origin** the page was opened from
(derived from `window.location`); the dev server (Vite) proxy `/v1` and `/health` to the backend. This means it works unchanged
whether you open the app on `localhost` or via the node's hostname/IP.

## Observability

Logging and tracing use [Logfire](https://logfire.pydantic.dev/). FastAPI is
instrumented (a span per request), each transcription runs inside a `transcribe`
span, and stdlib logging (uvicorn, NeMo, ...) is routed through Logfire too. Set
`LOGFIRE_TOKEN` to export to the Logfire backend; without it, output still
renders to the console.

## Notes

- Each replica is pinned to a GPU and switches to local attention only while
  serving a long-form request, then restores the default attention config.
- GPU access in Compose uses the `deploy.resources.reservations.devices` form;
  with 4× H200 the default loads 4 replicas (~one per GPU). Set `NUM_WORKERS` to
  cap that if you want fewer.
