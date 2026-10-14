# Parakeet ASR Server

FastAPI service wrapping NVIDIA Parakeet/Canary ASR models
via NeMo. See the repository root `README.md` for the full overview.

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/health` | Liveness + readiness, worker count |
| `POST` | `/v1/transcribe` | Plain-text transcription |
| `POST` | `/v1/transcribe/timestamps` | Transcription with word/segment/char timestamps |
| `POST` | `/v1/transcribe/longform` | Long-form transcription (local-attention windowing) |
| `WS`   | `/v1/transcribe/stream` | Live streaming transcription (PCM-16 frames) |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible transcription; `stream=true` → SSE |
| `POST` | `/v1/audio/translations` | OpenAI-compatible translation to English |
| `GET`  | `/v1/models` | OpenAI-compatible model list |

All `POST` endpoints accept a `multipart/form-data` upload with field name
`file` containing audio in any format libsndfile or ffmpeg can decode (any
sample rate / channel count).

## Local development

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

## Configuration (env vars)

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `MODEL_NAME` | `nvidia/canary-1b-v2` | Model id to load |
| `SERVED_MODEL_NAME` | _(= `MODEL_NAME`)_ | Model id reported by `/v1/models` |
| `NUM_WORKERS` | `0` (one per GPU) | Number of model replicas |
| `DEVICES` | _(auto)_ | Explicit device list, e.g. `cuda:0,cuda:1` |
| `MAX_UPLOAD_MB` | `200` | Max upload size |
| `LONGFORM_CONTEXT` | `256` | Local-attention context window for long-form |
| `SSE_CHUNK_DURATION` | `20.0` | Max seconds of audio per SSE delta |
| `SSE_SPLIT_SEARCH` | `4.0` | Seconds searched for a quiet split point |
