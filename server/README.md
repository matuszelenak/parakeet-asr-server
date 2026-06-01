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

All `POST` endpoints accept a `multipart/form-data` upload with field name
`file` containing a WAV file (any sample rate / channel count).

## Local development

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

## Configuration (env vars)

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `MODEL_NAME` | `nvidia/parakeet-tdt-0.6b-v3` | Model id to load |
| `NUM_WORKERS` | `0` (one per GPU) | Number of model replicas |
| `DEVICES` | _(auto)_ | Explicit device list, e.g. `cuda:0,cuda:1` |
| `MAX_UPLOAD_MB` | `200` | Max upload size |
| `LONGFORM_CONTEXT` | `256` | Local-attention context window for long-form |
