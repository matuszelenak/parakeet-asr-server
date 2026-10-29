# Parakeet ASR Server

FastAPI service wrapping NVIDIA Parakeet/Canary ASR models via NeMo.
Overview, deployment and benchmarks live in the
[repository README](../README.md) and [docs/](../docs).

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/health` | Readiness, model, precision, queue/batch/stream state |
| `GET`  | `/metrics` | Prometheus metrics |
| `POST` | `/v1/transcribe` | Plain-text transcription |
| `POST` | `/v1/transcribe/timestamps` | Transcription with word/segment/char timestamps |
| `POST` | `/v1/transcribe/longform` | Long-form transcription (local-attention windowing) |
| `WS`   | `/v1/transcribe/stream` | Live streaming transcription (PCM-16 frames) |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible transcription; `stream=true` → SSE |
| `POST` | `/v1/audio/translations` | OpenAI-compatible translation to English |
| `GET`  | `/v1/models` | OpenAI-compatible model list |

All `POST` endpoints accept a `multipart/form-data` upload with field name
`file` containing audio in any format libsndfile or ffmpeg can decode (any
sample rate / channel count). Fields and response formats:
[docs/api.md](../docs/api.md).

## Local development

```bash
uv sync
uv run granian --interface asgi --host 0.0.0.0 --port 9000 app.main:app
```

The containers use granian for a reason: uvicorn 0.54 stops completing
WebSocket upgrades once a transducer model has run an inference in the process,
so live sessions only work once. It is fine for HTTP-only work — see
[docs/pitfalls.md](../docs/pitfalls.md#uvicorn-drops-websocket-upgrades-after-transducer-inference).

## Configuration

Every variable, with defaults, is documented in
[docs/configuration.md](../docs/configuration.md); `../.env.example` is the
short copy-and-edit version.
