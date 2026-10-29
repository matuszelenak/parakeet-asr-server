# API reference

All endpoints share one loaded model per replica. `POST` endpoints take
`multipart/form-data` with a `file` field (any common audio format).

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET`  | `/health` | Readiness, worker count, model, precision, queue/batch/stream state |

`/health.model` is what is actually loaded (a HuggingFace id or a local `.nemo`
path — an operator detail), while `/health.served_model` repeats the id that
`/v1/models` advertises. Client UIs should label themselves with
`served_model`. Full field list and the metrics on `/metrics`:
[deployment.md](deployment.md#observability).
| `GET`  | `/metrics` | Prometheus metrics (see [deployment.md](deployment.md#observability)) |
| `POST` | `/v1/transcribe` | Plain-text transcription |
| `POST` | `/v1/transcribe/timestamps` | Word / segment / char timestamps |
| `POST` | `/v1/transcribe/longform` | Long-form transcription (local attention) |
| `WS`   | `/v1/transcribe/stream` | Real-time streaming transcription |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible transcription (+ SSE streaming) |
| `POST` | `/v1/audio/translations` | OpenAI-compatible translation to English |
| `GET`  | `/v1/models` | OpenAI-compatible model list |
| `GET`  | `/docs` | Swagger UI |

```bash
curl -F file=@sample.wav http://localhost:9000/v1/transcribe
curl -F file=@sample.wav http://localhost:9000/v1/transcribe/timestamps
curl -F file=@sample.wav http://localhost:9000/v1/transcribe/longform
```

Timestamps come back as words (and segments); a model that only reports
char-level frames is normalised to words by the server.

## OpenAI-compatible endpoints

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
models. See [Source / target language](#source--target-language).

Errors on these routes use the OpenAI envelope
(`{"error": {"message", "type", "param", "code"}}`) so OpenAI SDKs and LiteLLM
surface the real message.

```bash
curl http://localhost:9000/v1/audio/transcriptions \
  -F file=@sample.m4a -F model=parakeet -F language=sk
```

### SSE streaming (`stream=true`)

With `stream=true` (and `response_format` `json`/`text`; it is ignored for the
timestamped formats, as on OpenAI) the response is `text/event-stream` in
OpenAI's `gpt-4o-transcribe` format:

```
data: {"type":"transcript.text.delta","delta":"Hello there."}

data: {"type":"transcript.text.delta","delta":" How are you?"}

data: {"type":"transcript.text.done","text":"Hello there. How are you?"}
```

Parakeet/Canary are not incremental models, so the audio is split into chunks of
at most `SSE_CHUNK_DURATION` seconds, cut at the quietest point near the limit
(`SSE_SPLIT_SEARCH`). Each chunk is transcribed in turn and emitted as one
delta, so the first text arrives after one chunk rather than after the whole
file. An error after the stream has started is sent as a
`data: {"error": {...}}` event, which the OpenAI SDKs raise as an exception.

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

## LiteLLM proxy

[`litellm/config.yaml`](../litellm/config.yaml) is a ready-to-use example. It
exposes the server in two ways:

1. **As a regular model** (`model_name: parakeet`, provider `openai/` with
   `api_base` pointing at this server). Clients call LiteLLM's standard
   `POST /v1/audio/transcriptions` with `model=parakeet` and get keys, budgets,
   logging, load balancing and health checks (`mode: audio_transcription`). All
   response formats work. LiteLLM sends `response_format=verbose_json` by
   default so that it can read `duration` for cost tracking.
2. **As a pass-through route** (`/parakeet/...` → this server, with LiteLLM key
   auth). This is how SSE streaming and the native endpoints reach clients
   through LiteLLM, e.g. `base_url="http://litellm:4000/parakeet/v1"`.

```bash
litellm --config litellm/config.yaml --port 4000
curl http://localhost:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-change-me" -F model=parakeet -F file=@sample.wav
```

> **Streaming caveat (LiteLLM ≤ 1.105.1, tested):** LiteLLM's own
> `/v1/audio/transcriptions` route does not stream. It drops `stream` and returns
> the complete JSON response. Its pass-through route forwards the SSE events
> intact (the OpenAI SDK's `stream=True` works through it), but for
> `multipart/form-data` uploads it buffers the upstream response, so all the
> deltas arrive together when transcription finishes. For truly progressive
> streaming, call the server directly. The WebSocket live endpoint is not proxied
> by LiteLLM either; expose it directly or through a plain reverse proxy.

## WebSocket streaming protocol

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

A `{"type": "end"}` sent as the very first frame is honoured (it finalises an
empty session instead of being swallowed).

This endpoint is for **live** audio. A client that pushes faster than the model
can decode is closed with code 1013 once `STREAM_MAX_QUEUE_SECONDS` of audio is
queued; use SSE (above) to feed a file as fast as the server can take it. Session
level limits: `MAX_CONCURRENT_STREAMS`, `STREAM_MAX_SESSION_SECONDS`.

## Source / target language

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
them is **ignored** (no 400), so pointing the same client at a different model
family does not start failing requests; `_resolve_languages(..., strict=True)`
exists for callers that want validation, but no route exposes it.

`DEFAULT_SOURCE_LANG` / `DEFAULT_TARGET_LANG` set the deployment default for
models that take language arguments — important because Canary defaults to
`en→en`, which *translates* your audio into English. See
[configuration.md](configuration.md#language-defaults).
