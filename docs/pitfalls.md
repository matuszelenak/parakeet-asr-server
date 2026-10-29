# Pitfalls

Non-obvious failures encountered while building and benchmarking this server, and
what the code does about them. Most of them cost hours; none of them are
theoretical.

## uvicorn drops WebSocket upgrades after transducer inference

**Symptom**: live transcription works exactly once per process. The first session
connects, streams and finishes; every later upgrade hangs or returns 400, while
all HTTP routes keep working.

**Cause**: with uvicorn 0.54, once a transducer model (Parakeet-TDT family) has
run an inference in the process, the upgrade request reaches the server truncated
as `invalid HTTP request line: GET/v1/transcribe/stream HTTP/1.1`. This happened
with every WebSocket implementation tried (`websockets-sansio`, legacy
`websockets`, `wsproto`, plus the h11 and asyncio-loop setups), was not present
with Canary, and was not present in a host venv. Related upstream reports:
uvicorn #3107, #3175, #2825.

**Fix**: the images run `granian` (Rust/Hyper) instead, with `--workers 1`.
Verified sequence: upgrade → session with audio → upgrade → batch request →
upgrade → 12 concurrent sessions keeping real time. The interpreter is pinned to
Python 3.12 in both images so the runtime the live path was verified on is the
one that ships.

## bf16 hurts transducer models on long audio

**Symptom**: WER jumps from 10.4 % to 14.5 % after switching a Parakeet model to
bf16, with no error anywhere.

**Cause**: a transducer accumulates errors over a long decode; the precision loss
in the encoder is not absorbed by an autoregressive AED decoder the way it is in
Canary (where bf16 was free, see [benchmarks.md](benchmarks.md#precision--numerics)).

**Fix**: `TORCH_DTYPE=float32` for transducer models (`.env.parakeet`), bf16 for
AED models. Feeding shorter chunks is the alternative (30 s chunks in bf16 scored
9.7 %). `TORCH_DTYPE=auto` cannot know the model family, so it is a per-deployment
decision.

Also use `longform=true` for long audio with a transducer: bounded local
attention is both more accurate (8.7 % vs 10.4 % WER) and cheaper (~4 GB vs ~13 GB
VRAM), because whole-file full-context attention is what makes it slow and
memory hungry.

## Canary translates instead of transcribing by default

**Symptom**: a Slovak file comes back as English text, WER against the Slovak
reference 113 %.

**Cause**: Canary is an AST model whose default is `en→en`; with a target
language different from the source it translates. A request that sends no
language fields therefore gets *translation*, which looks like catastrophic ASR.

**Fix**: `DEFAULT_SOURCE_LANG` / `DEFAULT_TARGET_LANG` (`.env.canary` sets both to
`sk`). Per-request `language` / `source_lang` / `target_lang` still override.

## Language hints break a model swap without leniency

**Symptom**: after swapping Canary for Parakeet, every client that sent
`language=sk` gets 400 — including this project's own UI, its scripts and LiteLLM
configs, and clients using the WebSocket endpoint fail the upgrade.

**Cause**: transducers auto-detect the language and take no language arguments,
but the endpoints validated hints strictly.

**Fix**: `_resolve_languages(..., strict=False)` is the default — hints are
ignored for models that do not take them. `GET /health` advertises
`supports_languages` so clients can hide the selectors.

## Batch padding changes the tail of a shorter clip

**Symptom**: occasionally the last word or two of a clip differs between a batched
request and the same clip decoded alone.

**Cause**: NeMo pads all items in a forward pass to the longest one, and that can
change the tail hypothesis of the shorter clip. It happens with the file/lhotse
path too — it is inherent to batched NeMo decoding, not to the scheduler. Greedy
decoding plus 1:1 order mapping means a request always gets *its own* transcript.

**Fix**: `BATCH_DURATION_RATIO` (default 2×) keeps mismatched lengths in separate
batches, and identical-length inputs are unaffected. Set it to `1` to batch only
equal durations, or `0` to disable the check.

## NeMo rebuilds attention modules in fp32 on long-form toggle

**Symptom**: `RuntimeError: expected scalar type BFloat16 but found Float` on
the first long-form request after a batch request in a bf16 replica.

**Cause**: NeMo swaps the encoder's attention implementation when long-form mode
is toggled and recreates the modules in fp32.

**Fix**: the worker re-applies the configured dtype after each attention switch
(`model_pool._apply_dtype`), and restores the default attention config after the
long-form call.

## SSE chunks stay on the temp-WAV path

**Symptom**: an experiment to pass audio to the model as an in-memory array
instead of a temp WAV left 4 of 31 SSE chunks with a different transcript.

**Cause**: NeMo's chunked long-form decoding is applied on the file path; the
array path takes a different decoding route, so chunk boundaries and context
handling differ slightly.

**Fix**: array input is used only where it is byte-comparable — batch requests up
to `INLINE_AUDIO_MAX_SECONDS` and the streaming buffers. SSE keeps writing chunk
WAVs, trading ~16 % of an inference for identical output.

## The WebSocket endpoint is not a bulk upload path

**Symptom**: a script pushing a whole file over `/v1/transcribe/stream` gets
closed with code 1013.

**Cause**: it is an admission control for *live* audio — a client that pushes
faster than the model decodes accumulates unbounded backlog and latency.

**Fix**: `STREAM_MAX_QUEUE_SECONDS` (default 120 s of unprocessed audio) closes
the session with 1013 instead. For files use
`POST /v1/audio/transcriptions` with `stream=true` (SSE), which is paced by the
server.

## LiteLLM buffers (and drops) SSE

**Symptom**: `stream=true` through LiteLLM either returns one complete JSON body
or delivers all deltas at once.

**Cause** (LiteLLM ≤ 1.105.1, tested): its own `/v1/audio/transcriptions` route
drops `stream`; its pass-through route forwards SSE correctly but buffers
`multipart/form-data` uploads until the upstream response completes. The
WebSocket endpoint is not proxied at all.

**Fix**: none server-side — call the server directly for progressive streaming,
or accept whole-response behaviour through LiteLLM. Documented in
[api.md](api.md#litellm-proxy).

## The VAD model made the first live session feel broken

**Symptom**: clicking the mic took several seconds before "Listening…" on a cold
cache, every visit.

**Cause**: the browser VAD pulls `ort-wasm-simd-threaded.wasm` (13.6 MB) and
`silero_vad_v5.onnx` (2.2 MB) from this server. They were served uncompressed and
without `Cache-Control`, and the fetch only started on click — 16.5 MB of pure
latency after the button press.

**Fix**: gzip them at image build (16.5 MB → 5.6 MB on the wire), a week of
`Cache-Control`, and `prewarmVad()` prefetches them when the Live tab opens so the
transfer overlaps with the user picking up the mic. Compressing per request was
rejected: ~1 s of the single event loop would stall live streams.

## Compose happily recreates a container instead of failing

**Symptom**: `make run-canary` while a Parakeet instance was running silently
moved the Canary container onto the Parakeet GPU.

**Cause**: both recipes shared a `GPU=0` default; changing `device_ids` makes
compose *recreate* the container, so the old GPU assignment was replaced without
any error.

**Fix**: per-instance defaults (`GPU_PARAKEET=0`, `GPU_CANARY=1`), an explicit
`GPU=<n>` still wins, and `run-both` refuses to start both on the same GPU. Note
also that `--env-file` **replaces** `.env` for compose (shell environment >
`--env-file` > `.env`), which is why each profile is complete on its own.

## YouTube auto-captions are a noisy reference

**Symptom**: two configs differing by 9 % relative WER, where listening to both
transcripts suggests they are equally good.

**Cause**: the auto-caption reference itself carries ~4.5 WER points of error
(hand-adjudicated over a 120 s window).

**Fix**: treat sub-1-point differences as noise, and adjudicate a window by hand
before drawing conclusions. Comparisons across model families are still valid —
the transducer/AED gaps were 3–5 points.

## audio.cpp quirks (observed during the comparison)

Not this project's bugs, but they shaped the comparison in
[benchmarks.md](benchmarks.md#against-audiocpp):

- `full_context` aborts on audio longer than ~30 s ("Parakeet TDT encoder relative
  position frames exceed maximum"); the usable mode is the windowed long-form one.
- f16 weights produced garbage (661 % WER) — use f32.
- Streaming is hard-limited to one slot (`does not support the requested slots …
  (capacity=1)`), so its live mode cannot serve concurrent sessions at all.
