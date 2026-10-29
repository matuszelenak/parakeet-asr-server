# Benchmarks

Every number here was measured on the reference machine and test material below.
They are indicative, not guarantees: the transducer/AED difference in particular
is a property of the models, not of the server.

## Test setup

| | |
| --- | --- |
| GPU | 2 × RTX 3090 (24 GB), one replica per GPU |
| Audio | 559 s Slovak podcast (`Ekonomický newsfilter`), 16 kHz mono |
| Reference | YouTube auto-captions |
| Metric | WER / CER after normalisation (lowercase, punctuation stripped) |

**Reference quality**: the auto-caption reference itself is noisy — hand
adjudication of a 120 s window put its own error at roughly **4.5 WER points**.
Differences below ~1 point between two *good* configurations are therefore
inside noise; the tables are useful for spotting the 3–5 point regressions
(bf16 on a transducer, translating instead of transcribing), not for ranking
near-identical settings.

## Concurrency & capacity

Batching scheduler on one RTX 3090 with `nvidia/canary-1b-v2`:

| scenario | eager fp32, no batching | scheduler (fp32) | scheduler + bf16 |
| -------- | ----------------------- | ---------------- | ---------------- |
| 20 s clips, 4 concurrent | 2.5 req/s, 1.0 s latency | 5.1 req/s, 0.78 s | **7.6 req/s, 0.52 s** |
| 20 s clips, 8 concurrent | 2.5 req/s, 1.9 s latency | 6.4 req/s, 1.2 s | **11.8 req/s, 0.68 s** |
| 559 s file, `/v1/transcribe/longform` | 6.5 s (86× RT) | 6.5 s | **3.5 s (160× RT)** |
| 559 s file, GPU-only (no HTTP) | 110× RT | 133× RT | **336× RT** |
| live streams on one GPU | 2 keep up, 3rd falls behind | 4 keep up | **12 keep up** |
| overload (45 concurrent) | unbounded queueing | 12 fast `503`s + `Retry-After` | same |

Parakeet Ultra, fp32, same file:

| | |
| --- | --- |
| 20 s clips × 8 concurrent | 23.5 req/s, 471 audio-s/s |
| 559 s file, `/v1/transcribe` | 3.4–4.3 s (130–180× RT), WER 10.4–10.9 % |
| 559 s file, `/v1/transcribe/longform` | **2.2 s (254× RT), WER 8.33 %, CER 2.18 %** |

## Precision / numerics

`canary-1b-v2`, 559 s file, against the word-timed reference:

| setting | RTFx (GPU-only) | WER | CER | transcript |
| ------- | --------------- | --- | --- | ---------- |
| fp32 (strict) | 110× | 10.97 % | 4.26 % | baseline |
| fp32 + TF32 (`MATMUL_PRECISION=high`) | 208× | 10.97 % | 4.26 % | **byte-identical** |
| bf16 | 336× | 10.79 % | 4.19 % | ~10 character spans differ |
| fp16 | 331× | 11.07 % | 4.29 % | ~2 character spans differ |

TF32 is the free win — same output, ~1.9× the throughput — and is on by default.
Quality differences between fp32 and bf16 are at reference-noise level (~1 word
in 1 050), so bf16 is the default for AED models.

**This does not transfer to transducers**: see the Parakeet table below.

## Model comparison

`MODEL_PATH` / `MODEL_NAME` swap the model with no code change. Same 559 s
Slovak file:

| | canary-1b-v2 (bf16) | parakeet-ultra (fp32) | parakeet-ultra (bf16) |
| --- | --- | --- | --- |
| whole file, GPU-only | 336× RT | 226× RT | 364× RT |
| 8 × 20 s batch | 328× RT | **927× RT** | – |
| 20 s window + timestamps (streaming shape) | 276 ms | **42 ms** | – |
| WER vs reference | 11.26 % | 10.41 % | 14.47 % |
| CER vs reference | 4.10 % | **2.99 %** | 4.68 % |
| translation (AST) | yes | no | no |

Parakeet Ultra = `moondream/parakeet-ultra` (weights published as
`ncannings/parakeet-ultra-nemo`), a post-trained `parakeet-tdt-0.6b-v3` covering
25 European languages.

Takeaways:

- The transducer is 2.5–6× faster per inference and its punctuation /
  capitalisation is better, at comparable accuracy (lower CER, WER within
  reference noise). It cannot translate.
- **bf16 hurts the transducer on long audio** (10.4 % → 14.5 % WER whole-file).
  Use `TORCH_DTYPE=float32`, or feed it shorter chunks (30 s chunks in bf16
  scored 9.7 %).
- With `longform=true` Parakeet gets both better and cheaper on long audio:
  **WER 8.70 % / CER 3.0 % at 211× realtime**, VRAM ~4 GB instead of ~13 GB for
  whole-file full-context attention (both measured before TF32 was enabled; a
  production run of the same settings measured 8.33 % / 2.18 % CER at 254× RT).
  Use long-form for anything longer than a couple of minutes with this model.

To try Parakeet Ultra yourself:

```bash
# once (~2.5 GB)
python -c "from huggingface_hub import hf_hub_download as d; \
           d('ncannings/parakeet-ultra-nemo', 'ultra.nemo', local_dir='/models/parakeet-ultra')"

MODEL_PATH=/models/parakeet-ultra/ultra.nemo \
MODEL_NAME=parakeet-ultra \
TORCH_DTYPE=float32 \
docker compose up -d server
```

### Against audio.cpp

The same file was also run through [audio.cpp](https://github.com/ggml-org/audio.cpp)
with Parakeet weights, for reference:

| | this server (parakeet fp32) | audio.cpp |
| --- | --- | --- |
| long-form, 8 s windows | 254× RT | 260× RT |
| WER | 8.33 % | 9.65 % |
| CER | 2.18 % | 2.55 % |
| live streaming | 12 sessions / GPU, 10 by default | 1 session, lag 2–3.6 s (WER 7.50 % / CER 2.40 %) |
| timestamps | words + segments | words |
| translation | via Canary | no |

Roughly the same throughput per GPU on batch work; this server's advantages are
concurrent live sessions, word/segment timestamps and admission control.
Its quirks (`full_context` failing on >30 s audio, f16 weights producing
garbage, streaming hard-limited to one slot) are in
[pitfalls.md](pitfalls.md#audiocpp-quirks-observed-during-the-comparison).

## Live streaming tuning

The four streaming knobs interact strongly and the right values depend on the
model family. Measured on a 120 s excerpt (real-time paced client):

| config | WER | CER | commit lag p50 |
| ------ | --- | --- | -------------- |
| Canary, `1.5 / 10 / 30 / 2` (defaults) | 9.17–9.56 % | 3.38–3.72 % | ~0.06 s |
| Parakeet, `1.5 / 10 / 30 / 2` | 9.58–12.30 % | 2.64–3.64 % | ~0.06 s |
| Parakeet, `20 / 30 / 60 / 3` (shipped in `.env.parakeet`) | **6.25 %** | **2.23 %** | 0.585 s |

The transducer needs much more context before its hypothesis stabilises: wider
`STREAM_CONTEXT_DURATION`, a longer buffer before trimming, a higher
`STREAM_MAX_DURATION`, and one more agreeing hypothesis. The trade is ~0.5 s of
extra commit latency. The AED model is the other way round — short context,
commits almost immediately, and quality does not improve with wider windows.

`STREAM_HOLDBACK` was not worth a latency cost in either case.

## Client-side VAD cost

The browser-side Silero VAD (ONNX Runtime Web, single-threaded, 512-sample
frames) was measured as a download problem, not a compute problem:

| asset | raw | gzip (shipped) |
| ----- | --- | -------------- |
| `ort-wasm-simd-threaded.wasm` | 13.6 MB | 3.5 MB |
| `silero_vad_v5.onnx` | 2.2 MB | 1.85 MB |
| worklet bundle | 2.4 KB | – |

Hence the build-time precompression, the week-long `Cache-Control`, and the
prefetch that starts when the Live tab opens. See
[pitfalls.md](pitfalls.md#the-vad-model-made-the-first-live-session-feel-broken).
