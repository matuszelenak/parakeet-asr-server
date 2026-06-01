<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { Recorder } from './lib/recorder'
  import { blobToWav16kMono } from './lib/wav'
  import {
    transcribe,
    fetchCapabilities,
    StreamTranscriber,
    type Mode,
    type TranscriptionResult,
    type LanguageInfo,
    type StreamEvent,
  } from './lib/api'
  import { LANGUAGES, DEFAULT_LANGUAGE } from './lib/languages'

  const MODES: { value: Mode; label: string; hint: string }[] = [
    { value: 'plain', label: 'Plain text', hint: 'Just the transcript' },
    { value: 'timestamps', label: 'Timestamped', hint: 'Word & segment timings' },
    { value: 'longform', label: 'Long form', hint: 'For very long recordings' },
  ]

  let mode = $state<Mode>('plain')
  let recording = $state(false)
  let busy = $state(false)
  let error = $state<string | null>(null)
  let result = $state<TranscriptionResult | null>(null)
  let audioUrl = $state<string | null>(null)

  // Language selection — only shown when the configured model supports it.
  let supportsLanguages = $state(false)
  let languages = $state<LanguageInfo[]>(LANGUAGES)
  let sourceLang = $state<string>(DEFAULT_LANGUAGE)
  let targetLang = $state<string>(DEFAULT_LANGUAGE)
  let isTranslation = $derived(supportsLanguages && sourceLang !== targetLang)

  let recorder: Recorder | null = null
  let wavBlob: Blob | null = null

  // ── Streaming state ─────────────────────────────────────────────────────────
  let streaming = $state(false)
  let streamError = $state<string | null>(null)
  let committedText = $state('')
  let partialText = $state('')
  let streamTranscriber: StreamTranscriber | null = null

  function handleStreamEvent(event: StreamEvent) {
    if (event.type === 'partial') {
      partialText = event.text
    } else {
      if (event.text) committedText = committedText ? committedText + ' ' + event.text : event.text
      partialText = ''
      if (event.type === 'final') cleanupStream()
    }
  }

  function cleanupStream() {
    streamTranscriber?.close()
    streamTranscriber = null
    streaming = false
  }

  async function startStream() {
    streamError = null
    committedText = ''
    partialText = ''

    const t = new StreamTranscriber(
      handleStreamEvent,
      (msg) => { streamError = msg; streaming = false; streamTranscriber = null },
      () => { if (streamTranscriber === t) { streamTranscriber = null; streaming = false } },
    )

    try {
      const opts = supportsLanguages ? { sourceLang, targetLang } : {}
      await t.start(opts)
      streamTranscriber = t
      streaming = true
    } catch (e) {
      streamError = (e as Error).message
      t.close()
    }
  }

  function stopStream() {
    if (!streamTranscriber) return
    streaming = false
    streamTranscriber.stop()
  }

  onDestroy(() => {
    streamTranscriber?.stop()
    streamTranscriber?.close()
  })

  onMount(async () => {
    try {
      const caps = await fetchCapabilities()
      supportsLanguages = caps.supportsLanguages
      if (caps.languages.length) languages = caps.languages
    } catch {
      // Capabilities are best-effort; default to no language selection.
    }
  })

  function reset() {
    error = null
    result = null
  }

  async function startRecording() {
    reset()
    recorder = new Recorder()
    try {
      await recorder.start()
      recording = true
    } catch (e) {
      error = `microphone error: ${(e as Error).message}`
      recorder = null
    }
  }

  async function stopRecording() {
    if (!recorder) return
    recording = false
    busy = true
    try {
      const raw = await recorder.stop()
      await prepare(raw)
    } catch (e) {
      error = (e as Error).message
    } finally {
      recorder = null
      busy = false
    }
  }

  async function onFile(event: Event) {
    const input = event.target as HTMLInputElement
    const file = input.files?.[0]
    if (!file) return
    reset()
    busy = true
    try {
      await prepare(file)
    } catch (e) {
      error = (e as Error).message
    } finally {
      busy = false
    }
  }

  async function prepare(raw: Blob) {
    wavBlob = await blobToWav16kMono(raw)
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    audioUrl = URL.createObjectURL(wavBlob)
    result = nullz
  }

  async function runTranscription() {
    if (!wavBlob) return
    reset()
    busy = true
    try {
      const options = supportsLanguages ? { sourceLang, targetLang } : {}
      result = await transcribe(wavBlob, mode, options)
    } catch (e) {
      error = (e as Error).message
    } finally {
      busy = false
    }
  }

  function fmt(t: number): string {
    const m = Math.floor(t / 60)
    const s = (t % 60).toFixed(2).padStart(5, '0')
    return `${m}:${s}`
  }
</script>

<main>
  <header>
    <h1>🦜 Parakeet/Canary ASR</h1>
    <p>Record or upload audio and transcribe it with NVIDIA Parakeet/Canary.</p>
  </header>

  <section class="card stream-section">
    <h2>Live Transcription</h2>

    <div class="controls">
      {#if streaming}
        <button class="record stop" onclick={stopStream}>⏹ Stop</button>
        <span class="status pulse">Listening…</span>
      {:else}
        <button class="record" onclick={startStream}>🎙 Start</button>
      {/if}
    </div>

    <div class="live-transcript" class:live-transcript--active={streaming}>
      {#if committedText && partialText}
        <strong>{committedText}</strong> <em>{partialText}</em>
      {:else if committedText}
        <strong>{committedText}</strong>
      {:else if partialText}
        <em>{partialText}</em>
      {:else}
        <span class="placeholder">Transcript will appear here as you speak…</span>
      {/if}
    </div>

    {#if streamError}
      <p class="error">⚠ {streamError}</p>
    {/if}
  </section>

  <section class="card">
    <fieldset class="modes">
      <legend>Mode</legend>
      {#each MODES as m (m.value)}
        <label class="mode" class:selected={mode === m.value}>
          <input type="radio" name="mode" value={m.value} bind:group={mode} />
          <span class="mode-label">{m.label}</span>
          <span class="mode-hint">{m.hint}</span>
        </label>
      {/each}
    </fieldset>

    {#if supportsLanguages}
      <div class="langs">
        <label>
          <span>Source language</span>
          <select bind:value={sourceLang}>
            {#each languages as lang (lang.code)}
              <option value={lang.code}>{lang.name}</option>
            {/each}
          </select>
        </label>
        <span class="arrow" class:translate={isTranslation}>→</span>
        <label>
          <span>Target language</span>
          <select bind:value={targetLang}>
            {#each languages as lang (lang.code)}
              <option value={lang.code}>{lang.name}</option>
            {/each}
          </select>
        </label>
      </div>
      {#if isTranslation}
        <p class="hint-line">Translating speech from source to target language.</p>
      {/if}
    {/if}

    <div class="controls">
      {#if recording}
        <button class="record stop" onclick={stopRecording}>
          ⏹ Stop recording
        </button>
      {:else}
        <button class="record" onclick={startRecording} disabled={busy}>
          🎙 Record
        </button>
      {/if}

      <label class="upload">
        📁 Upload WAV
        <input type="file" accept="audio/wav,audio/*" onchange={onFile} disabled={busy || recording} />
      </label>
    </div>

    {#if recording}
      <p class="status pulse">Recording… speak now.</p>
    {/if}

    {#if audioUrl}
      <audio controls src={audioUrl}></audio>
      <button class="primary" onclick={runTranscription} disabled={busy || recording}>
        {busy ? 'Transcribing…' : 'Transcribe'}
      </button>
    {/if}

    {#if error}
      <p class="error">⚠ {error}</p>
    {/if}
  </section>

  {#if result}
    <section class="card result">
      <h2>Transcript</h2>
      <p class="transcript">{result.text || '(empty)'}</p>

      {#if result.segments && result.segments.length}
        <h3>Segments</h3>
        <table>
          <thead>
            <tr><th>Start</th><th>End</th><th>Text</th></tr>
          </thead>
          <tbody>
            {#each result.segments as seg (seg.start + seg.segment)}
              <tr>
                <td class="time">{fmt(seg.start)}</td>
                <td class="time">{fmt(seg.end)}</td>
                <td>{seg.segment}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}

      {#if result.words && result.words.length}
        <details>
          <summary>{result.words.length} word timestamps</summary>
          <div class="words">
            {#each result.words as w (w.start + w.word)}
              <span class="word" title={`${fmt(w.start)} – ${fmt(w.end)}`}>{w.word}</span>
            {/each}
          </div>
        </details>
      {/if}
    </section>
  {/if}
</main>

<style>
  main {
    max-width: 720px;
    margin: 0 auto;
    padding: 2rem 1rem 4rem;
  }

  header h1 {
    margin: 0 0 0.25rem;
    font-size: 1.8rem;
  }
  header p {
    margin: 0 0 1.5rem;
    color: #9aa0ad;
  }

  .card {
    background: #181b22;
    border: 1px solid #262b36;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
  }

  .modes {
    border: none;
    padding: 0;
    margin: 0 0 1rem;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
  }
  .modes legend {
    color: #9aa0ad;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
  }
  .mode {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: 0.6rem;
    border: 1px solid #2c323d;
    border-radius: 8px;
    cursor: pointer;
  }
  .mode.selected {
    border-color: #5b8cff;
    background: #1b2230;
  }
  .mode input {
    display: none;
  }
  .mode-label {
    font-weight: 600;
  }
  .mode-hint {
    font-size: 0.75rem;
    color: #9aa0ad;
  }

  .langs {
    display: flex;
    align-items: flex-end;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }
  .langs label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.8rem;
    color: #9aa0ad;
    flex: 1;
  }
  .langs select {
    font: inherit;
    color: #e7e9ee;
    background: #222733;
    border: 1px solid #2c323d;
    border-radius: 8px;
    padding: 0.5rem;
  }
  .arrow {
    padding-bottom: 0.5rem;
    color: #5b6472;
  }
  .arrow.translate {
    color: #5b8cff;
  }
  .hint-line {
    margin: -0.5rem 0 1rem;
    font-size: 0.8rem;
    color: #5b8cff;
  }

  .controls {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    flex-wrap: wrap;
  }

  button,
  .upload {
    font: inherit;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    border: 1px solid #2c323d;
    background: #222733;
    color: #e7e9ee;
    cursor: pointer;
  }
  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .record {
    background: #2a3140;
  }
  .record.stop {
    background: #b3322c;
    border-color: #b3322c;
  }
  .primary {
    margin-top: 1rem;
    display: block;
    background: #5b8cff;
    border-color: #5b8cff;
    color: #0f1115;
    font-weight: 600;
  }
  .upload {
    position: relative;
    overflow: hidden;
  }
  .upload input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }

  audio {
    display: block;
    width: 100%;
    margin-top: 1rem;
  }

  .status {
    color: #9aa0ad;
    margin: 0.75rem 0 0;
  }
  .pulse {
    color: #ff6b6b;
    animation: pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse {
    50% { opacity: 0.4; }
  }
  .error {
    color: #ff8a80;
  }

  .transcript {
    line-height: 1.6;
    font-size: 1.05rem;
    white-space: pre-wrap;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }
  th, td {
    text-align: left;
    padding: 0.4rem 0.5rem;
    border-bottom: 1px solid #262b36;
  }
  .time {
    font-variant-numeric: tabular-nums;
    color: #9aa0ad;
    white-space: nowrap;
  }

  .words {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.5rem;
  }
  .word {
    background: #222733;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    cursor: default;
  }
  details summary {
    cursor: pointer;
    color: #9aa0ad;
    margin-top: 1rem;
  }

  .stream-section h2 {
    margin: 0 0 1rem;
    font-size: 1.1rem;
  }

  .live-transcript {
    margin-top: 1rem;
    min-height: 6rem;
    padding: 0.75rem 1rem;
    background: #12151b;
    border: 1px solid #262b36;
    border-radius: 8px;
    line-height: 1.7;
    font-size: 1.05rem;
    white-space: pre-wrap;
    transition: border-color 0.2s;
  }
  .live-transcript--active {
    border-color: #b3322c;
  }
  .live-transcript strong {
    font-weight: 700;
    color: #e7e9ee;
  }
  .live-transcript em {
    font-style: italic;
    color: #9aa0ad;
  }
  .placeholder {
    color: #4a5060;
    font-style: italic;
  }
</style>
