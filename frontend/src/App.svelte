<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { Recorder } from './lib/recorder'
  import { blobToWav16kMono } from './lib/wav'
  import {
    transcribe,
    fetchCapabilities,
    VadTranscriber,
    STREAM_PARAM_DEFAULTS,
    type Mode,
    type TranscriptionResult,
    type LanguageInfo,
    type StreamEvent,
    type LiveState,
    type StreamParams,
  } from './lib/api'
  import { LANGUAGES, DEFAULT_LANGUAGE } from './lib/languages'

  type Tab = 'live' | Mode

  const TABS: { value: Tab; label: string; hint: string }[] = [
    { value: 'live',       label: 'Live',        hint: 'Real-time microphone' },
    { value: 'plain',      label: 'Plain text',  hint: 'Just the transcript' },
    { value: 'timestamps', label: 'Timestamped', hint: 'Word & segment timings' },
    { value: 'longform',   label: 'Long form',   hint: 'For very long recordings' },
  ]

  let activeTab = $state<Tab>('live')
  let recording = $state(false)
  let busy = $state(false)
  let error = $state<string | null>(null)
  let result = $state<TranscriptionResult | null>(null)
  let audioUrl = $state<string | null>(null)

  let supportsLanguages = $state(false)
  let languages = $state<LanguageInfo[]>(LANGUAGES)
  let sourceLang = $state<string>(DEFAULT_LANGUAGE)
  let targetLang = $state<string>(DEFAULT_LANGUAGE)
  let isTranslation = $derived(supportsLanguages && sourceLang !== targetLang)

  let recorder: Recorder | null = null
  let wavBlob: Blob | null = null

  // ── VAD / live transcription state ──────────────────────────────────────────
  let streaming = $state(false)
  let vadLoading = $state(false)
  let vadError = $state<string | null>(null)
  let committedText = $state('')
  let partialText = $state('')
  let speechProb = $state(0)
  let speaking = $state(false)
  let liveState = $state<LiveState>('listening')
  let commitDelay = $state(500)
  let streamParams = $state<StreamParams>({ ...STREAM_PARAM_DEFAULTS })
  let vadTranscriber: VadTranscriber | null = null

  function handleStreamEvent(event: StreamEvent) {
    if (event.type === 'partial') {
      partialText = event.text
    } else if (event.type === 'committed') {
      committedText = committedText ? committedText + ' ' + event.text : event.text
      partialText = ''
    } else if (event.type === 'final') {
      if (event.text) committedText = committedText ? committedText + ' ' + event.text : event.text
      partialText = ''
    }
  }

  async function startVad() {
    vadError = null
    committedText = ''
    partialText = ''
    speechProb = 0
    speaking = false
    liveState = 'listening'
    vadLoading = true

    const t = new VadTranscriber(
      handleStreamEvent,
      (prob, spk) => { speechProb = prob; speaking = spk },
      (state) => { liveState = state },
      (msg) => { vadError = msg },
    )
    vadTranscriber = t

    try {
      await t.start(supportsLanguages ? { sourceLang, targetLang } : {}, commitDelay, streamParams)
      streaming = true
    } catch (e) {
      console.log(e)
      vadError = (e as Error).message
      vadTranscriber = null
    } finally {
      vadLoading = false
    }
  }

  async function stopVad() {
    streaming = false
    speaking = false
    speechProb = 0
    partialText = ''
    liveState = 'listening'
    await vadTranscriber?.stop()
    vadTranscriber = null
  }

  // Stop VAD when leaving the live tab.
  $effect(() => {
    if (activeTab !== 'live' && streaming) {
      void stopVad()
    }
  })

  onDestroy(async () => {
    await vadTranscriber?.destroy()
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
    result = null
  }

  async function runTranscription() {
    if (!wavBlob) return
    reset()
    busy = true
    try {
      const options = supportsLanguages ? { sourceLang, targetLang } : {}
      result = await transcribe(wavBlob, activeTab as Mode, options)
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

  // Hue: 120 (green) at prob=0 → 0 (red) at prob=1
  function probHue(p: number) { return Math.round((1 - p) * 120) }
</script>

<main>
  <header>
    <h1>🦜 Parakeet/Canary ASR</h1>
    <p>Record or upload audio and transcribe it with NVIDIA Parakeet/Canary.</p>
  </header>

  <section class="card">
    <fieldset class="modes">
      <legend>Mode</legend>
      {#each TABS as tab (tab.value)}
        <label class="mode" class:selected={activeTab === tab.value}>
          <input type="radio" name="mode" value={tab.value} bind:group={activeTab} />
          <span class="mode-label">{tab.label}</span>
          <span class="mode-hint">{tab.hint}</span>
        </label>
      {/each}
    </fieldset>

    {#if supportsLanguages}
      <div class="langs">
        <label>
          <span>Source language</span>
          <select bind:value={sourceLang} disabled={streaming}>
            {#each languages as lang (lang.code)}
              <option value={lang.code}>{lang.name}</option>
            {/each}
          </select>
        </label>
        <span class="arrow" class:translate={isTranslation}>→</span>
        <label>
          <span>Target language</span>
          <select bind:value={targetLang} disabled={streaming}>
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

    {#if activeTab === 'live'}
      <div class="controls">
        {#if streaming}
          <button class="record stop" onclick={stopVad}>⏹ Stop</button>
        {:else}
          <button class="record" onclick={startVad} disabled={vadLoading}>
            {vadLoading ? 'Loading…' : '🎙 Start'}
          </button>
        {/if}
      </div>

      <div class="commit-row">
        <span class="commit-label">Commit delay</span>
        <input class="commit-slider" type="range" min="200" max="1000" step="50"
               bind:value={commitDelay} disabled={streaming} />
        <span class="commit-value">{commitDelay} ms</span>
      </div>

      <details class="advanced" open={false}>
        <summary class="advanced-toggle">Advanced</summary>
        <div class="advanced-grid">
          <label>Min. duration
            <div class="param-row">
              <input type="range" min="0.3" max="3" step="0.1"
                bind:value={streamParams.minDuration} disabled={streaming} />
              <span>{streamParams.minDuration.toFixed(1)} s</span>
            </div>
          </label>
          <label>Retranscribe every
            <div class="param-row">
              <input type="range" min="0.1" max="2" step="0.1"
                bind:value={streamParams.retranscribeInterval} disabled={streaming} />
              <span>{streamParams.retranscribeInterval.toFixed(1)} s</span>
            </div>
          </label>
          <label>Stable words
            <div class="param-row">
              <input type="range" min="1" max="10" step="1"
                bind:value={streamParams.stableWords} disabled={streaming} />
              <span>{streamParams.stableWords}</span>
            </div>
          </label>
          <label>Stable iterations
            <div class="param-row">
              <input type="range" min="1" max="5" step="1"
                bind:value={streamParams.stableIters} disabled={streaming} />
              <span>{streamParams.stableIters}</span>
            </div>
          </label>
          <label>Max. buffer
            <div class="param-row">
              <input type="range" min="5" max="60" step="5"
                bind:value={streamParams.maxDuration} disabled={streaming} />
              <span>{streamParams.maxDuration.toFixed(0)} s</span>
            </div>
          </label>
          <label>Context window
            <div class="param-row">
              <input type="range" min="0" max="10" step="0.5"
                bind:value={streamParams.contextDuration} disabled={streaming} />
              <span>{streamParams.contextDuration.toFixed(1)} s</span>
            </div>
          </label>
          <button class="reset-btn"
            onclick={() => { streamParams = { ...STREAM_PARAM_DEFAULTS } }}
            disabled={streaming}>
            Reset to defaults
          </button>
        </div>
      </details>

      {#if streaming || vadLoading}
        <div class="vad-row">
          <div class="vad-dot" class:vad-dot--speaking={speaking}
               style="background: hsl({probHue(speechProb)}deg,80%,48%); box-shadow: 0 0 {speechProb * 14}px hsl({probHue(speechProb)}deg,80%,48%)">
          </div>
          <div class="vad-track">
            <div class="vad-level"
                 style="width: {speechProb * 100}%; background: hsl({probHue(speechProb)}deg,80%,48%)">
            </div>
          </div>
          <span class="vad-label" class:pulse={speaking || liveState === 'committing'}>
            {#if vadLoading}
              Loading model…
            {:else if liveState === 'speaking'}
              Speaking…
            {:else if liveState === 'committing'}
              Committing… ({commitDelay} ms)
            {:else}
              Listening…
            {/if}
          </span>
        </div>
      {/if}

      {#if vadError}
        <p class="error">⚠ {vadError}</p>
      {/if}
    {:else}
      <div class="controls">
        {#if recording}
          <button class="record stop" onclick={stopRecording}>⏹ Stop recording</button>
        {:else}
          <button class="record" onclick={startRecording} disabled={busy}>🎙 Record</button>
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
    {/if}

    <div class="output" class:output--active={streaming && speaking}>
      {#if activeTab === 'live'}
        {#if committedText || partialText}
          {committedText}{#if partialText}{committedText ? ' ' : ''}<em class="partial">{partialText}</em>{/if}
        {:else}
          <span class="placeholder">Transcript will appear here as you speak…</span>
        {/if}
      {:else if result}
        <p class="transcript">{result.text || '(empty)'}</p>
      {:else}
        <span class="placeholder">Record or upload audio, then click Transcribe…</span>
      {/if}
    </div>

    {#if result?.segments?.length}
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

    {#if result?.words?.length}
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
    grid-template-columns: repeat(4, 1fr);
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
  button:disabled,
  select:disabled {
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

  /* ── Advanced collapsible ───────────────────────────────────────────────── */
  .advanced {
    margin-top: 0.75rem;
    border: 1px solid #262b36;
    border-radius: 8px;
    overflow: hidden;
  }
  .advanced-toggle {
    display: block;
    padding: 0.5rem 0.75rem;
    font-size: 0.82rem;
    color: #9aa0ad;
    cursor: pointer;
    user-select: none;
    list-style: none;
  }
  .advanced-toggle::marker,
  .advanced-toggle::-webkit-details-marker { display: none; }
  .advanced-toggle::before {
    content: '▶ ';
    font-size: 0.65rem;
    vertical-align: middle;
    transition: transform 0.15s;
    display: inline-block;
  }
  details[open] .advanced-toggle::before {
    transform: rotate(90deg);
  }
  .advanced-grid {
    padding: 0.5rem 0.75rem 0.75rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem 1.2rem;
  }
  .advanced-grid label {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    font-size: 0.78rem;
    color: #9aa0ad;
  }
  .param-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .param-row input[type='range'] {
    flex: 1;
    accent-color: #5b8cff;
    cursor: pointer;
  }
  .param-row input[type='range']:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .param-row span {
    min-width: 3.2rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: #e7e9ee;
    font-size: 0.78rem;
  }
  .reset-btn {
    grid-column: 1 / -1;
    margin-top: 0.25rem;
    font-size: 0.78rem;
    padding: 0.35rem 0.75rem;
    background: transparent;
    border-color: #2c323d;
    color: #9aa0ad;
  }
  .reset-btn:hover:not(:disabled) {
    color: #e7e9ee;
    border-color: #5b8cff;
  }

  /* ── Commit-delay slider ────────────────────────────────────────────────── */
  .commit-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.75rem;
  }
  .commit-label {
    font-size: 0.8rem;
    color: #9aa0ad;
    flex-shrink: 0;
  }
  .commit-slider {
    flex: 1;
    accent-color: #5b8cff;
    cursor: pointer;
  }
  .commit-slider:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .commit-value {
    font-size: 0.8rem;
    color: #9aa0ad;
    min-width: 3.5rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  /* ── VAD activity indicator ─────────────────────────────────────────────── */
  .vad-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.8rem;
  }
  .vad-dot {
    flex-shrink: 0;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #3a4050;
    transition: background 0.08s, box-shadow 0.08s;
  }
  .vad-dot--speaking {
    animation: vad-pulse 0.7s ease-in-out infinite;
  }
  @keyframes vad-pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.45); }
  }
  .vad-track {
    flex: 1;
    height: 5px;
    border-radius: 3px;
    background: #12151b;
    border: 1px solid #262b36;
    overflow: hidden;
  }
  .vad-level {
    height: 100%;
    width: 0%;
    border-radius: 3px;
    transition: width 60ms linear, background 60ms linear;
  }
  .vad-label {
    flex-shrink: 0;
    font-size: 0.78rem;
    color: #9aa0ad;
    min-width: 9rem;
    text-align: right;
  }

  /* ── Unified output area ────────────────────────────────────────────────── */
  .output {
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
  .output--active {
    border-color: #3c8c3c;
  }

  .transcript {
    line-height: 1.6;
    font-size: 1.05rem;
    white-space: pre-wrap;
    margin: 0;
  }

  .placeholder {
    color: #4a5060;
    font-style: italic;
  }
  .partial {
    color: #6a7282;
    font-style: italic;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin-top: 1rem;
  }
  h3 {
    margin: 1rem 0 0.25rem;
    font-size: 0.95rem;
    color: #9aa0ad;
    font-weight: 600;
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
</style>
