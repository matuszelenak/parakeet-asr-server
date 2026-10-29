<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import Icon from './lib/Icon.svelte'
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

  const TABS: { value: Tab; label: string; hint: string; icon: string }[] = [
    { value: 'live',       label: 'Live',        hint: 'Realtime mic',      icon: 'wave' },
    { value: 'plain',      label: 'Plain text',  hint: 'Just the words',    icon: 'file-text' },
    { value: 'timestamps', label: 'Timestamped', hint: 'Word & segment time', icon: 'clock' },
    { value: 'longform',   label: 'Long form',   hint: 'Very long audio',   icon: 'layers' },
  ]

  let activeTab = $state<Tab>('live')
  let recording = $state(false)
  let busy = $state(false)
  let error = $state<string | null>(null)
  let result = $state<TranscriptionResult | null>(null)
  let audioUrl = $state<string | null>(null)
  let audioMeta = $state<{ name: string; seconds: number } | null>(null)
  let dragging = $state(false)
  let copied = $state(false)
  let backendOffline = $state(false)

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
  let transcriptBody = $state<HTMLDivElement | null>(null)

  // ── Derived view state ─────────────────────────────────────────────────────
  let displayText = $derived(
    activeTab === 'live'
      ? [committedText, partialText].filter(Boolean).join(' ')
      : (result?.text ?? ''),
  )
  let hasText = $derived(displayText.trim().length > 0)
  let wordCount = $derived(hasText ? displayText.trim().split(/\s+/).length : 0)
  let liveStatus = $derived(
    vadLoading
      ? 'Starting…'
      : liveState === 'speaking'
        ? 'Speaking…'
        : liveState === 'committing'
          ? 'Committing…'
          : 'Listening…',
  )
  let commitPercent = $derived(Math.round(speechProb * 100))

  // Keep the newest words visible while streaming.
  $effect(() => {
    void committedText
    void partialText
    if (streaming && transcriptBody) {
      transcriptBody.scrollTop = transcriptBody.scrollHeight
    }
  })

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

  function toggleLive() {
    if (streaming) void stopVad()
    else void startVad()
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
      backendOffline = true
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
      error = `Microphone unavailable: ${(e as Error).message}`
      recorder = null
    }
  }

  async function stopRecording() {
    if (!recorder) return
    recording = false
    busy = true
    try {
      const raw = await recorder.stop()
      await prepare(raw, 'Microphone recording')
    } catch (e) {
      error = (e as Error).message
    } finally {
      recorder = null
      busy = false
    }
  }

  function onFile(event: Event) {
    const input = event.target as HTMLInputElement
    const file = input.files?.[0]
    if (file) void loadFile(file)
    input.value = ''
  }

  async function loadFile(file: File) {
    reset()
    busy = true
    try {
      await prepare(file, file.name)
    } catch (e) {
      error = (e as Error).message
    } finally {
      busy = false
    }
  }

  function onDragOver(event: DragEvent) {
    event.preventDefault()
    if (!busy && !recording) dragging = true
  }

  function onDragLeave() {
    dragging = false
  }

  function onDrop(event: DragEvent) {
    event.preventDefault()
    dragging = false
    if (busy || recording) return
    const file = event.dataTransfer?.files?.[0]
    if (!file) return
    if (!file.type.startsWith('audio/') && !/\.(wav|mp3|m4a|mp4|webm|ogg|flac|aac)$/i.test(file.name)) {
      error = `Unsupported file type: ${file.type || file.name}`
      return
    }
    void loadFile(file)
  }

  async function prepare(raw: Blob, name: string) {
    wavBlob = await blobToWav16kMono(raw)
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    audioUrl = URL.createObjectURL(wavBlob)
    audioMeta = { name, seconds: Math.max(0, (wavBlob.size - 44) / 32000) }
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

  async function copyTranscript() {
    if (!hasText) return
    try {
      await navigator.clipboard.writeText(displayText.trim())
      copied = true
      setTimeout(() => (copied = false), 1600)
    } catch {
      error = 'Could not copy to the clipboard.'
    }
  }

  function downloadTranscript() {
    if (!hasText) return
    const url = URL.createObjectURL(new Blob([displayText.trim()], { type: 'text/plain' }))
    const a = document.createElement('a')
    a.href = url
    a.download = `transcript-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  function clearAll() {
    result = null
    committedText = ''
    partialText = ''
    error = null
    vadError = null
  }

  function swapLanguages() {
    const tmp = sourceLang
    sourceLang = targetLang
    targetLang = tmp
  }

  function fmt(t: number): string {
    const m = Math.floor(t / 60)
    const s = (t % 60).toFixed(2).padStart(5, '0')
    return `${m}:${s}`
  }

  function fmtDuration(seconds: number): string {
    const total = Math.round(seconds)
    return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
  }

  // Hue: 120 (green) at prob=0 → 0 (red) at prob=1
  function probHue(p: number) { return Math.round((1 - p) * 120) }
</script>

<div class="shell">
  <h1 class="sr-only">Parakeet / Canary ASR</h1>

  <main>
    {#if backendOffline}
      <div class="alert" role="status">
        <Icon name="alert" size={16} />
        <span>Server unreachable — check that the backend is running.</span>
      </div>
    {/if}

    <section class="panel">
      <div class="modes" role="group" aria-label="Transcription mode">
        {#each TABS as tab (tab.value)}
          <button
            type="button"
            aria-pressed={activeTab === tab.value}
            class="mode"
            class:active={activeTab === tab.value}
            onclick={() => (activeTab = tab.value)}
          >
            <span class="mode-icon"><Icon name={tab.icon} size={16} /></span>
            <span class="mode-copy">
              <span class="mode-label">{tab.label}</span>
              <span class="mode-hint">{tab.hint}</span>
            </span>
          </button>
        {/each}
      </div>

      {#if supportsLanguages}
        <div class="langs">
          <label class="field">
            <span class="field-label">Source language</span>
            <span class="select-wrap">
              <select bind:value={sourceLang} disabled={streaming}>
                {#each languages as lang (lang.code)}
                  <option value={lang.code}>{lang.name}</option>
                {/each}
              </select>
              <Icon name="chevron" size={14} />
            </span>
          </label>

          <button
            type="button"
            class="swap"
            onclick={swapLanguages}
            disabled={streaming}
            aria-label="Swap source and target languages"
            title="Swap languages"
          >
            <Icon name="swap" size={16} />
          </button>

          <label class="field">
            <span class="field-label">Target language</span>
            <span class="select-wrap">
              <select bind:value={targetLang} disabled={streaming}>
                {#each languages as lang (lang.code)}
                  <option value={lang.code}>{lang.name}</option>
                {/each}
              </select>
              <Icon name="chevron" size={14} />
            </span>
          </label>

          <span class="translate-badge" class:on={isTranslation}>
            <Icon name="sparkle" size={13} />
            {isTranslation ? 'Translating' : 'Transcribing'}
          </span>
        </div>
      {/if}

      {#if activeTab === 'live'}
        <div class="live">
          <button
            type="button"
            class="mic"
            class:live={streaming}
            onclick={toggleLive}
            disabled={vadLoading}
            aria-label={streaming ? 'Stop live transcription' : 'Start live transcription'}
          >
            <span class="mic-ring" aria-hidden="true"></span>
            <span class="mic-core"><Icon name={streaming ? 'stop' : 'mic'} size={26} /></span>
          </button>

          <div class="live-info">
            <div class="live-head">
              <span
                class="live-dot"
                class:speaking
                class:committing={streaming && liveState === 'committing'}
                style={streaming
                  ? `background: hsl(${probHue(speechProb)}deg 80% 55%)`
                  : ''}
              ></span>
              <span class="live-status">{streaming || vadLoading ? liveStatus : 'Ready to listen'}</span>
              {#if streaming && liveState === 'committing'}
                <span class="live-sub">commit delay {commitDelay} ms</span>
              {/if}
            </div>

            <p class="live-help">
              {#if streaming}
                Speak naturally — partial words appear instantly, finished phrases are committed.
              {:else if vadLoading}
                Loading the Silero VAD model…
              {:else}
                Press the microphone to transcribe your voice in realtime. Audio never leaves this machine.
              {/if}
            </p>

            <div class="meter" class:idle={!streaming}>
              <div
                class="meter-fill"
                style="width: {streaming ? speechProb * 100 : 0}%; background: hsl({probHue(speechProb)}deg 82% 58%)"
              ></div>
            </div>
            <div class="meter-legend">
              <span>Voice activity</span>
              <span class="mono">{streaming ? `${commitPercent}%` : '—'}</span>
            </div>
          </div>
        </div>

        <div class="tuning">
          <div class="slider-row">
            <div class="slider-label">
              <span>Commit delay</span>
              <small>Pause before a phrase is finalised</small>
            </div>
            <input
              class="range"
              type="range"
              min="200"
              max="1000"
              step="50"
              bind:value={commitDelay}
              disabled={streaming}
              aria-label="Commit delay in milliseconds"
            />
            <output class="value">{commitDelay} ms</output>
          </div>
        </div>

        <details class="advanced">
          <summary>
            <span class="advanced-title"><Icon name="sliders" size={15} /> Advanced tuning</span>
            <span class="chev"><Icon name="chevron" size={15} /></span>
          </summary>

          <div class="advanced-grid">
            <label class="param">
              <span class="param-label">Min. duration <span class="mono">{streamParams.minDuration.toFixed(1)} s</span></span>
              <input class="range" type="range" min="0.3" max="3" step="0.1"
                bind:value={streamParams.minDuration} disabled={streaming} />
            </label>
            <label class="param">
              <span class="param-label">Retranscribe every <span class="mono">{streamParams.retranscribeInterval.toFixed(1)} s</span></span>
              <input class="range" type="range" min="0.1" max="2" step="0.1"
                bind:value={streamParams.retranscribeInterval} disabled={streaming} />
            </label>
            <label class="param">
              <span class="param-label">Stable words <span class="mono">{streamParams.stableWords}</span></span>
              <input class="range" type="range" min="1" max="10" step="1"
                bind:value={streamParams.stableWords} disabled={streaming} />
            </label>
            <label class="param">
              <span class="param-label">Stable iterations <span class="mono">{streamParams.stableIters}</span></span>
              <input class="range" type="range" min="1" max="5" step="1"
                bind:value={streamParams.stableIters} disabled={streaming} />
            </label>
            <label class="param">
              <span class="param-label">Max. buffer <span class="mono">{streamParams.maxDuration.toFixed(0)} s</span></span>
              <input class="range" type="range" min="5" max="60" step="5"
                bind:value={streamParams.maxDuration} disabled={streaming} />
            </label>
            <label class="param">
              <span class="param-label">Context window <span class="mono">{streamParams.contextDuration.toFixed(1)} s</span></span>
              <input class="range" type="range" min="0" max="10" step="0.5"
                bind:value={streamParams.contextDuration} disabled={streaming} />
            </label>
            <label class="param">
              <span class="param-label">Trim buffer after <span class="mono">{streamParams.trimDuration.toFixed(0)} s</span></span>
              <input class="range" type="range" min="4" max="30" step="1"
                bind:value={streamParams.trimDuration} disabled={streaming} />
            </label>
            <label class="param">
              <span class="param-label">Hold back <span class="mono">{streamParams.holdback.toFixed(1)} s</span></span>
              <input class="range" type="range" min="0" max="1.5" step="0.1"
                bind:value={streamParams.holdback} disabled={streaming} />
            </label>
            <button
              type="button"
              class="btn btn-quiet advanced-reset"
              onclick={() => { streamParams = { ...STREAM_PARAM_DEFAULTS } }}
              disabled={streaming}
            >
              <Icon name="reset" size={14} /> Reset to defaults
            </button>
          </div>
        </details>

        {#if vadError}
          <div class="alert" role="alert">
            <Icon name="alert" size={16} />
            <span>{vadError}</span>
          </div>
        {/if}
      {:else}
        <div class="capture">
          <button
            type="button"
            class="btn btn-record"
            class:recording
            onclick={() => (recording ? stopRecording() : startRecording())}
            disabled={busy}
          >
            <Icon name={recording ? 'stop' : 'mic'} size={17} />
            {recording ? 'Stop recording' : 'Record'}
          </button>

          <label
            class="drop"
            class:dragging
            class:disabled={busy || recording}
            ondragover={onDragOver}
            ondragleave={onDragLeave}
            ondrop={onDrop}
          >
            <input type="file" accept="audio/*" onchange={onFile} disabled={busy || recording} />
            <span class="drop-icon"><Icon name="upload" size={18} /></span>
            <span class="drop-copy">
              <strong>Drop an audio file</strong>
              <span>or click to browse</span>
            </span>
            <span class="drop-hint">WAV · MP3 · M4A · WEBM</span>
          </label>
        </div>

        {#if recording}
          <p class="recording-note"><span class="rec-dot"></span> Recording — speak now</p>
        {/if}

        {#if audioUrl}
          <div class="player">
            <div class="player-head">
              <span class="player-file">
                <Icon name="file-audio" size={15} />
                <span class="player-name">{audioMeta?.name ?? 'audio'}</span>
              </span>
              {#if audioMeta}
                <span class="player-duration mono">{fmtDuration(audioMeta.seconds)}</span>
              {/if}
            </div>
            <!-- svelte-ignore a11y_media_has_caption -->
            <audio controls src={audioUrl} aria-label="Audio preview"></audio>
            <button
              type="button"
              class="btn btn-primary"
              onclick={runTranscription}
              disabled={busy || recording}
            >
              {busy ? 'Transcribing…' : 'Transcribe'}
              {#if !busy}<Icon name="sparkle" size={15} />{/if}
            </button>
          </div>
        {/if}

        {#if error}
          <div class="alert" role="alert">
            <Icon name="alert" size={16} />
            <span>{error}</span>
            <button type="button" class="alert-dismiss" aria-label="Dismiss error" onclick={() => (error = null)}>
              <Icon name="x" size={13} />
            </button>
          </div>
        {/if}
      {/if}
    </section>

    <section class="panel transcript-panel" class:live={streaming}>
      <div class="transcript-head">
        <div class="transcript-title">
          <span class="transcript-icon"><Icon name={activeTab === 'live' ? 'wave' : 'file-text'} size={16} /></span>
          <h2>{activeTab === 'live' ? 'Live transcript' : 'Transcript'}</h2>
          {#if streaming && activeTab === 'live'}
            <span class="badge badge-live">live</span>
          {/if}
        </div>

        <div class="transcript-actions">
          {#if wordCount > 0}
            <span class="word-count">{wordCount} {wordCount === 1 ? 'word' : 'words'}</span>
          {/if}
          <button
            type="button"
            class="icon-btn"
            onclick={copyTranscript}
            disabled={!hasText}
            title={copied ? 'Copied' : 'Copy transcript'}
            aria-label="Copy transcript"
          >
            <Icon name={copied ? 'check' : 'copy'} size={15} />
          </button>
          <button
            type="button"
            class="icon-btn"
            onclick={downloadTranscript}
            disabled={!hasText}
            title="Download as .txt"
            aria-label="Download transcript as text file"
          >
            <Icon name="download" size={15} />
          </button>
          <button
            type="button"
            class="icon-btn"
            onclick={clearAll}
            disabled={!hasText && !result}
            title="Clear transcript"
            aria-label="Clear transcript"
          >
            <Icon name="trash" size={15} />
          </button>
        </div>
      </div>

      <div class="transcript-body" class:streaming bind:this={transcriptBody}>
        {#if activeTab === 'live'}
          {#if committedText || partialText}
            <p class="transcript-text">{committedText}{#if partialText}{committedText ? ' ' : ''}<em class="partial">{partialText}</em>{/if}</p>
          {:else if vadLoading}
            <div class="empty">
              <span class="spinner"></span>
              <p>Preparing the speech model…</p>
            </div>
          {:else}
            <div class="empty">
              <span class="empty-icon"><Icon name="wave" size={20} /></span>
              <p>Your words will appear here as you speak.</p>
              <small>The transcript updates in realtime, phrase by phrase.</small>
            </div>
          {/if}
        {:else if result}
          <p class="transcript-text">{result.text?.trim() ? result.text : 'No speech detected in this audio.'}</p>
        {:else if busy}
          <div class="empty">
            <span class="spinner"></span>
            <p>Transcribing your audio…</p>
            <small>This can take a moment for long recordings.</small>
          </div>
        {:else}
          <div class="empty">
            <span class="empty-icon"><Icon name="file-text" size={20} /></span>
            <p>Nothing transcribed yet.</p>
            <small>Record or drop an audio file above, then hit Transcribe.</small>
          </div>
        {/if}
      </div>

      {#if error && activeTab === 'live'}
        <div class="alert" role="alert">
          <Icon name="alert" size={16} />
          <span>{error}</span>
        </div>
      {/if}

      {#if result?.segments?.length}
        <div class="subsection">
          <h3>Segments <span class="badge">{result.segments.length}</span></h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Start</th><th>End</th><th>Text</th></tr>
              </thead>
              <tbody>
                {#each result.segments as seg, i (seg.start + '-' + i)}
                  <tr>
                    <td class="time">{fmt(seg.start)}</td>
                    <td class="time">{fmt(seg.end)}</td>
                    <td>{seg.segment}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      {/if}

      {#if result?.words?.length}
        <details class="words-details">
          <summary>
            <span>Word timestamps <span class="badge">{result.words.length}</span></span>
            <span class="chev"><Icon name="chevron" size={15} /></span>
          </summary>
          <div class="words">
            {#each result.words as w, i (w.start + '-' + i)}
              <span class="word" title={`${fmt(w.start)} – ${fmt(w.end)}`}>{w.word}</span>
            {/each}
          </div>
        </details>
      {/if}
    </section>
  </main>
</div>

<style>
  .shell {
    width: 100%;
    max-width: 880px;
    margin: 0 auto;
    padding: 1.75rem 1.25rem 3rem;
  }

  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
    border: 0;
  }

  main {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 1.15rem;
  }

  /* ── Mode switcher ──────────────────────────────────────────────────────── */
  .modes {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.35rem;
    padding: 0.3rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--surface-2);
  }

  .mode {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.55rem 0.6rem;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-dim);
    text-align: left;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
  }
  .mode:hover:not(.active) {
    background: rgba(255, 255, 255, 0.04);
    color: var(--text);
  }
  .mode.active {
    background: var(--surface-3);
    border-color: var(--line-strong);
    color: var(--text);
  }

  .mode-icon {
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    flex-shrink: 0;
    color: var(--text-faint);
  }
  .mode.active .mode-icon {
    color: var(--accent);
  }

  .mode-copy {
    display: flex;
    flex-direction: column;
    min-width: 0;
    line-height: 1.2;
  }
  .mode-label {
    font-size: 0.84rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mode-hint {
    font-size: 0.68rem;
    color: var(--text-faint);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ── Language row ───────────────────────────────────────────────────────── */
  .langs {
    display: flex;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 0.7rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    flex: 1 1 12rem;
    min-width: 9rem;
  }
  .field-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text-faint);
  }

  .select-wrap {
    position: relative;
    display: flex;
    align-items: center;
  }
  .select-wrap select {
    appearance: none;
    width: 100%;
    padding: 0.5rem 2rem 0.5rem 0.7rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    color: var(--text);
    font-size: 0.86rem;
    transition: border-color 0.15s ease;
  }
  .select-wrap select:hover:not(:disabled) {
    border-color: var(--line-strong);
  }
  .select-wrap select:disabled {
    opacity: 0.55;
  }
  .select-wrap :global(svg) {
    position: absolute;
    right: 0.65rem;
    color: var(--text-faint);
    pointer-events: none;
  }

  .swap {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    flex-shrink: 0;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    color: var(--text-dim);
    transition: color 0.15s ease, border-color 0.15s ease;
  }
  .swap:hover:not(:disabled) {
    color: var(--accent-strong);
    border-color: var(--line-strong);
  }
  .swap:disabled {
    opacity: 0.45;
  }

  .translate-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    margin-bottom: 0.45rem;
    padding: 0.3rem 0.55rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    color: var(--text-faint);
    white-space: nowrap;
  }
  .translate-badge.on {
    color: var(--accent-strong);
    border-color: rgba(139, 155, 255, 0.3);
  }

  /* ── Live mode ──────────────────────────────────────────────────────────── */
  .live {
    display: flex;
    align-items: center;
    gap: 1.15rem;
    margin-top: 1rem;
    padding: 0.35rem 0.15rem;
  }

  .mic {
    position: relative;
    display: grid;
    place-items: center;
    width: 68px;
    height: 68px;
    flex-shrink: 0;
    padding: 0;
    border: none;
    border-radius: 50%;
    background: transparent;
    color: var(--text);
  }
  .mic-core {
    position: relative;
    z-index: 1;
    display: grid;
    place-items: center;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    border: 1px solid var(--line-strong);
    background: var(--surface-3);
    transition: transform 0.15s ease, background 0.2s ease, border-color 0.2s ease,
      color 0.2s ease;
  }
  .mic:hover:not(:disabled) .mic-core {
    border-color: rgba(139, 155, 255, 0.5);
    color: var(--accent-strong);
  }
  .mic:active:not(:disabled) .mic-core {
    transform: scale(0.97);
  }
  .mic:disabled .mic-core {
    opacity: 0.6;
  }

  .mic.live .mic-core {
    color: #fff;
    border-color: rgba(255, 118, 118, 0.6);
    background: #b93b3a;
  }

  .mic-ring {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 1px solid rgba(255, 118, 118, 0.45);
    opacity: 0;
    pointer-events: none;
  }
  .mic.live .mic-ring {
    animation: ripple 2.2s cubic-bezier(0.25, 0.5, 0.35, 1) infinite;
  }
  @keyframes ripple {
    0% { transform: scale(0.9); opacity: 0.7; }
    100% { transform: scale(1.5); opacity: 0; }
  }

  .live-info {
    flex: 1;
    min-width: 0;
  }
  .live-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-height: 1.4rem;
  }
  .live-dot {
    width: 8px;
    height: 8px;
    flex-shrink: 0;
    border-radius: 50%;
    background: var(--surface-3);
    transition: background 0.1s linear;
  }
  .live-dot.speaking {
    animation: breathe 1.1s ease-in-out infinite;
  }
  .live-dot.committing {
    animation: breathe 0.6s ease-in-out infinite;
  }
  @keyframes breathe {
    50% { transform: scale(1.5); }
  }

  .live-status {
    font-size: 0.95rem;
    font-weight: 600;
  }
  .live-sub {
    font-size: 0.74rem;
    color: var(--text-faint);
    font-variant-numeric: tabular-nums;
  }
  .live-help {
    margin: 0.15rem 0 0.75rem;
    font-size: 0.82rem;
    color: var(--text-dim);
    max-width: 46ch;
  }

  .meter {
    height: 6px;
    border-radius: 3px;
    background: var(--surface-2);
    border: 1px solid var(--line);
    overflow: hidden;
  }
  .meter-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 70ms linear, background 70ms linear;
  }
  .meter.idle .meter-fill {
    width: 0 !important;
  }
  .meter-legend {
    display: flex;
    justify-content: space-between;
    margin-top: 0.35rem;
    font-size: 0.7rem;
    color: var(--text-faint);
    letter-spacing: 0.02em;
  }

  /* ── Sliders ────────────────────────────────────────────────────────────── */
  .tuning {
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--line);
  }

  .slider-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
  }
  .slider-label {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    font-size: 0.82rem;
    line-height: 1.3;
  }
  .slider-label small {
    font-size: 0.7rem;
    color: var(--text-faint);
  }
  .value {
    flex-shrink: 0;
    min-width: 4.4rem;
    padding: 0.2rem 0.5rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface-2);
    font-family: var(--font-mono);
    font-size: 0.74rem;
    color: var(--text-dim);
    text-align: center;
    font-variant-numeric: tabular-nums;
  }

  .range {
    flex: 1;
    min-width: 4rem;
    height: 18px;
    appearance: none;
    background: transparent;
    cursor: pointer;
  }
  .range:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .range::-webkit-slider-runnable-track {
    height: 4px;
    border-radius: 2px;
    background: var(--surface-3);
  }
  .range::-moz-range-track {
    height: 4px;
    border-radius: 2px;
    background: var(--surface-3);
  }
  .range::-moz-range-progress {
    height: 4px;
    border-radius: 2px;
    background: rgba(139, 155, 255, 0.6);
  }
  .range::-webkit-slider-thumb {
    appearance: none;
    width: 14px;
    height: 14px;
    margin-top: -5px;
    border-radius: 50%;
    background: var(--accent);
    border: none;
  }
  .range::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--accent);
    border: none;
  }
  .range:hover:not(:disabled)::-webkit-slider-thumb {
    background: var(--accent-strong);
  }

  /* ── Advanced drawer ────────────────────────────────────────────────────── */
  .advanced {
    margin-top: 0.6rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--surface-2);
    overflow: hidden;
  }
  .advanced summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.8rem;
    font-size: 0.8rem;
    color: var(--text-dim);
    cursor: pointer;
    list-style: none;
    user-select: none;
  }
  .advanced summary::-webkit-details-marker {
    display: none;
  }
  .advanced summary:hover {
    color: var(--text);
  }

  .advanced-title {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-weight: 500;
  }

  .advanced-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem 1.4rem;
    padding: 0.85rem;
    border-top: 1px solid var(--line);
  }
  .param {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .param-label {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: var(--text-dim);
  }
  .param-label .mono {
    color: var(--text-faint);
  }
  .advanced-reset {
    grid-column: 1 / -1;
    justify-self: start;
    margin-top: 0.15rem;
  }

  /* ── Buttons ────────────────────────────────────────────────────────────── */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    padding: 0.55rem 0.95rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    color: var(--text);
    font-size: 0.86rem;
    font-weight: 500;
    white-space: nowrap;
    transition: background 0.15s ease, border-color 0.15s ease;
  }
  .btn:hover:not(:disabled) {
    background: var(--surface-3);
    border-color: var(--line-strong);
  }
  .btn:disabled {
    opacity: 0.5;
  }

  .btn-quiet {
    background: transparent;
    color: var(--text-dim);
    font-size: 0.76rem;
    padding: 0.4rem 0.7rem;
  }
  .btn-quiet:hover:not(:disabled) {
    color: var(--text);
    border-color: var(--line-strong);
    background: var(--surface-3);
  }

  .btn-record {
    background: var(--surface-2);
    border-color: var(--line-strong);
  }
  .btn-record.recording {
    color: #fff;
    border-color: rgba(255, 118, 118, 0.6);
    background: #b93b3a;
  }

  .btn-primary {
    width: 100%;
    margin-top: 0.65rem;
    padding: 0.68rem 1rem;
    border: 1px solid var(--accent);
    color: #0b0d13;
    font-weight: 600;
    background: var(--accent);
  }
  .btn-primary:hover:not(:disabled) {
    background: var(--accent-strong);
    border-color: var(--accent-strong);
  }

  /* ── Capture (record / drop) ────────────────────────────────────────────── */
  .capture {
    display: grid;
    grid-template-columns: auto 1fr;
    align-items: stretch;
    gap: 0.6rem;
    margin-top: 1rem;
  }

  .drop {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.65rem 0.85rem;
    border: 1px dashed var(--line-strong);
    border-radius: var(--radius-md);
    background: var(--surface-2);
    color: var(--text-dim);
    cursor: pointer;
    transition: border-color 0.15s ease, background 0.15s ease, color 0.15s ease;
  }
  .drop:hover,
  .drop.dragging {
    border-color: rgba(139, 155, 255, 0.6);
    color: var(--text);
  }
  .drop.dragging {
    border-style: solid;
  }
  .drop.disabled {
    opacity: 0.5;
    pointer-events: none;
  }
  .drop input {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
  }
  .drop-icon {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    flex-shrink: 0;
    border-radius: 4px;
    background: var(--surface-3);
    color: var(--accent);
  }
  .drop-copy {
    display: flex;
    flex-direction: column;
    min-width: 0;
    line-height: 1.25;
    font-size: 0.8rem;
  }
  .drop-copy strong {
    font-weight: 550;
    color: var(--text);
  }
  .drop-copy span {
    color: var(--text-faint);
    font-size: 0.74rem;
  }
  .drop-hint {
    margin-left: auto;
    flex-shrink: 0;
    font-size: 0.66rem;
    letter-spacing: 0.06em;
    color: var(--text-faint);
  }

  .recording-note {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.75rem;
    font-size: 0.8rem;
    color: #ffb1b1;
  }
  .rec-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--danger);
    animation: blink 1.2s ease-in-out infinite;
  }
  @keyframes blink {
    50% { opacity: 0.25; }
  }

  /* ── Player ─────────────────────────────────────────────────────────────── */
  .player {
    margin-top: 0.85rem;
    padding: 0.75rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-md);
    background: var(--surface-2);
  }
  .player-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.76rem;
  }
  .player-file {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    min-width: 0;
    color: var(--text-dim);
  }
  .player-name {
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
  .player-duration {
    flex-shrink: 0;
    color: var(--text-faint);
    font-size: 0.72rem;
  }
  .player audio {
    display: block;
    width: 100%;
    height: 34px;
    border-radius: 4px;
  }

  /* ── Transcript panel ───────────────────────────────────────────────────── */
  .transcript-panel {
    padding: 0;
    overflow: hidden;
  }
  .transcript-panel.live {
    border-color: rgba(139, 155, 255, 0.25);
  }

  .transcript-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding: 0.8rem 1rem;
    border-bottom: 1px solid var(--line);
  }
  .transcript-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-width: 0;
  }
  .transcript-icon {
    display: grid;
    place-items: center;
    color: var(--text-faint);
  }
  .transcript-title h2 {
    font-size: 0.86rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
  }

  .transcript-actions {
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }
  .word-count {
    margin-right: 0.35rem;
    font-size: 0.72rem;
    color: var(--text-faint);
    font-variant-numeric: tabular-nums;
  }
  .icon-btn {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    border: 1px solid transparent;
    border-radius: 4px;
    background: transparent;
    color: var(--text-faint);
    transition: color 0.15s ease, background 0.15s ease, border-color 0.15s ease;
  }
  .icon-btn:hover:not(:disabled) {
    color: var(--text);
    background: var(--surface-3);
    border-color: var(--line);
  }
  .icon-btn:disabled {
    opacity: 0.35;
  }

  .transcript-body {
    padding: 1.25rem 1rem 1.4rem;
    min-height: 9.5rem;
    max-height: 45vh;
    overflow-y: auto;
    font-size: 1.06rem;
    line-height: 1.75;
  }

  .transcript-text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    letter-spacing: -0.003em;
  }
  .partial {
    font-style: normal;
    color: var(--text-faint);
  }

  .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    min-height: 7.5rem;
    text-align: center;
    color: var(--text-faint);
  }
  .empty p {
    font-size: 0.9rem;
    color: var(--text-dim);
  }
  .empty small {
    font-size: 0.76rem;
    max-width: 34ch;
    line-height: 1.5;
  }
  .empty-icon {
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    margin-bottom: 0.15rem;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    color: var(--text-faint);
  }

  .spinner {
    width: 24px;
    height: 24px;
    margin-bottom: 0.2rem;
    border-radius: 50%;
    border: 2px solid var(--surface-3);
    border-top-color: var(--accent);
    animation: spin 0.85s linear infinite;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* ── Segments & words ───────────────────────────────────────────────────── */
  .subsection {
    padding: 0 1rem 1rem;
    border-top: 1px solid var(--line);
  }
  .subsection h3 {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.9rem 0 0.5rem;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-faint);
  }

  .table-wrap {
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
  }
  thead th {
    padding: 0.45rem 0.7rem;
    text-align: left;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-faint);
    background: var(--surface-2);
  }
  tbody td {
    padding: 0.45rem 0.7rem;
    border-top: 1px solid var(--line);
    vertical-align: top;
  }
  tbody tr:hover {
    background: var(--surface-2);
  }
  td.time {
    font-family: var(--font-mono);
    font-size: 0.76rem;
    color: var(--text-faint);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }

  .words-details {
    border-top: 1px solid var(--line);
    padding: 0 1rem 1rem;
  }
  .words-details summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding: 0.85rem 0 0.15rem;
    font-size: 0.76rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-faint);
    cursor: pointer;
    list-style: none;
  }
  .words-details summary::-webkit-details-marker {
    display: none;
  }
  .words-details summary > span {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
  }

  .chev {
    display: grid;
    place-items: center;
    color: var(--text-faint);
    transition: transform 0.2s ease;
  }
  .advanced[open] .chev,
  .words-details[open] .chev {
    transform: rotate(180deg);
  }

  .words {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.6rem;
  }
  .word {
    padding: 0.12rem 0.45rem;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: var(--surface-2);
    font-size: 0.78rem;
    color: var(--text-dim);
    transition: color 0.12s ease, border-color 0.12s ease;
  }
  .word:hover {
    color: var(--text);
    border-color: var(--line-strong);
  }

  .badge {
    padding: 0.05rem 0.4rem;
    border-radius: 4px;
    background: var(--surface-3);
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0;
    color: var(--text-dim);
    font-variant-numeric: tabular-nums;
  }
  .badge-live {
    color: #ffa8a8;
    background: rgba(255, 118, 118, 0.14);
    text-transform: uppercase;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    animation: blink 1.6s ease-in-out infinite;
  }

  /* ── Alerts ─────────────────────────────────────────────────────────────── */
  .alert {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    margin: 0.85rem 0 0;
    padding: 0.6rem 0.75rem;
    border: 1px solid rgba(255, 118, 118, 0.28);
    border-radius: var(--radius-sm);
    background: rgba(255, 118, 118, 0.07);
    color: #ffb4b4;
    font-size: 0.8rem;
    line-height: 1.45;
  }
  main > .alert {
    margin: 0;
  }
  .alert :global(svg) {
    flex-shrink: 0;
    margin-top: 0.1rem;
  }
  .alert-dismiss {
    display: grid;
    place-items: center;
    margin-left: auto;
    padding: 0.15rem;
    flex-shrink: 0;
    border: none;
    border-radius: 3px;
    background: transparent;
    color: inherit;
    opacity: 0.7;
  }
  .alert-dismiss:hover {
    opacity: 1;
    background: rgba(255, 255, 255, 0.08);
  }

  .mono {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }

  /* ── Responsive ─────────────────────────────────────────────────────────── */
  @media (max-width: 760px) {
    .shell {
      padding: 1.25rem 0.85rem 2rem;
    }
    .modes {
      grid-template-columns: 1fr 1fr;
    }
    .mode-hint {
      display: none;
    }
    .panel {
      padding: 0.9rem;
      border-radius: var(--radius-md);
    }
    .transcript-panel {
      padding: 0;
    }
    .live {
      gap: 0.9rem;
    }
    .mic {
      width: 60px;
      height: 60px;
    }
    .mic-core {
      width: 54px;
      height: 54px;
    }
    .live-help {
      font-size: 0.78rem;
    }
    .capture {
      grid-template-columns: 1fr;
    }
    .btn-record {
      width: 100%;
    }
    .advanced-grid {
      grid-template-columns: 1fr;
    }
    .drop-hint {
      display: none;
    }
    .langs {
      gap: 0.55rem;
    }
    .translate-badge {
      margin-bottom: 0;
    }
    .transcript-body {
      max-height: none;
      font-size: 1rem;
    }
    .word-count {
      display: none;
    }
  }

  @media (max-width: 420px) {
    .slider-row {
      flex-wrap: wrap;
    }
    .range {
      order: 3;
      flex-basis: 100%;
    }
  }
</style>
