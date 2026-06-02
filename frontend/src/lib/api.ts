// By default, the API is assumed to live at the same address the frontend was
// opened from (same protocol/host/port). Requests are routed to the backend by
// a same-origin proxy: the Vite dev server in development.
// Set VITE_API_BASE to point at a different origin if you are not proxying.
const configured = import.meta.env.VITE_API_BASE
const BASE =
  configured && configured.length > 0 ? configured : window.location.origin

export type Mode = 'plain' | 'timestamps' | 'longform'

const ENDPOINTS: Record<Mode, string> = {
  plain: '/v1/transcribe',
  timestamps: '/v1/transcribe/timestamps',
  longform: '/v1/transcribe/longform',
}

export interface WordTimestamp {
  word: string
  start: number
  end: number
}

export interface SegmentTimestamp {
  segment: string
  start: number
  end: number
}

export interface CharTimestamp {
  char: string
  start: number
  end: number
}

export interface TranscriptionResult {
  text: string
  words?: WordTimestamp[]
  segments?: SegmentTimestamp[]
  chars?: CharTimestamp[]
}

export interface LanguageInfo {
  code: string
  name: string
}

export interface Capabilities {
  supportsLanguages: boolean
  languages: LanguageInfo[]
}

export interface StreamEvent {
  type: 'partial' | 'committed' | 'final'
  text: string
  start: number
  id: number
}

// State of the live transcription pipeline reported to the UI.
export type LiveState = 'listening' | 'speaking' | 'committing'

export interface StreamParams {
  minDuration: number
  retranscribeInterval: number
  stableWords: number
  stableIters: number
  maxDuration: number
  contextDuration: number
}

export const STREAM_PARAM_DEFAULTS: StreamParams = {
  minDuration: 1.0,
  retranscribeInterval: 0.5,
  stableWords: 4,
  stableIters: 2,
  maxDuration: 30.0,
  contextDuration: 3.0,
}

export async function fetchCapabilities(): Promise<Capabilities> {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) return { supportsLanguages: false, languages: [] }
  const body = await res.json()
  return {
    supportsLanguages: Boolean(body.supports_languages),
    languages: Array.isArray(body.languages) ? body.languages : [],
  }
}

export interface TranscribeOptions {
  sourceLang?: string
  targetLang?: string
}

// VAD gates the microphone so only speech frames reach the server.
// The WebSocket streaming endpoint drives real-time partial→committed transcription.
// After the user stops speaking a configurable delay elapses, then {type:'end'} is
// sent which causes the server to finalise the current segment and close the
// connection. The class then reopens the WebSocket automatically for the next
// utterance.
export class VadTranscriber {
  private myvad: import('@ricky0123/vad-web').MicVAD | null = null
  private ws: WebSocket | null = null
  private _wsOptions: TranscribeOptions = {}
  private _streamParams: StreamParams = STREAM_PARAM_DEFAULTS
  private _speaking = false
  private _stopping = false
  private _expectingClose = false
  private _commitTimer: ReturnType<typeof setTimeout> | null = null

  constructor(
    private readonly onEvent: (event: StreamEvent) => void,
    private readonly onActivity: (prob: number, speaking: boolean) => void,
    private readonly onState: (state: LiveState) => void,
    private readonly onError: (message: string) => void,
  ) {}

  private async _openWs(): Promise<void> {
    const options = this._wsOptions
    const params = new URLSearchParams()
    if (options.sourceLang) params.set('source_lang', options.sourceLang)
    if (options.targetLang) params.set('target_lang', options.targetLang)
    const qs = params.toString() ? `?${params}` : ''
    const wsBase = BASE.replace(/^https:/, 'wss:').replace(/^http:/, 'ws:')

    this.ws = new WebSocket(`${wsBase}/v1/transcribe/stream${qs}`)
    this.ws.binaryType = 'arraybuffer'

    await new Promise<void>((resolve, reject) => {
      this.ws!.onopen = () => resolve()
      this.ws!.onerror = () => reject(new Error('Could not connect to server'))
    })

    // Send inference parameters as the first frame so the server can apply
    // them before any audio arrives.
    const p = this._streamParams
    this.ws.send(JSON.stringify({
      type: 'configure',
      min_duration: p.minDuration,
      retranscribe_interval: p.retranscribeInterval,
      stable_words: p.stableWords,
      stable_iters: p.stableIters,
      max_duration: p.maxDuration,
      context_duration: p.contextDuration,
    }))

    this.ws.onmessage = (ev) => {
      try { this.onEvent(JSON.parse(ev.data as string) as StreamEvent) } catch {}
    }
    this.ws.onerror = () => {
      if (!this._stopping && !this._expectingClose) this.onError('WebSocket error')
    }
    // When we sent {type:'end'} the server finalises and closes the socket.
    // Reopen it silently for the next utterance.
    this.ws.onclose = () => {
      if (this._expectingClose && !this._stopping) {
        this._expectingClose = false
        this._openWs().then(() => {
          this.onState('listening')
        }).catch(() => {
          if (!this._stopping) this.onError('Failed to reconnect after commit')
        })
      } else if (!this._stopping) {
        this.onError('Connection closed unexpectedly')
      }
    }
  }

  async start(
    options: TranscribeOptions = {},
    commitDelay = 500,
    streamParams: StreamParams = STREAM_PARAM_DEFAULTS,
  ): Promise<void> {
    this._stopping = false
    this._wsOptions = options
    this._streamParams = streamParams

    await this._openWs()
    this.onState('listening')

    const { MicVAD } = await import('@ricky0123/vad-web')

    this.myvad = await MicVAD.new({
      model: 'v5',
      startOnLoad: true,
      baseAssetPath: '/',
      onnxWASMBasePath: '/',
      ortConfig: (ort) => {
        ort.env.logLevel = 'error'
        ort.env.wasm.numThreads = 1
      },
      onFrameProcessed: (probs, frame) => {
        this.onActivity(probs.isSpeech, this._speaking)
        if (this._speaking && this.ws?.readyState === WebSocket.OPEN) {
          const i16 = new Int16Array(frame.length)
          for (let i = 0; i < frame.length; i++) {
            i16[i] = Math.max(-32768, Math.min(32767, Math.round(frame[i] * 32767)))
          }
          this.ws.send(i16.buffer)
        }
      },
      onSpeechStart: () => {
        // Cancel any pending commit if the user starts speaking again.
        if (this._commitTimer !== null) {
          clearTimeout(this._commitTimer)
          this._commitTimer = null
        }
        this._speaking = true
        this.onActivity(1, true)
        this.onState('speaking')
      },
      onSpeechRealStart: () => {},
      onVADMisfire: () => {
        this._speaking = false
        this.onActivity(0, false)
        this.onState('listening')
      },
      onSpeechEnd: () => {
        this._speaking = false
        this.onActivity(0, false)
        this.onState('committing')
        // After the delay, tell the server to finalise and commit.
        this._commitTimer = setTimeout(() => {
          this._commitTimer = null
          if (!this._stopping && this.ws?.readyState === WebSocket.OPEN) {
            this._expectingClose = true
            this.ws.send(JSON.stringify({ type: 'end' }))
          }
        }, commitDelay)
      },
    })
  }

  async stop(): Promise<void> {
    this._stopping = true
    this._speaking = false
    if (this._commitTimer !== null) {
      clearTimeout(this._commitTimer)
      this._commitTimer = null
    }
    if (this.myvad?.listening) await this.myvad.pause()
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'end' }))
    }
  }

  async destroy(): Promise<void> {
    this._stopping = true
    if (this._commitTimer !== null) {
      clearTimeout(this._commitTimer)
      this._commitTimer = null
    }
    if (this.myvad) {
      try { await this.myvad.destroy() } catch {}
      this.myvad = null
    }
    this.ws?.close()
    this.ws = null
    this._speaking = false
  }
}

export async function transcribe(
  wavBlob: Blob,
  mode: Mode,
  options: TranscribeOptions = {},
): Promise<TranscriptionResult> {
  const path = ENDPOINTS[mode]

  const form = new FormData()
  form.append('file', wavBlob, 'recording.wav')
  if (options.sourceLang) form.append('source_lang', options.sourceLang)
  if (options.targetLang) form.append('target_lang', options.targetLang)

  const res = await fetch(`${BASE}${path}`, { method: 'POST', body: form })
  if (!res.ok) {
    let detail = `request failed (${res.status})`
    try {
      const body = await res.json()
      if (body.detail) detail = body.detail
    } catch {
      // non-JSON error body; keep the default message
    }
    throw new Error(detail)
  }
  return res.json()
}
