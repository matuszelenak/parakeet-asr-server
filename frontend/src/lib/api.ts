// By default the API is assumed to live at the same address the frontend was
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

export class StreamTranscriber {
  private ws: WebSocket | null = null
  private audioCtx: AudioContext | null = null
  private source: MediaStreamAudioSourceNode | null = null
  private processor: ScriptProcessorNode | null = null
  private stream: MediaStream | null = null
  private _stopping = false

  constructor(
    private readonly onEvent: (event: StreamEvent) => void,
    private readonly onError: (message: string) => void,
    private readonly onDone: () => void,
  ) {}

  async start(options: TranscribeOptions = {}): Promise<void> {
    this._stopping = false

    const params = new URLSearchParams()
    if (options.sourceLang) params.set('source_lang', options.sourceLang)
    if (options.targetLang) params.set('target_lang', options.targetLang)
    const qs = params.toString() ? `?${params}` : ''
    const wsBase = BASE.replace(/^https:/, 'wss:').replace(/^http:/, 'ws:')
    const url = `${wsBase}/v1/transcribe/stream${qs}`

    this.ws = new WebSocket(url)
    this.ws.binaryType = 'arraybuffer'

    await new Promise<void>((resolve, reject) => {
      this.ws!.onopen = () => resolve()
      this.ws!.onerror = () => reject(new Error('Could not connect to server'))
    })

    this.ws.onmessage = (ev: MessageEvent) => {
      try {
        this.onEvent(JSON.parse(ev.data as string) as StreamEvent)
      } catch {}
    }
    this.ws.onerror = () => {
      if (!this._stopping) this.onError('WebSocket error')
    }
    this.ws.onclose = () => {
      if (!this._stopping) this.onError('Connection closed unexpectedly')
      this.onDone()
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'Microphone access requires a secure context. ' +
        'Open the app over HTTPS or via localhost (not a plain HTTP IP address).',
      )
    }

    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    this.audioCtx = new AudioContext({ sampleRate: 16_000 })
    this.source = this.audioCtx.createMediaStreamSource(this.stream)
    this.processor = this.audioCtx.createScriptProcessor(4096, 1, 1)

    this.processor.onaudioprocess = (ev: AudioProcessingEvent) => {
      if (this.ws?.readyState !== WebSocket.OPEN) return
      const f32 = ev.inputBuffer.getChannelData(0)
      const i16 = new Int16Array(f32.length)
      for (let i = 0; i < f32.length; i++) {
        i16[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)))
      }
      this.ws.send(i16.buffer)
    }

    // Muted gain node: keeps the processor running without echoing mic to speakers.
    const muted = this.audioCtx.createGain()
    muted.gain.value = 0
    this.source.connect(this.processor)
    this.processor.connect(muted)
    muted.connect(this.audioCtx.destination)
  }

  stop(): void {
    this._stopping = true
    if (this.processor) {
      this.processor.onaudioprocess = null
      this.processor.disconnect()
      this.processor = null
    }
    if (this.source) {
      this.source.disconnect()
      this.source = null
    }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop())
      this.stream = null
    }
    if (this.audioCtx) {
      this.audioCtx.close()
      this.audioCtx = null
    }
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'end' }))
    }
  }

  close(): void {
    this._stopping = true
    this.ws?.close()
    this.ws = null
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
