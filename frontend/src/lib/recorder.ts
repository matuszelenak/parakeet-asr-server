// Thin wrapper around getUserMedia + MediaRecorder. Produces a Blob in whatever
// container the browser supports (usually webm/opus); it is converted to WAV
// client-side before upload (see wav.ts).

export class Recorder {
  private mediaRecorder: MediaRecorder | null = null
  private stream: MediaStream | null = null
  private chunks: Blob[] = []

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    this.chunks = []
    this.mediaRecorder = new MediaRecorder(this.stream)
    this.mediaRecorder.ondataavailable = (event: BlobEvent) => {
      if (event.data && event.data.size > 0) this.chunks.push(event.data)
    }
    this.mediaRecorder.start()
  }

  stop(): Promise<Blob> {
    return new Promise((resolve, reject) => {
      const recorder = this.mediaRecorder
      const stream = this.stream
      if (!recorder || !stream) {
        reject(new Error('recorder is not running'))
        return
      }
      recorder.onstop = () => {
        const blob = new Blob(this.chunks, {
          type: recorder.mimeType || 'audio/webm',
        })
        stream.getTracks().forEach((track) => track.stop())
        this.mediaRecorder = null
        this.stream = null
        resolve(blob)
      }
      recorder.stop()
    })
  }
}
