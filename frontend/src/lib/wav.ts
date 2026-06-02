// Decode an arbitrary audio Blob (e.g. webm/opus from MediaRecorder, or an
// uploaded file) and re-encode it as a 16 kHz mono 16-bit PCM WAV. The backend
// accepts any WAV, but normalizing here keeps uploads small and consistent.

const TARGET_SAMPLE_RATE = 16000

export async function blobToWav16kMono(blob: Blob): Promise<Blob> {
  const arrayBuffer = await blob.arrayBuffer()

  const AudioContextClass: typeof AudioContext =
    globalThis.AudioContext ??
    (globalThis as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext

  const decodeCtx = new AudioContextClass()
  let decoded: AudioBuffer
  try {
    decoded = await decodeCtx.decodeAudioData(arrayBuffer)
  } finally {
    await decodeCtx.close()
  }

  // Resample + downmix to mono via an OfflineAudioContext. Routing a
  // multichannel source into a 1-channel destination downmixes per spec.
  const frameCount = Math.max(1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE))
  const offline = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE)
  const source = offline.createBufferSource()
  source.buffer = decoded
  source.connect(offline.destination)
  source.start()
  const rendered = await offline.startRendering()

  return encodeWav(rendered.getChannelData(0), TARGET_SAMPLE_RATE)
}

export function float32ToWav(samples: Float32Array, sampleRate = 16000): Blob {
  return encodeWav(samples, sampleRate)
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const numSamples = samples.length
  const buffer = new ArrayBuffer(44 + numSamples * 2)
  const view = new DataView(buffer)

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i))
    }
  }

  // RIFF header
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + numSamples * 2, true)
  writeString(8, 'WAVE')

  // fmt chunk (PCM)
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true) // chunk size
  view.setUint16(20, 1, true) // audio format = PCM
  view.setUint16(22, 1, true) // channels = mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample

  // data chunk
  writeString(36, 'data')
  view.setUint32(40, numSamples * 2, true)

  let offset = 44
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }

  return new Blob([view], { type: 'audio/wav' })
}
