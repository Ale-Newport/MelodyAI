/** Linear-interpolation resampler (adequate for 16 kHz speech/melody analysis). */
export function resampleLinear(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const n = Math.floor(input.length / ratio);
  const out = new Float32Array(n);
  // Simple anti-aliasing: average over the decimation span when downsampling.
  const span = Math.max(1, Math.floor(ratio));
  for (let i = 0; i < n; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    if (span > 1) {
      let acc = 0;
      let cnt = 0;
      for (let k = 0; k < span && i0 + k < input.length; k++) {
        acc += input[i0 + k];
        cnt++;
      }
      out[i] = cnt ? acc / cnt : 0;
    } else {
      const frac = pos - i0;
      const a = input[i0] ?? 0;
      const b = input[i0 + 1] ?? a;
      out[i] = a + (b - a) * frac;
    }
  }
  return out;
}

/** Encode mono float samples as a 16-bit PCM WAV file. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

/** Decode any browser-supported audio file and convert it to a 16 kHz mono WAV.
 *  Falls back to the original file if the browser cannot decode it (the server may). */
export async function fileToWav(file: Blob, targetRate = 16000): Promise<Blob> {
  try {
    const ctx = new OfflineAudioContext(1, 1, 44100);
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    const mono = new Float32Array(buf.length);
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < d.length; i++) mono[i] += d[i] / buf.numberOfChannels;
    }
    return encodeWav(resampleLinear(mono, buf.sampleRate, targetRate), targetRate);
  } catch {
    return file;
  }
}
