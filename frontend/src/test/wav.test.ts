import { encodeWav, resampleLinear } from "@/lib/wav";
import { blobToArrayBuffer } from "./setup";

describe("wav encoding", () => {
  it("writes a valid 16-bit mono RIFF header and clips samples", async () => {
    const blob = encodeWav(new Float32Array([0, 0.5, -0.5, 2, -2]), 16000);
    const buf = new DataView(await blobToArrayBuffer(blob));
    const str = (o: number, n: number) => String.fromCharCode(...Array.from({ length: n }, (_, i) => buf.getUint8(o + i)));
    expect(str(0, 4)).toBe("RIFF");
    expect(str(8, 4)).toBe("WAVE");
    expect(buf.getUint32(24, true)).toBe(16000);
    expect(buf.getUint16(34, true)).toBe(16);
    expect(buf.getUint32(40, true)).toBe(10);
    expect(buf.getInt16(44 + 3 * 2, true)).toBe(32767); // clipped +2
    expect(buf.getInt16(44 + 4 * 2, true)).toBe(-32768); // clipped -2
  });

  it("resamples 48 kHz to 16 kHz preserving duration and a sine frequency", () => {
    const sr = 48000;
    const x = new Float32Array(sr).map((_, i) => Math.sin((2 * Math.PI * 440 * i) / sr));
    const y = resampleLinear(x, sr, 16000);
    expect(y.length).toBe(16000);
    let crossings = 0;
    for (let i = 1; i < y.length; i++) if (y[i - 1] < 0 && y[i] >= 0) crossings++;
    expect(Math.abs(crossings - 440)).toBeLessThanOrEqual(2);
  });
});
