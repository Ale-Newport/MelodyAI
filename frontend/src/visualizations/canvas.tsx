/**
 * Canvas visualisations for melodic analysis. Every plot draws data returned by the
 * backend (pitch tracks, notes, chroma, DTW matrices); nothing here is decorative.
 */
import { useEffect, useRef, type MutableRefObject } from "react";
import type { Alignment, NotesData, PitchTrackData } from "@/types/api";
import { midiName } from "@/lib/utils";

export function cssColor(name: string, alpha = 1): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim() || "160 160 160";
  const [r, g, b] = v.split(/\s+/);
  return alpha < 1 ? `rgba(${r},${g},${b},${alpha})` : `rgb(${r},${g},${b})`;
}

type Draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => void;

/** Canvas that redraws on resize and whenever ``deps`` change (handles device pixel ratio). */
export function Plot({ draw, deps, height = 180, label, className }: { draw: Draw; deps: unknown[]; height?: number; label: string; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      canvas.width = Math.max(1, Math.floor(w * dpr));
      canvas.height = Math.max(1, Math.floor(h * dpr));
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      draw(ctx, w, h);
    };
    render();
    const ro = new ResizeObserver(render);
    ro.observe(canvas);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return <canvas ref={ref} role="img" aria-label={label} className={className ?? "w-full rounded-xl bg-elevated/40"} style={{ height }} />;
}

const PAD = { l: 40, r: 8, t: 8, b: 20 };

function axes(ctx: CanvasRenderingContext2D, w: number, h: number, x0: number, x1: number, xlabel = "s") {
  ctx.strokeStyle = cssColor("border");
  ctx.fillStyle = cssColor("muted");
  ctx.font = "10px 'JetBrains Mono Variable', monospace";
  ctx.lineWidth = 1;
  const span = x1 - x0;
  const step = span > 40 ? 10 : span > 16 ? 4 : span > 6 ? 2 : span > 2 ? 0.5 : 0.25;
  for (let t = Math.ceil(x0 / step) * step; t <= x1; t += step) {
    const x = PAD.l + ((t - x0) / (span || 1)) * (w - PAD.l - PAD.r);
    ctx.beginPath();
    ctx.moveTo(x, h - PAD.b);
    ctx.lineTo(x, h - PAD.b + 4);
    ctx.stroke();
    ctx.fillText(`${+t.toFixed(2)}${xlabel}`, x - 8, h - 5);
  }
}

/* ------------------------------------------------------------------ live input */
export function LiveWaveform({ analyser, active, height = 96 }: { analyser: MutableRefObject<AnalyserNode | null>; active: boolean; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let raf = 0;
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const buf = new Float32Array(2048);
    const history: number[] = [];
    const loop = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const a = analyser.current;
      if (a && active) {
        a.getFloatTimeDomainData(buf);
        let peak = 0;
        for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i]));
        history.push(Math.min(1, peak * 1.6));
        if (history.length > w / 3) history.shift();
      }
      ctx.fillStyle = cssColor("highlight");
      const n = history.length;
      for (let i = 0; i < n; i++) {
        const v = Math.max(0.02, history[i]) * (h / 2 - 2);
        const x = w - (n - i) * 3;
        ctx.globalAlpha = 0.35 + 0.65 * (i / n);
        ctx.fillRect(x, h / 2 - v, 2, 2 * v);
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [analyser, active]);
  return <canvas ref={ref} aria-label="Live input level" className="w-full" style={{ height }} />;
}

/* ------------------------------------------------------------------ waveform */
export function Waveform({ min, max, duration, height = 120 }: { min: number[]; max: number[]; duration: number; height?: number }) {
  return (
    <Plot
      label="Waveform"
      height={height}
      deps={[min, max]}
      draw={(ctx, w, h) => {
        const n = max.length;
        const mid = (h - PAD.b) / 2;
        ctx.fillStyle = cssColor("highlight", 0.85);
        for (let i = 0; i < n; i++) {
          const x = PAD.l + (i / n) * (w - PAD.l - PAD.r);
          ctx.fillRect(x, mid - max[i] * mid, Math.max(1, (w - PAD.l) / n), (max[i] - min[i]) * mid || 1);
        }
        axes(ctx, w, h, 0, duration);
      }}
    />
  );
}

/* ------------------------------------------------------------------ spectrogram */
// Single-hue sequential map in the brand violet (dark → violet → near white), the same in both themes so
// that luminance always grows with the value; the cyan pitch / warping path stays readable on top of it.
const SEQUENTIAL = [
  [10, 9, 16], [30, 22, 58], [58, 40, 118], [99, 70, 190], [139, 108, 236], [167, 139, 250], [204, 188, 252], [244, 240, 255],
];
function colormap(v: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, v)) * (SEQUENTIAL.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = SEQUENTIAL[i];
  const b = SEQUENTIAL[Math.min(i + 1, SEQUENTIAL.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export function Spectrogram({ data, hop, freqs, pitch, height = 220 }: { data: number[][]; hop: number; freqs: number[]; pitch?: PitchTrackData | null; height?: number }) {
  return (
    <Plot
      label="Spectrogram with pitch overlay"
      height={height}
      deps={[data, pitch]}
      draw={(ctx, w, h) => {
        const bins = data.length;
        const frames = data[0]?.length ?? 0;
        if (!frames) return;
        const img = ctx.createImageData(frames, bins);
        for (let b = 0; b < bins; b++) {
          for (let f = 0; f < frames; f++) {
            const [r, g, bl] = colormap(data[b][f] / 255);
            const o = ((bins - 1 - b) * frames + f) * 4;
            img.data[o] = r;
            img.data[o + 1] = g;
            img.data[o + 2] = bl;
            img.data[o + 3] = 255;
          }
        }
        const off = document.createElement("canvas");
        off.width = frames;
        off.height = bins;
        off.getContext("2d")!.putImageData(img, 0, 0);
        const pw = w - PAD.l - PAD.r;
        const ph = h - PAD.t - PAD.b;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(off, PAD.l, PAD.t, pw, ph);
        const dur = frames * hop;
        const lf0 = Math.log(freqs[0]);
        const lf1 = Math.log(freqs[freqs.length - 1]);
        const yOf = (hz: number) => PAD.t + ph * (1 - (Math.log(hz) - lf0) / (lf1 - lf0));
        ctx.fillStyle = cssColor("muted");
        ctx.font = "10px 'JetBrains Mono Variable', monospace";
        for (const f of [100, 200, 500, 1000, 2000, 4000]) {
          if (f < freqs[0] || f > freqs[freqs.length - 1]) continue;
          ctx.fillText(f >= 1000 ? `${f / 1000}k` : `${f}`, 4, yOf(f) + 3);
        }
        if (pitch) {
          ctx.strokeStyle = "#22d3ee";
          ctx.lineWidth = 2;
          ctx.beginPath();
          let pen = false;
          pitch.times.forEach((t, i) => {
            const hz = pitch.f0_hz[i];
            if (hz == null || !pitch.voiced[i]) {
              pen = false;
              return;
            }
            const x = PAD.l + (t / dur) * pw;
            const y = yOf(hz);
            if (pen) ctx.lineTo(x, y);
            else ctx.moveTo(x, y);
            pen = true;
          });
          ctx.stroke();
        }
        axes(ctx, w, h, 0, dur);
      }}
    />
  );
}

/* ------------------------------------------------------------------ pitch contour + notes */
export function PitchContour({ track, notes, height = 220, showConfidence = true, showNotes = true }: {
  track: PitchTrackData;
  notes?: { onset: number; duration: number; midi: number }[];
  height?: number;
  showConfidence?: boolean;
  showNotes?: boolean;
}) {
  return (
    <Plot
      label="Pitch contour (MIDI) with voicing confidence and segmented notes"
      height={height}
      deps={[track, notes, showConfidence, showNotes]}
      draw={(ctx, w, h) => {
        const vals = track.midi.filter((m): m is number => m != null);
        if (!vals.length) {
          ctx.fillStyle = cssColor("muted");
          ctx.fillText("No voiced frames", w / 2 - 40, h / 2);
          return;
        }
        const lo = Math.floor(Math.min(...vals)) - 2;
        const hi = Math.ceil(Math.max(...vals)) + 2;
        const dur = track.times[track.times.length - 1] || 1;
        const pw = w - PAD.l - PAD.r;
        const ph = h - PAD.t - PAD.b;
        const X = (t: number) => PAD.l + (t / dur) * pw;
        const Y = (m: number) => PAD.t + ph * (1 - (m - lo) / (hi - lo));
        // semitone grid with note names
        ctx.font = "10px 'JetBrains Mono Variable', monospace";
        for (let m = lo; m <= hi; m++) {
          const black = [1, 3, 6, 8, 10].includes(((m % 12) + 12) % 12);
          ctx.fillStyle = black ? cssColor("elevated", 0.6) : "transparent";
          ctx.fillRect(PAD.l, Y(m + 0.5), pw, Y(m - 0.5) - Y(m + 0.5));
          if (m % 12 === 0 || hi - lo < 16) {
            ctx.fillStyle = cssColor("muted");
            ctx.fillText(midiName(m), 2, Y(m) + 3);
          }
        }
        if (showConfidence) {
          // voicing probability as a band along the bottom
          track.times.forEach((t, i) => {
            ctx.fillStyle = cssColor(track.voiced[i] ? "viz-voiced" : "viz-unvoiced", 0.15 + 0.6 * track.voiced_prob[i]);
            ctx.fillRect(X(t), h - PAD.b - 4, Math.max(1, pw / track.times.length), 4);
          });
        }
        if (showNotes && notes) {
          ctx.fillStyle = cssColor("brand", 0.28);
          ctx.strokeStyle = cssColor("brand", 0.8);
          for (const n of notes) {
            const x = X(n.onset);
            const y = Y(n.midi + 0.45);
            ctx.fillRect(x, y, X(n.onset + n.duration) - x, Y(n.midi - 0.45) - y);
            ctx.strokeRect(x, y, X(n.onset + n.duration) - x, Y(n.midi - 0.45) - y);
          }
        }
        ctx.strokeStyle = cssColor("highlight");
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        let pen = false;
        track.times.forEach((t, i) => {
          const m = track.midi[i];
          if (m == null) {
            pen = false;
            return;
          }
          if (pen) ctx.lineTo(X(t), Y(m));
          else ctx.moveTo(X(t), Y(m));
          pen = true;
        });
        ctx.stroke();
        axes(ctx, w, h, 0, dur);
      }}
    />
  );
}

/* ------------------------------------------------------------------ piano roll */
export function PianoRoll({ notes, reference, height = 220 }: { notes: NotesData; reference?: NotesData; height?: number }) {
  return (
    <Plot
      label="Piano roll of detected notes"
      height={height}
      deps={[notes, reference]}
      draw={(ctx, w, h) => {
        const all = [...notes.pitches, ...(reference?.pitches ?? [])];
        if (!all.length) return;
        const lo = Math.floor(Math.min(...all)) - 1;
        const hi = Math.ceil(Math.max(...all)) + 1;
        const end = (n: NotesData) => (n.onsets.length ? n.onsets[n.onsets.length - 1] + n.durations[n.durations.length - 1] : 0);
        const dur = Math.max(end(notes), reference ? end(reference) : 0) || 1;
        const pw = w - PAD.l - PAD.r;
        const ph = h - PAD.t - PAD.b;
        const rowH = ph / (hi - lo + 1);
        ctx.font = "10px 'JetBrains Mono Variable', monospace";
        for (let m = lo; m <= hi; m++) {
          const y = PAD.t + (hi - m) * rowH;
          const black = [1, 3, 6, 8, 10].includes(((m % 12) + 12) % 12);
          ctx.fillStyle = black ? cssColor("elevated", 0.9) : cssColor("surface", 0.4);
          ctx.fillRect(PAD.l, y, pw, rowH);
          if (rowH > 9 || m % 12 === 0) {
            ctx.fillStyle = cssColor("muted");
            ctx.fillText(midiName(m), 2, y + rowH * 0.75);
          }
        }
        const drawNotes = (n: NotesData, color: string, inset: number) => {
          ctx.fillStyle = color;
          n.onsets.forEach((o, i) => {
            const x = PAD.l + (o / dur) * pw;
            const y = PAD.t + (hi - Math.round(n.pitches[i])) * rowH;
            ctx.beginPath();
            ctx.roundRect(x, y + inset, Math.max(2, (n.durations[i] / dur) * pw - 1), rowH - 2 * inset, 3);
            ctx.fill();
          });
        };
        if (reference) drawNotes(reference, cssColor("muted", 0.45), 1);
        drawNotes(notes, cssColor("brand", 0.9), reference ? rowH * 0.25 : 1);
        axes(ctx, w, h, 0, dur);
      }}
    />
  );
}

/* ------------------------------------------------------------------ chroma */
const PCS = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
export function Chromagram({ data, rate, height = 170 }: { data: number[][]; rate: number; height?: number }) {
  return (
    <Plot
      label="Chromagram (CQT chroma)"
      height={height}
      deps={[data]}
      draw={(ctx, w, h) => {
        const T = data[0]?.length ?? 0;
        if (!T) return;
        const pw = w - PAD.l - PAD.r;
        const ph = h - PAD.t - PAD.b;
        const rh = ph / 12;
        const cw = pw / T;
        ctx.font = "10px 'JetBrains Mono Variable', monospace";
        for (let k = 0; k < 12; k++) {
          const y = PAD.t + (11 - k) * rh;
          ctx.fillStyle = cssColor("muted");
          ctx.fillText(PCS[k], 6, y + rh * 0.7);
          for (let t = 0; t < T; t++) {
            const [r, g, b] = colormap(data[k][t]);
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(PAD.l + t * cw, y, cw + 0.5, rh - 0.5);
          }
        }
        axes(ctx, w, h, 0, T / rate);
      }}
    />
  );
}

/* ------------------------------------------------------------------ DTW */
export function CostMatrix({ alignment, height = 260 }: { alignment: Alignment; height?: number }) {
  return (
    <Plot
      label="Accumulated DTW cost matrix with optimal warping path"
      height={height}
      deps={[alignment]}
      draw={(ctx, w, h) => {
        const M = alignment.accumulated_cost;
        if (!M?.length) return;
        const rows = M.length;
        const cols = M[0].length;
        const pw = w - PAD.l - PAD.r;
        const ph = h - PAD.t - PAD.b;
        const img = ctx.createImageData(cols, rows);
        for (let i = 0; i < rows; i++)
          for (let j = 0; j < cols; j++) {
            const [r, g, b] = colormap(1 - M[i][j]);
            const o = ((rows - 1 - i) * cols + j) * 4;
            img.data.set([r, g, b, 255], o);
          }
        const off = document.createElement("canvas");
        off.width = cols;
        off.height = rows;
        off.getContext("2d")!.putImageData(img, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(off, PAD.l, PAD.t, pw, ph);
        const qLen = alignment.query_contour.length;
        const rLen = alignment.reference_contour.length;
        ctx.strokeStyle = "#22d3ee";
        ctx.lineWidth = 2;
        ctx.beginPath();
        alignment.path.forEach(([i, j], k) => {
          const x = PAD.l + (j / Math.max(1, rLen - 1)) * pw;
          const y = PAD.t + ph * (1 - i / Math.max(1, qLen - 1));
          if (k) ctx.lineTo(x, y);
          else ctx.moveTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = cssColor("muted");
        ctx.font = "10px 'JetBrains Mono Variable', monospace";
        ctx.save();
        ctx.translate(10, PAD.t + ph / 2 + 20);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText("query →", 0, 0);
        ctx.restore();
        axes(ctx, w, h, alignment.reference_t0, alignment.reference_t0 + rLen / alignment.rate_hz);
      }}
    />
  );
}

export function AlignedContours({ alignment, height = 200 }: { alignment: Alignment; height?: number }) {
  return (
    <Plot
      label="Query contour warped onto the reference contour"
      height={height}
      deps={[alignment]}
      draw={(ctx, w, h) => {
        const ref = alignment.reference_contour;
        const q = alignment.query_contour;
        const [a, b] = [alignment.path[0]?.[1] ?? 0, alignment.path[alignment.path.length - 1]?.[1] ?? ref.length - 1];
        const lo0 = Math.max(0, a - 15);
        const hi0 = Math.min(ref.length - 1, b + 15);
        const seg = ref.slice(lo0, hi0 + 1);
        const vals = [...seg, ...alignment.path.map(([i]) => q[i])];
        const lo = Math.floor(Math.min(...vals)) - 1;
        const hi = Math.ceil(Math.max(...vals)) + 1;
        const pw = w - PAD.l - PAD.r;
        const ph = h - PAD.t - PAD.b;
        const X = (j: number) => PAD.l + ((j - lo0) / Math.max(1, hi0 - lo0)) * pw;
        const Y = (m: number) => PAD.t + ph * (1 - (m - lo) / (hi - lo));
        ctx.fillStyle = cssColor("highlight", 0.08);
        ctx.fillRect(X(a), PAD.t, X(b) - X(a), ph);
        ctx.font = "10px 'JetBrains Mono Variable', monospace";
        for (let m = lo; m <= hi; m += hi - lo > 14 ? 5 : 2) {
          ctx.fillStyle = cssColor("muted");
          ctx.fillText(midiName(m), 2, Y(m) + 3);
        }
        ctx.lineWidth = 3;
        ctx.strokeStyle = cssColor("muted", 0.7);
        ctx.beginPath();
        seg.forEach((m, k) => (k ? ctx.lineTo(X(lo0 + k), Y(m)) : ctx.moveTo(X(lo0 + k), Y(m))));
        ctx.stroke();
        ctx.lineWidth = 2;
        ctx.strokeStyle = cssColor("brand");
        ctx.beginPath();
        alignment.path.forEach(([i, j], k) => (k ? ctx.lineTo(X(j), Y(q[i])) : ctx.moveTo(X(j), Y(q[i]))));
        ctx.stroke();
        const t0 = alignment.reference_t0 + lo0 / alignment.rate_hz;
        axes(ctx, w, h, t0, alignment.reference_t0 + hi0 / alignment.rate_hz);
      }}
    />
  );
}
