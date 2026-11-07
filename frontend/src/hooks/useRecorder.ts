import { useCallback, useEffect, useRef, useState } from "react";
import { encodeWav, resampleLinear } from "@/lib/wav";

// Raw PCM capture in an AudioWorklet (no lossy codec, no browser-specific container).
const WORKLET = `
class Capture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("melodyai-capture", Capture);
`;

export type RecorderState = "idle" | "requesting" | "recording" | "denied" | "error";

export interface Recording {
  blob: Blob;
  duration: number;
}

export function useRecorder(opts: { maxSeconds: number; deviceId?: string | null; onComplete: (r: Recording) => void }) {
  const [state, setState] = useState<RecorderState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const startRef = useRef(0);
  const rafRef = useRef(0);
  const doneRef = useRef(opts.onComplete);
  doneRef.current = opts.onComplete;

  const teardown = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    ctxRef.current?.close().catch(() => undefined);
    ctxRef.current = null;
    analyserRef.current = null;
    setLevel(0);
  }, []);

  const finish = useCallback(
    (emit: boolean) => {
      const ctx = ctxRef.current;
      const rate = ctx?.sampleRate ?? 48000;
      const chunks = chunksRef.current;
      chunksRef.current = [];
      teardown();
      setState("idle");
      if (!emit) return;
      const total = chunks.reduce((n, c) => n + c.length, 0);
      const pcm = new Float32Array(total);
      let off = 0;
      for (const c of chunks) {
        pcm.set(c, off);
        off += c.length;
      }
      const out = resampleLinear(pcm, rate, 16000);
      doneRef.current({ blob: encodeWav(out, 16000), duration: out.length / 16000 });
    },
    [teardown],
  );

  const start = useCallback(async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setState("error");
      setError("Microphone access requires a secure context (https or localhost).");
      return;
    }
    setState("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: opts.deviceId ? { exact: opts.deviceId } : undefined,
          echoCancellation: false, // processing blocks distort pitch; we want the raw voice
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1,
        },
      });
      streamRef.current = stream;
      const ctx = new AudioContext();
      ctxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      src.connect(analyser);
      analyserRef.current = analyser;
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      await ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      const node = new AudioWorkletNode(ctx, "melodyai-capture");
      node.port.onmessage = (e) => chunksRef.current.push(e.data as Float32Array);
      src.connect(node);
      const sink = ctx.createGain();
      sink.gain.value = 0;
      node.connect(sink).connect(ctx.destination);
      chunksRef.current = [];
      startRef.current = performance.now();
      setElapsed(0);
      setState("recording");
      const buf = new Float32Array(analyser.fftSize);
      const tick = () => {
        const t = (performance.now() - startRef.current) / 1000;
        setElapsed(t);
        analyser.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 4));
        if (t >= opts.maxSeconds) {
          finish(true);
          return;
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    } catch (e: any) {
      teardown();
      const denied = e?.name === "NotAllowedError" || e?.name === "SecurityError";
      setState(denied ? "denied" : "error");
      setError(denied ? "Microphone permission was denied." : e?.message ?? "Could not start recording");
    }
  }, [opts.deviceId, opts.maxSeconds, finish, teardown]);

  useEffect(() => () => teardown(), [teardown]);

  return {
    state,
    error,
    elapsed,
    level,
    analyser: analyserRef,
    start,
    stop: () => finish(true),
    cancel: () => finish(false),
  };
}
