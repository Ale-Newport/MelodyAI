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

function requestMicrophone(deviceId?: string | null) {
  return navigator.mediaDevices.getUserMedia({
    audio: {
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      echoCancellation: false, // processing blocks distort pitch; we want the raw voice
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });
}

function errorName(error: unknown) {
  return error && typeof error === "object" && "name" in error ? error.name : undefined;
}

export function useRecorder(opts: { maxSeconds: number; deviceId?: string | null; onComplete: (r: Recording) => void }) {
  const [state, setState] = useState<RecorderState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const attemptRef = useRef<{ recording: boolean } | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const startRef = useRef(0);
  const rafRef = useRef(0);
  const doneRef = useRef(opts.onComplete);
  doneRef.current = opts.onComplete;

  const teardown = useCallback(() => {
    attemptRef.current = null;
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    ctxRef.current?.close().catch(() => undefined);
    ctxRef.current = null;
    analyserRef.current = null;
    chunksRef.current = [];
  }, []);

  const finish = useCallback(
    (emit: boolean) => {
      const recorded = attemptRef.current?.recording;
      const ctx = ctxRef.current;
      const rate = ctx?.sampleRate ?? 48000;
      const chunks = chunksRef.current;
      teardown();
      setLevel(0);
      setState("idle");
      if (!emit || !recorded) return;
      const total = chunks.reduce((n, c) => n + c.length, 0);
      if (!total) {
        setState("error");
        setError("No microphone audio was captured.");
        return;
      }
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
    if (attemptRef.current) return;
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setState("error");
      setError("Microphone access requires a secure context (https or localhost).");
      return;
    }
    const attempt = { recording: false };
    attemptRef.current = attempt;
    setState("requesting");
    setElapsed(0);
    try {
      let stream: MediaStream;
      try {
        stream = await requestMicrophone(opts.deviceId);
      } catch (e) {
        if (attemptRef.current !== attempt) return;
        if (!opts.deviceId || errorName(e) !== "OverconstrainedError") throw e;
        stream = await requestMicrophone();
      }
      if (attemptRef.current !== attempt) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const ctx = new AudioContext();
      ctxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      src.connect(analyser);
      analyserRef.current = analyser;
      const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
      try {
        await ctx.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      if (attemptRef.current !== attempt) return;
      const node = new AudioWorkletNode(ctx, "melodyai-capture");
      node.port.onmessage = (e) => {
        if (attemptRef.current === attempt) chunksRef.current.push(e.data as Float32Array);
      };
      src.connect(node);
      const sink = ctx.createGain();
      sink.gain.value = 0;
      node.connect(sink).connect(ctx.destination);
      startRef.current = performance.now();
      attempt.recording = true;
      setState("recording");
      const buf = new Float32Array(analyser.fftSize);
      const tick = () => {
        if (attemptRef.current !== attempt) return;
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
    } catch (e) {
      if (attemptRef.current !== attempt) return;
      teardown();
      const name = errorName(e);
      const denied = name === "NotAllowedError" || name === "SecurityError";
      const message = e && typeof e === "object" && "message" in e && typeof e.message === "string" ? e.message : "";
      setState(denied ? "denied" : "error");
      setError(denied ? "Microphone permission was denied." : message || "Could not start recording");
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
