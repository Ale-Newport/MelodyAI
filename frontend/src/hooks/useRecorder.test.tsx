import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useRecorder } from "./useRecorder";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function fakeStream() {
  const stop = vi.fn();
  const stream = { getTracks: () => [{ stop }] } as unknown as MediaStream;
  return { stream, stop };
}

const getUserMedia = vi.fn<(options: MediaStreamConstraints) => Promise<MediaStream>>();
const closeContext = vi.fn(async () => undefined);
const addModule = vi.fn<(url: string) => Promise<void>>();
const createUrl = vi.fn(() => "blob:recorder-worklet");
const revokeUrl = vi.fn();
const openContext = vi.fn(function () {
  return {
    sampleRate: 16000,
    close: closeContext,
    audioWorklet: { addModule },
    createMediaStreamSource: () => ({ connect: vi.fn() }),
    createAnalyser: () => ({ fftSize: 0 }),
    createGain: () => ({ gain: { value: 1 }, connect: vi.fn() }),
  };
});
const port = { onmessage: null as ((event: { data: Float32Array }) => void) | null };
const openWorklet = vi.fn(function () {
  return { port, connect: (sink: object) => sink };
});

beforeEach(() => {
  vi.clearAllMocks();
  getUserMedia.mockReset();
  addModule.mockReset().mockResolvedValue(undefined);
  port.onmessage = null;
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia } });
  vi.stubGlobal("AudioContext", openContext);
  vi.stubGlobal("AudioWorkletNode", openWorklet);
  vi.stubGlobal("requestAnimationFrame", vi.fn(() => 1));
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.stubGlobal("URL", class extends URL {
    static createObjectURL = createUrl;
    static revokeObjectURL = revokeUrl;
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function mount(deviceId?: string) {
  const onComplete = vi.fn();
  return {
    ...renderHook(() => useRecorder({ maxSeconds: 15, deviceId, onComplete })),
    onComplete,
  };
}

describe("starting and stopping microphone capture", () => {
  it.each(["stop", "cancel"] as const)("abandons pending permission on %s without emitting a WAV", async (action) => {
    const pending = deferred<MediaStream>();
    const { stream, stop } = fakeStream();
    getUserMedia.mockReturnValueOnce(pending.promise);
    const { result, onComplete } = mount();
    let starting!: Promise<void>;

    act(() => { starting = result.current.start(); });
    expect(result.current.state).toBe("requesting");
    act(() => result.current[action]());
    expect(result.current.state).toBe("idle");
    await act(async () => {
      pending.resolve(stream);
      await starting;
    });

    expect(stop).toHaveBeenCalledOnce();
    expect(openContext).not.toHaveBeenCalled();
    expect(onComplete).not.toHaveBeenCalled();
    expect(result.current.state).toBe("idle");
  });

  it("stops the late stream after unmounting", async () => {
    const pending = deferred<MediaStream>();
    const { stream, stop } = fakeStream();
    getUserMedia.mockReturnValueOnce(pending.promise);
    const { result, unmount, onComplete } = mount();
    let starting!: Promise<void>;

    act(() => { starting = result.current.start(); });
    unmount();
    pending.resolve(stream);
    await starting;

    expect(stop).toHaveBeenCalledOnce();
    expect(openContext).not.toHaveBeenCalled();
    expect(onComplete).not.toHaveBeenCalled();
  });

  it("ignores a second start while permission is pending", async () => {
    const pending = deferred<MediaStream>();
    getUserMedia.mockReturnValue(pending.promise);
    const { result } = mount();

    await act(async () => {
      const first = result.current.start();
      const second = result.current.start();
      pending.resolve(fakeStream().stream);
      await Promise.all([first, second]);
    });

    expect(getUserMedia).toHaveBeenCalledOnce();
    expect(openContext).toHaveBeenCalledOnce();
    expect(result.current.state).toBe("recording");
  });

  it("cancels while the worklet loads and releases the object URL after failure", async () => {
    const pending = deferred<void>();
    const { stream, stop } = fakeStream();
    getUserMedia.mockResolvedValueOnce(stream);
    addModule.mockReturnValueOnce(pending.promise);
    const { result, onComplete } = mount();
    let starting!: Promise<void>;

    await act(async () => { starting = result.current.start(); });
    expect(addModule).toHaveBeenCalledWith("blob:recorder-worklet");
    act(() => result.current.cancel());
    expect(stop).toHaveBeenCalledOnce();
    expect(closeContext).toHaveBeenCalledOnce();
    await act(async () => {
      pending.reject(new Error("worklet load aborted"));
      await starting;
    });

    expect(revokeUrl).toHaveBeenCalledExactlyOnceWith("blob:recorder-worklet");
    expect(openWorklet).not.toHaveBeenCalled();
    expect(onComplete).not.toHaveBeenCalled();
    expect(result.current.state).toBe("idle");
    expect(result.current.error).toBeNull();
  });

  it("keeps a new recording when an older permission request resolves", async () => {
    const pending = deferred<MediaStream>();
    const stale = fakeStream();
    const current = fakeStream();
    getUserMedia.mockReturnValueOnce(pending.promise).mockResolvedValueOnce(current.stream);
    const { result, onComplete } = mount();
    let first!: Promise<void>;

    act(() => { first = result.current.start(); });
    act(() => result.current.cancel());
    await act(() => result.current.start());
    await act(async () => {
      pending.resolve(stale.stream);
      await first;
    });

    expect(stale.stop).toHaveBeenCalledOnce();
    expect(current.stop).not.toHaveBeenCalled();
    expect(result.current.state).toBe("recording");
    port.onmessage?.({ data: new Float32Array([0.1, -0.1]) });
    act(() => result.current.stop());
    expect(current.stop).toHaveBeenCalledOnce();
    expect(onComplete).toHaveBeenCalledOnce();
    expect(onComplete.mock.calls[0][0].duration).toBe(2 / 16000);
  });

  it("retries a missing saved microphone with the default device", async () => {
    getUserMedia
      .mockRejectedValueOnce(new DOMException("", "OverconstrainedError"))
      .mockResolvedValueOnce(fakeStream().stream);
    const { result } = mount("disconnected");

    await act(() => result.current.start());

    expect(getUserMedia).toHaveBeenCalledTimes(2);
    const [first, second] = getUserMedia.mock.calls.map(([constraints]) => constraints.audio);
    expect(first).toMatchObject({ deviceId: { exact: "disconnected" } });
    expect(second).toMatchObject({ echoCancellation: false, channelCount: 1 });
    expect(second).not.toHaveProperty("deviceId");
    expect(result.current.state).toBe("recording");
  });

  it("shows a fallback error when the browser provides no error message", async () => {
    getUserMedia.mockRejectedValue(new DOMException("", "OverconstrainedError"));
    const { result } = mount();

    await act(() => result.current.start());

    expect(getUserMedia).toHaveBeenCalledOnce();
    expect(result.current.state).toBe("error");
    expect(result.current.error).toBe("Could not start recording");
  });

  it("does not emit when stopped before the first audio samples arrive", async () => {
    const { stream, stop } = fakeStream();
    getUserMedia.mockResolvedValueOnce(stream);
    const { result, onComplete } = mount();

    await act(() => result.current.start());
    act(() => result.current.stop());

    expect(stop).toHaveBeenCalledOnce();
    expect(onComplete).not.toHaveBeenCalled();
    expect(result.current.state).toBe("error");
    expect(result.current.error).toBe("No microphone audio was captured.");
  });
});
