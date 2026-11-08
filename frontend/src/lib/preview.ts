import { useSyncExternalStore } from "react";

/**
 * One shared audio element for every "listen" button (local reference audio in the results and the
 * practice catalog, and the 30 s Deezer previews of open-world songs): starting one stops whatever was
 * playing. Nothing ever starts on its own — playback begins only from `togglePreview`, i.e. a user click.
 *
 * A preview that cannot play (an expired or missing Deezer preview, a network error) is remembered as
 * `failed` until it is tried again, so its button can say "preview unavailable" instead of doing nothing.
 */
let audio: HTMLAudioElement | null = null;
let currentKey: string | null = null; // which preview the shared element holds
let playingKey: string | null = null; // the key that is audibly playing right now, if any
let failedKey: string | null = null; // the last preview that could not play
const listeners = new Set<() => void>();

const notify = () => listeners.forEach((l) => l());

function setPlaying(key: string | null) {
  if (playingKey === key) return;
  playingKey = key;
  notify();
}

function setFailed(key: string | null) {
  if (failedKey === key) return;
  failedKey = key;
  notify();
}

/** Plays `url` under `key` (pausing anything else), or pauses it if it is the one playing. */
export function togglePreview(key: string, url: string, startAt?: number | null) {
  if (audio && currentKey === key && failedKey !== key) {
    if (playingKey === key) {
      audio.pause();
      setPlaying(null);
      return;
    }
  } else {
    audio?.pause();
    audio = new Audio(url);
    currentKey = key;
    const el = audio;
    el.onended = () => {
      if (audio === el) setPlaying(null);
    };
    el.onerror = () => {
      if (audio !== el) return;
      setPlaying(null);
      setFailed(key);
    };
    if (startAt != null && startAt > 0) el.currentTime = startAt;
  }
  const el = audio!;
  if (failedKey === key) setFailed(null); // trying again
  setPlaying(key);
  // Older engines (and jsdom) return undefined instead of a promise.
  Promise.resolve(el.play()).catch((err: unknown) => {
    if (audio !== el) return; // replaced by another preview meanwhile
    setPlaying(null);
    if ((err as { name?: string } | null)?.name !== "AbortError") setFailed(key);
  });
}

/** Stops playback (only if `key` is the active preview, when given). */
export function stopPreview(key?: string) {
  if (!audio || (key && currentKey !== key)) return;
  audio.pause();
  audio = null;
  currentKey = null;
  failedKey = null;
  setPlaying(null);
}

function subscribe(l: () => void) {
  listeners.add(l);
  return () => listeners.delete(l);
}

const snapshot = () => playingKey;
const failedSnapshot = () => failedKey;

/** The key of the preview that is playing, or null. */
export function usePreviewActive(): string | null {
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}

/** True while the preview registered under `key` is playing. */
export function usePreviewPlaying(key: string): boolean {
  return usePreviewActive() === key;
}

/** True when the preview registered under `key` could not be played (until it is tried again). */
export function usePreviewFailed(key: string): boolean {
  return useSyncExternalStore(subscribe, failedSnapshot, failedSnapshot) === key;
}
