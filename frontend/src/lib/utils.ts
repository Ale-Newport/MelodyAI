import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatTime(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "–";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function pct(v: number | null | undefined, digits = 0): string {
  return v == null ? "–" : `${(v * 100).toFixed(digits)}%`;
}

export function fixed(v: number | null | undefined, digits = 3): string {
  return v == null || !isFinite(v) ? "–" : v.toFixed(digits);
}

/** Deterministic hue from a string (used for generated song artwork). */
export function hueFromString(s: string): number {
  // FNV-1a with a final avalanche step, so ids that differ in one character get distant hues.
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  h ^= h >>> 15;
  h = Math.imul(h, 2246822507);
  h ^= h >>> 13;
  return (h >>> 0) % 360;
}

const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
export function midiName(m: number): string {
  const r = Math.round(m);
  return `${NOTE_NAMES[((r % 12) + 12) % 12]}${Math.floor(r / 12) - 1}`;
}

export function externalSearchLinks(title: string, artist: string) {
  const q = encodeURIComponent(`${title} ${artist}`);
  return [
    { name: "YouTube Music", url: `https://music.youtube.com/search?q=${q}` },
    { name: "Spotify", url: `https://open.spotify.com/search/${q}` },
    { name: "Apple Music", url: `https://music.apple.com/us/search?term=${q}` },
  ];
}
