import type { HistoryItem, RecognitionResult, Song, UserSettings } from "@/types/api";
import { DEMO_UNAVAILABLE, demoUrl } from "@/lib/demo";
import { ApiError } from "@/services/api";

/** Answers API requests from the captured demo files (see `lib/demo.ts`). */
const cache = new Map<string, Promise<unknown>>();

function load<T>(rel: string): Promise<T> {
  let p = cache.get(rel);
  if (!p) {
    p = fetch(demoUrl(`api/${rel}.json`)).then((r) => {
      if (!r.ok) throw new ApiError(404, DEMO_UNAVAILABLE);
      return r.json();
    });
    cache.set(rel, p);
  }
  return p as Promise<T>;
}

// Settings changed in the demo (theme, language…) stay in this browser.
const SETTINGS_KEY = "melodyai.demo.settings";
function localSettings(): Partial<UserSettings> {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

interface QueryRow {
  algorithm: string;
  condition: string;
  target: string | null;
  rank: number | null;
}

export async function demoRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  const url = new URL(path, "http://demo.invalid");
  const rel = url.pathname.replace(/^\/api\//, "").replace(/\/$/, "");
  const q = url.searchParams;

  if (method === "GET") {
    if (rel === "settings") return { ...(await load<UserSettings>("settings")), ...localSettings() } as T;
    if (rel === "jobs" || rel === "world/songs") return [] as T;
    if (rel === "recognitions") return (await load<HistoryItem[]>("recognitions")).slice(0, Number(q.get("limit") ?? 20)) as T;
    if (rel === "songs") {
      const { items } = await load<{ items: Song[] }>("songs");
      const text = (q.get("q") ?? "").toLowerCase();
      const status = q.get("status");
      const found = items.filter(
        (s) => (!status || s.index_status === status) && (!text || `${s.title} ${s.artist} ${s.album ?? ""}`.toLowerCase().includes(text)),
      );
      return { items: found, total: found.length } as T;
    }
    const queries = rel.match(/^experiments\/([^/]+)\/queries$/);
    if (queries) {
      const rows = await load<QueryRow[]>(`experiments/${queries[1]}/queries`).catch(() => [] as QueryRow[]);
      const alg = q.get("algorithm");
      const cond = q.get("condition");
      const errors = q.get("only_errors") === "true";
      const found = rows.filter(
        (r) => (!alg || r.algorithm === alg) && (!cond || r.condition === cond) && !(errors && (r.target == null || r.rank === 1)),
      );
      return { total: found.length, items: found.slice(0, Number(q.get("limit") ?? 200)) } as T;
    }
    return load<T>(rel);
  }

  if (method === "PUT" && rel === "settings") {
    const patch = JSON.parse(String(init?.body ?? "{}")) as Partial<UserSettings>;
    const next = { ...localSettings(), ...patch };
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
    } catch {
      /* storage unavailable: the change lasts until reload */
    }
    return { ...(await load<UserSettings>("settings")), ...next } as T;
  }

  if (method === "POST" && rel === "recognize") {
    const form = init?.body as FormData;
    if (form.get("algorithm")) throw new ApiError(501, DEMO_UNAVAILABLE);
    const mode = String(form.get("mode") ?? "auto");
    const key = !form.get("audio") && form.get("text") ? "lyrics" : mode;
    const examples = await load<Record<string, string>>("examples");
    return structuredClone(await load<RecognitionResult>(`recognitions/${examples[key] ?? examples.hum}`)) as T;
  }
  if (method === "POST" && rel === "audio/analyze") return load<T>("audio/analyze");

  throw new ApiError(501, DEMO_UNAVAILABLE);
}
