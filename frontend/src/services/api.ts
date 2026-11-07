import type {
  AnalysisResult,
  CatalogStatus,
  EvalMetrics,
  EvalRunSummary,
  Health,
  HistoryItem,
  Job,
  ModelsInfo,
  Mode,
  RecognitionResult,
  ScaleMetrics,
  Song,
  TrainingRun,
  UserSettings,
} from "@/types/api";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const body = await res.json();
      const d = body.detail;
      msg = typeof d === "string" ? d : d?.message ?? (Array.isArray(d) ? d.map((x: any) => x.msg).join("; ") : msg);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, msg || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export interface RecognizeParams {
  mode: Mode;
  audio?: Blob | null;
  audioName?: string;
  text?: string;
  algorithm?: string;
  language?: string;
  topK?: number;
  pitchExtractor?: string;
  fusionMethod?: string;
  retainAudio?: boolean;
}

export const api = {
  health: () => request<Health>("/api/health"),
  models: () => request<ModelsInfo>("/api/models"),
  settings: () => request<UserSettings>("/api/settings"),
  saveSettings: (patch: Partial<UserSettings>) => request<UserSettings>("/api/settings", json("PUT", patch)),

  recognize(p: RecognizeParams, signal?: AbortSignal) {
    const fd = new FormData();
    fd.set("mode", p.mode);
    if (p.audio) fd.set("audio", p.audio, p.audioName ?? "recording.wav");
    if (p.text?.trim()) fd.set("text", p.text.trim());
    if (p.algorithm) fd.set("algorithm", p.algorithm);
    if (p.language) fd.set("language", p.language);
    if (p.topK) fd.set("top_k", String(p.topK));
    if (p.pitchExtractor) fd.set("pitch_extractor", p.pitchExtractor);
    if (p.fusionMethod) fd.set("fusion_method", p.fusionMethod);
    if (p.retainAudio) fd.set("retain_audio", "true");
    return request<RecognitionResult>("/api/recognize", { method: "POST", body: fd, signal });
  },
  recognition: (id: string) => request<RecognitionResult>(`/api/recognitions/${encodeURIComponent(id)}`),
  history: (limit = 20) => request<HistoryItem[]>(`/api/recognitions?limit=${limit}`),
  deleteRecognition: (id: string) => request(`/api/recognitions/${encodeURIComponent(id)}`, { method: "DELETE" }),
  clearHistory: () => request<{ deleted: number }>("/api/recognitions", { method: "DELETE" }),

  songs: (q = "", status = "") =>
    request<{ items: Song[]; total: number }>(
      `/api/songs?limit=500${q ? `&q=${encodeURIComponent(q)}` : ""}${status ? `&status=${status}` : ""}`,
    ),
  song: (id: string) => request<Song>(`/api/songs/${encodeURIComponent(id)}`),
  addSong: (fd: FormData) => request<Song>("/api/songs", { method: "POST", body: fd }),
  updateSong: (id: string, patch: Partial<Song>) => request<Song>(`/api/songs/${encodeURIComponent(id)}`, json("PATCH", patch)),
  setLyrics: (id: string, body: { text: string; language: string; source?: string; license?: string }) =>
    request<Song>(`/api/songs/${encodeURIComponent(id)}/lyrics`, json("PUT", body)),
  deleteSong: (id: string) => request(`/api/songs/${encodeURIComponent(id)}`, { method: "DELETE" }),
  reindexSong: (id: string) => request<Song>(`/api/songs/${encodeURIComponent(id)}/reindex`, { method: "POST" }),
  songAudioUrl: (id: string) => `/api/songs/${encodeURIComponent(id)}/audio`,
  songMelody: (id: string) =>
    request<{ notes: { onsets: number[]; durations: number[]; pitches: number[] }; contour: number[]; contour_rate_hz: number; contour_t0: number }>(
      `/api/songs/${encodeURIComponent(id)}/melody`,
    ),
  catalogStatus: () => request<CatalogStatus>("/api/catalog/status"),
  duplicates: () => request<{ key: string[]; song_ids: string[] }[]>("/api/catalog/duplicates"),
  bulkImport: (fd: FormData) =>
    request<{ results: { title: string; status: string; id?: string; error?: string }[] }>("/api/catalog/import", {
      method: "POST",
      body: fd,
    }),
  reindexAll: (all: boolean) => request<Job>("/api/catalog/index", json("POST", { all })),
  importDemo: () => request<Job>("/api/catalog/demo", { method: "POST" }),

  analyze(audio: Blob, profile: string, extractor: string, songId?: string) {
    const fd = new FormData();
    fd.set("audio", audio, "recording.wav");
    fd.set("profile", profile);
    fd.set("extractor", extractor);
    if (songId) fd.set("song_id", songId);
    return request<AnalysisResult>("/api/audio/analyze", { method: "POST", body: fd });
  },

  experiments: () =>
    request<{ evaluations: EvalRunSummary[]; scale: { run_id: string; created_at: string; elapsed_s: number }[]; training: TrainingRun[] }>(
      "/api/experiments",
    ),
  scaleRun: (id: string) => request<ScaleMetrics>(`/api/experiments/${encodeURIComponent(id)}`),
  experiment: (id: string) => request<EvalMetrics>(`/api/experiments/${encodeURIComponent(id)}`),
  experimentQueries: (id: string, params: { algorithm?: string; condition?: string; only_errors?: boolean }) => {
    const q = new URLSearchParams();
    if (params.algorithm) q.set("algorithm", params.algorithm);
    if (params.condition) q.set("condition", params.condition);
    if (params.only_errors) q.set("only_errors", "true");
    q.set("limit", "500");
    return request<{ total: number; items: any[] }>(`/api/experiments/${encodeURIComponent(id)}/queries?${q}`);
  },
  figureUrl: (id: string, name: string) => `/api/experiments/${encodeURIComponent(id)}/figures/${name}`,
  evaluate: (config: "quick" | "default") => request<Job>("/api/experiments/evaluate", json("POST", { config })),
  train: (body: { config: "default" | "smoke"; epochs?: number; loss?: string; lr?: number; batch_size?: number; resume?: boolean }) =>
    request<Job>("/api/training/jobs", json("POST", body)),
  jobs: (kind?: string) => request<Job[]>(`/api/jobs${kind ? `?kind=${kind}` : ""}`),
  job: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}`),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }),
  activateNeural: (checkpoint: string) => request("/api/models/neural/activate", json("POST", { checkpoint })),
  downloadAsr: (model: string) => request<Job>("/api/models/asr/download", json("POST", { model })),
};
