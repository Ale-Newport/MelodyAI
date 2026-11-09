export type Mode = "auto" | "clip" | "hum" | "whistle" | "lyrics" | "combined";
export type Status = "match" | "uncertain" | "no_match";

export interface SongSummary {
  id: string;
  title: string;
  artist: string;
  album?: string | null;
  year?: number | null;
  language?: string | null;
  reference_type?: string;
  duration_s?: number | null;
  listen_url?: string | null;
  license?: string | null;
  source?: string | null;
  has_audio?: boolean;
  /** Open-world songs (reference_type "online"): album art, 30 s preview and streaming links. */
  cover_url?: string | null;
  preview_url?: string | null;
  /** Deezer track id: a fresh preview is resolved by `GET /api/world/preview/{deezer_id}` (stored URLs expire). */
  deezer_id?: number | null;
  links?: Record<string, string>;
}

export interface Song extends SongSummary {
  reference_filename?: string | null;
  composition_id?: string | null;
  content_hash: string;
  index_status: "pending" | "processing" | "indexed" | "error" | "stale";
  index_error?: string | null;
  feature_version?: string | null;
  n_notes?: number | null;
  n_windows?: number | null;
  has_lyrics: boolean;
  lyrics_language?: string | null;
  extra: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
  lyrics?: { text: string; language: string; source?: string | null; license?: string | null } | null;
}

export interface EngineResultRef {
  rank: number;
  score: number;
  distance?: number;
  start_time?: number | null;
  end_time?: number | null;
  matched_text?: string;
  line_range?: [number, number];
  evidence?: number;
  speed?: number;
  pitch_semitones?: number;
}

/** Per-result evidence from the open-world lyrics search. */
export interface WorldResultRef extends EngineResultRef {
  coverage?: number;
  sources?: string[];
}

export interface ResultItem {
  rank: number;
  song_id: string;
  score: number;
  song: SongSummary;
  engines: {
    melody?: EngineResultRef;
    lyrics?: EngineResultRef;
    fingerprint?: EngineResultRef;
    world?: WorldResultRef;
    /** Open-world melody search (hummed melodies of real songs, Lakh MIDI). */
    world_melody?: EngineResultRef & { source?: string };
  };
  match_start?: number | null;
  match_end?: number | null;
}

/* ------------------------------------------------------------------ open-world lyrics search */
export type WorldSearchStatus = "ok" | "no_candidates" | "no_query" | "disabled" | "error";

/** A song proposed by an online/local source and verified against its real lyrics. */
export interface WorldCandidate {
  song_id: string; // "world:" + WorldSong.id
  title: string;
  artist: string;
  album: string | null;
  language: string | null;
  duration_s: number | null;
  cover_url: string | null;
  preview_url: string | null; // never an expired signed URL; prefer deezer_id
  deezer_id?: number | null;
  rank: number; // Deezer popularity, 0 = unknown
  links: Record<string, string>; // deezer / spotify / youtube
  score: number;
  coverage: number;
  char_similarity: number;
  matched_text: string;
  line_range: [number, number];
  sources: string[];
}

export interface WorldSourceRun {
  status: "ok" | "error" | "skipped" | "unavailable";
  n: number;
  elapsed_s: number;
  message: string;
}

/** `engines.world` of a recognition, and the body of `POST /api/world/search`. */
export interface WorldEngineResult {
  status: WorldSearchStatus;
  query_text: string;
  /** Whether online sources were asked; `offline_reason` says why not ("setting_off" | "server_offline"). */
  online: boolean;
  offline_reason?: "" | "setting_off" | "server_offline";
  elapsed_s: number;
  message: string;
  sources: Record<string, WorldSourceRun>;
  candidates: WorldCandidate[];
  decision?: Decision;
}

export interface WorldSource {
  name: string;
  label: string;
  online: boolean;
  requires_key: boolean;
  available: boolean;
}

export type WorldSyncPresetName = "quick" | "standard" | "large";

export interface WorldSyncPreset {
  name: WorldSyncPresetName;
  label: string;
  approx_songs: number;
  approx_minutes: number;
}

export interface WorldLastSync {
  preset: string;
  /** completed | cancelled | error (older servers omit it). */
  status?: string | null;
  finished_at: string | number | null; // ISO date or epoch seconds
  added: number;
  collected: number;
}

/** A popular-songs sync run by another process on the same store (e.g. `melodyai world-sync`). */
export interface WorldRunningSync {
  preset: string | null;
  started_at: number | null;
  updated_at: number | null;
  total: number;
  done: number;
}

/** `GET /api/world/status`. */
export interface WorldStatus {
  songs: number;
  /** The user setting and a server that is not offline. */
  online_enabled: boolean;
  /** False when the server runs with MELODYAI_OFFLINE (older servers omit it). */
  server_online?: boolean;
  sources: WorldSource[];
  last_sync: WorldLastSync | null;
  db_bytes: number;
  presets: WorldSyncPreset[];
  running_sync?: WorldRunningSync | null;
  /** Hummed-melody index of real songs (Lakh MIDI), built with `make lakh`. */
  melodies?: WorldMelodyStatus;
}

export interface WorldMelodyStatus {
  built: boolean;
  songs: number;
  melodies: number;
  windows?: number;
  built_at?: string | null;
  source?: string;
  loaded?: boolean;
  error?: string;
}

/** `engines.world_melody` of a recognition. */
export interface WorldMelodyEngineResult {
  status: "ok" | "insufficient_melody" | "error";
  candidates: { song_id: string; title: string; artist: string; score: number; start_time?: number | null; end_time?: number | null }[];
  decision?: { status: "match" | "uncertain" | "no_match" };
  elapsed_s?: number;
  source?: string;
  message?: string;
}

/** `GET /api/world/songs`: a WorldSong without its lyrics. */
export interface WorldSongMeta {
  id: string;
  title: string;
  artist: string;
  album: string | null;
  duration_s: number | null;
  language: string | null;
  rank: number;
  deezer_id: number | null;
  isrc: string | null;
  lrclib_id: number | null;
  cover_url: string | null;
  link: string | null;
  preview_url: string | null;
  source: string;
  added_at: number;
}

export interface PitchTrackData {
  extractor: string;
  hop_s: number;
  times: number[];
  f0_hz: (number | null)[];
  midi: (number | null)[];
  voiced_prob: number[];
  voiced: number[];
}

export interface NotesData {
  onsets: number[];
  durations: number[];
  pitches: number[];
  confidence: number[];
}

export interface Alignment {
  rate_hz: number;
  query_contour: number[];
  reference_contour: number[];
  reference_t0: number;
  reference_match: [number, number];
  path: [number, number][];
  distance: number;
  tempo_scale: number;
  query_transposition_semitones: number;
  reference_median_note: string;
  accumulated_cost: number[][] | null;
}

export interface Decision {
  status: Status;
  confidence?: "high" | "medium" | "low" | null;
  top_score?: number;
  margin?: number;
  thresholds?: { accept: number; margin: number; uncertain: number };
  threshold_source?: "default" | "calibrated";
  calibrated_probability?: number;
  reason?: string;
}

export interface Transcript {
  text: string;
  language: string | null;
  language_probability: number;
  speech_detected: boolean;
  reason: string;
  model: string;
  elapsed_s: number;
}

export interface ProviderMatch {
  title: string;
  artists: string[];
  album: string | null;
  release_date: string | null;
  score: number;
  score_kind: string;
  play_offset_s: number | null;
  duration_s: number | null;
  isrc: string | null;
  links: Record<string, string>;
}

export interface ProviderResult {
  provider: string;
  kind: "music" | "humming";
  status: "match" | "no_result" | "error" | "disabled";
  matches: ProviderMatch[];
  message: string;
  elapsed_s: number;
  sent_seconds: number;
}

export interface RecognitionResult {
  id?: string;
  external?: ProviderResult[];
  mode: Mode;
  resolved_mode: string;
  status: Status;
  message: string;
  decision: Decision;
  score_kind: "relative" | "calibrated_probability";
  results: ResultItem[];
  warnings: string[];
  elapsed_s: number;
  audio_retained?: boolean;
  fusion?: { method: string };
  query?: {
    quality: Record<string, number | boolean | null>;
    pitch_track: PitchTrackData | null;
    notes: NotesData;
    contour: number[];
    contour_rate_hz: number;
  };
  engines: {
    fingerprint?: { status: string; elapsed_s: number; candidates: { song_id: string; score: number; evidence: number }[] };
    melody?: {
      algorithm: string;
      status: string;
      reason?: string;
      reliability: number;
      elapsed_s: number;
      profile?: string;
      timings?: Record<string, number>;
      candidates: { song_id: string; score: number; distance: number }[];
    };
    lyrics?: {
      source: "text" | "asr";
      status?: string;
      reason?: string;
      query_text?: string;
      transcript?: Transcript;
      reliability?: number;
      candidates: { song_id: string; score: number; matched_text: string; coverage: number }[];
    };
    /** Present whenever there was a query text (typed or transcribed). */
    world?: WorldEngineResult;
    /** Present when a melody was analysed and the Lakh melody index is built. */
    world_melody?: WorldMelodyEngineResult;
  };
  alignment?: Alignment | null;
}

export interface HistoryItem {
  id: string;
  created_at: string;
  mode: Mode;
  resolved_mode: string | null;
  status: Status;
  top_song: SongSummary | null;
  top_score: number | null;
  audio_retained: boolean;
}

export interface Health {
  status: string;
  version: string;
  songs: number;
  indexed: number;
  with_lyrics: number;
  asr_ready: boolean;
  neural_model: string | null;
  device: string;
  ffmpeg: boolean;
  fingerprints?: number;
  providers_enabled?: boolean;
  limits: { max_upload_mb: number; max_query_seconds: number };
}

export interface ModelsInfo {
  pitch_extractors: { name: string; label: string; available: boolean }[];
  melody_algorithms: { name: string; label: string }[];
  default_algorithm: string;
  fusion_methods: string[];
  asr: {
    package_installed: boolean;
    models: { name: string; cached: boolean; installed_package: boolean }[];
    /** "auto": the best cached model (`resolved`), or nothing cached yet. */
    auto?: { name: string; installed_package: boolean; cached: boolean; resolved: string | null };
    active: string;
    downloads_allowed: boolean;
    note: string;
  };
  neural: {
    active: { checkpoint: string; model_id: string; device: string; epoch?: number; val?: Record<string, number> } | null;
    error: string | null;
    checkpoints: { id: string; source: string }[];
  };
  inference_device: string;
  providers: { name: string; label: string; configured: boolean; music: boolean; humming: boolean; host: string | null; invalid_host: boolean; max_seconds_sent: number }[];
  providers_enabled: boolean;
}

/** "auto" = best model already downloaded (large-v3-turbo > large-v3 > medium > small > base > tiny). */
export type AsrModel = "auto" | "tiny" | "base" | "small" | "medium" | "large-v3" | "large-v3-turbo";

export interface UserSettings {
  ui_language: "en" | "es";
  theme: "dark" | "light" | "system";
  default_mode: Mode;
  melody_algorithm: string;
  pitch_extractor: string;
  fusion_method: "weighted" | "rrf" | "combmnz";
  asr_enabled: boolean;
  asr_model: AsrModel;
  /** Look the transcribed words up on LRCLIB/Deezer (never audio). */
  online_search_enabled: boolean;
  asr_language: "en" | "es" | null;
  inference_device: "auto" | "cpu" | "mps" | "cuda";
  min_record_seconds: number;
  max_record_seconds: number;
  top_k: number;
  store_history: boolean;
  retention: "discard" | "keep_with_consent";
  microphone_id: string | null;
  external_provider_enabled: boolean;
  provider_policy: "fallback" | "always";
}

export interface Job {
  id: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted";
  progress: number;
  message: string | null;
  config: Record<string, unknown>;
  result: any;
  error: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface CatalogStatus {
  songs: number;
  by_status: Record<string, number>;
  with_lyrics: number;
  feature_version: string;
  melody_index: { songs: number; windows: number; window_lengths: number[]; version: number };
  lyrics_index: { songs: number };
  fingerprint_index?: { songs: number; postings: number; segments: number; bytes: number; version: string };
  neural_index: unknown;
  stats: Record<string, number>;
}

export interface RetrievalMetrics {
  top1: number;
  top3: number;
  top5: number;
  top10: number;
  mrr: number;
  n: number;
}

export interface RejectionMetrics {
  threshold: number;
  accepted_correct_rate: number | null;
  accepted_wrong_rate_in_catalog: number | null;
  false_accept_rate_ooc: number | null;
  precision_of_accepts: number;
  coverage_in_catalog: number;
  auc_correct_vs_ooc: number | null;
}

export interface AlgoMetrics {
  overall: RetrievalMetrics;
  by_condition: Record<string, RetrievalMetrics>;
  latency: { mean_ms: number; p50_ms: number; p95_ms: number };
  rejection: {
    thresholds_dev: { accept: number; uncertain: number; margin: number };
    test: RejectionMetrics;
    dev: RejectionMetrics;
    platt?: { coef: number[]; intercept: number };
    calibration_test?: {
      ece: number;
      brier: number;
      n: number;
      reliability: { bin: [number, number]; n: number; confidence: number; accuracy: number }[];
    };
  };
}

export interface EvalRunSummary {
  run_id: string;
  created_at: string;
  elapsed_s: number;
  dataset: { type: string; catalog: string; queries: number; conditions: string[] };
  algorithms: Record<string, { top1: number; top5: number; mrr: number; latency_ms: number }>;
}

export interface EvalMetrics {
  run_id: string;
  created_at: string;
  seed: number;
  elapsed_s: number;
  dataset: EvalRunSummary["dataset"] & { query_hash: string; out_of_catalog_seeds: string };
  indexing: Record<string, number>;
  algorithms: Record<string, string>;
  melody: Record<string, AlgoMetrics>;
  notes: Record<string, any>;
  pitch_extractors?: {
    algorithm: string;
    results: Record<string, { overall: RetrievalMetrics; by_condition: Record<string, RetrievalMetrics>; pitch_time: { mean_ms: number } }>;
    unavailable: string[];
  };
  lyrics?: AlgoMetrics;
  asr?: Record<string, any>;
  fusion?: Record<string, any>;
  polyphonic?: { algorithm: string; songs: number; n_queries: number; modalities: Record<string, { retrieval: RetrievalMetrics; note_f1_mean: number }> };
  figures: string[];
  env: Record<string, any>;
}

export interface TrainingRun {
  name: string;
  status: string;
  best_val_mrr?: number;
  epochs_run?: number;
  finished_at?: string;
  device?: string;
  n_params?: number;
  train_seconds?: number;
  history: { epoch: number; loss: number; lr: number; val_top1: number; val_top5: number; val_mrr: number; epoch_s: number }[];
}

export interface AnalysisResult {
  profile: string;
  extractor: string;
  quality: Record<string, number | boolean | null>;
  timings: Record<string, number>;
  waveform: { min: number[]; max: number[]; duration_s: number };
  spectrogram: { data: number[][]; hop_s: number; freqs_hz: number[]; db_range: [number, number] };
  pitch_track: PitchTrackData | null;
  notes: { onset: number; duration: number; midi: number; name: string; confidence: number }[];
  intervals: number[];
  contour: number[];
  contour_rate_hz: number;
  chroma: { data: number[][]; rate_hz: number };
  informative: boolean;
  matched?: { song_id: string; score: number; start_time: number; end_time: number };
  comparison?: { song_id: string; song?: Song; alignment: Alignment | null; reference_notes: NotesData };
}

export interface ScaleMetrics {
  run_id: string;
  kind: "scale";
  created_at: string;
  elapsed_s: number;
  env: Record<string, any>;
  fingerprint?: {
    clip_seconds: number;
    accept_threshold: number;
    sizes: Record<string, {
      overall: RetrievalMetrics;
      by_condition: Record<string, RetrievalMetrics & { accepted_correct: number }>;
      false_accept: Record<string, number>;
      auc_correct_vs_negatives: number | null;
      latency: { mean_ms: number; p50_ms: number; p95_ms: number };
      index: { songs: number; postings: number; bytes: number; audio_hours: number; mb_per_audio_hour: number; postings_per_second: number };
    }>;
  };
  melody?: {
    n_candidates: number;
    nlist: number;
    m: number;
    nprobe: number;
    sizes: Record<string, {
      two_stage: RetrievalMetrics;
      neural_ann_only: RetrievalMetrics;
      candidate_recall: number;
      candidate_recall_at?: Record<string, number>;
      latency_stage1: { mean_ms: number; p95_ms: number };
      latency_stage2: { mean_ms: number; p95_ms: number };
      auc_correct_vs_ooc: number | null;
      ann_vectors: number;
      ann_index_mb: number;
      bytes_per_window: number;
      exhaustive_note_align?: RetrievalMetrics & { latency: { mean_ms: number; p95_ms: number }; n_queries: number };
    }>;
  };
}
