import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, Globe2, ListMusic, StopCircle, Trash2 } from "lucide-react";
import { api } from "@/services/api";
import { useAppSettings } from "@/hooks/useAppSettings";
import { FINISHED, useJob } from "@/hooks/useJob";
import { Badge, Button, Card, CardHeader, Dialog, ErrorBox, Field, Progress, Select, Slider, Switch } from "@/components/ui/primitives";
import type { AsrModel, Job, UserSettings, WorldSource, WorldSyncPresetName } from "@/types/api";
import { ApiError } from "@/services/api";
import type { TKey } from "@/lib/i18n";
import { approxDuration, toDate } from "@/lib/utils";

type T = (key: TKey, vars?: Record<string, string | number>) => string;

/** Brand names of the online sources (the same in every language); what each does is in the tooltip. */
const SOURCE_NAMES: Record<string, string> = { lrclib: "LRCLIB", deezer: "Deezer", genius: "Genius", claude: "Claude" };
const JOB_STATUSES = ["queued", "running", "completed", "failed", "cancelled", "interrupted"];

const presetName = (t: T, name: string | null | undefined) =>
  name === "quick" || name === "standard" || name === "large" ? t(`set.popular.${name}` as TKey) : name ?? "";

/** The sync job's (English) progress messages, in the UI language; unknown messages are shown as they are. */
export function syncMessage(t: T, msg: string | null | undefined): string {
  if (!msg) return "";
  let m = msg.match(/^Lyrics (\d+)\/(\d+) · \+(\d+) · ETA (.+)$/);
  if (m) return t("set.popular.msg.lyrics", { done: m[1], total: m[2], added: m[3], eta: m[4] });
  if ((m = msg.match(/^Collecting songs \((\w+)\)$/))) return t("set.popular.msg.collectingPreset", { preset: presetName(t, m[1]) });
  if ((m = msg.match(/^Collecting: .*\((\d+) songs\)$/))) return t("set.popular.msg.collected", { n: m[1] });
  if (msg.startsWith("Collecting: ")) return t("set.popular.msg.collecting");
  if ((m = msg.match(/^LRCLIB asked to slow down: waiting (.+)$/))) return t("set.popular.msg.slowDown", { wait: m[1] });
  if ((m = msg.match(/^\w+: \+(\d+) songs \((\d+) looked up, (\d+) without lyrics\) in (.+)$/)))
    return t("set.popular.msg.finished", { added: m[1], looked: m[2], missing: m[3], time: m[4] });
  if (msg === "cancellation requested") return t("set.popular.msg.cancelling");
  return msg;
}

const jobStatus = (t: T, status: string) => (JOB_STATUSES.includes(status) ? t(`job.status.${status}` as TKey) : status);

const ASR_MODELS: AsrModel[] = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"];
/** What "auto" resolves to on the server: the best model already in the cache. */
const ASR_PREFERENCE: AsrModel[] = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"];

function Row({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 border-t border-border/40 px-5 py-4 first:border-t-0 sm:flex-row sm:items-center sm:justify-between">
      <div className="max-w-md">
        <div className="text-sm font-medium">{label}</div>
        {hint && <div className="text-xs text-muted">{hint}</div>}
      </div>
      <div className="sm:w-72">{children}</div>
    </div>
  );
}

function SyncJobCard({ job, onCancel }: { job: Job; onCancel: () => void }) {
  const { t } = useAppSettings();
  const done = FINISHED.includes(job.status);
  const r = (job.result ?? null) as { added?: number; collected?: number } | null;
  return (
    <div className="mt-3 rounded-xl border border-border/60 bg-elevated/40 p-3" aria-live="polite">
      <div className="mb-2 flex items-center justify-between gap-3 text-sm">
        <span className="flex min-w-0 items-center gap-2">
          <Badge tone={job.status === "completed" ? "highlight" : job.status === "failed" ? "brand" : "neutral"}>{jobStatus(t, job.status)}</Badge>
          <span className="truncate text-xs text-muted" title={job.message ?? undefined}>{syncMessage(t, job.message)}</span>
        </span>
        {!done && (
          <Button size="sm" variant="danger" onClick={onCancel}>
            <StopCircle className="h-4 w-4" /> {t("common.cancel")}
          </Button>
        )}
      </div>
      <Progress value={job.progress} />
      {job.status === "completed" && r && <p className="mt-2 text-xs text-highlight">{t("set.popular.done", { added: r.added ?? 0, collected: r.collected ?? 0 })}</p>}
      {job.error && <p className="mt-2 text-xs text-brand">{job.error}</p>}
    </div>
  );
}

/** Open-world song search: the privacy-relevant toggle, its sources and the popular-songs sync. */
function OnlineSearchCard({ onChange }: { onChange: (patch: Partial<UserSettings>) => Promise<void> }) {
  const { t, settings } = useAppSettings();
  const qc = useQueryClient();
  const lang = settings.ui_language;
  // A sync started earlier (or in another tab) keeps running on the server: pick it up again.
  const jobs = useQuery({ queryKey: ["jobs", "world_sync"], queryFn: () => api.jobs("world_sync"), retry: false });
  const [jobId, setJobId] = useState<string | null>(null);
  const activeId = jobId ?? jobs.data?.find((j) => !FINISHED.includes(j.status))?.id ?? null;
  const job = useJob(activeId, () => qc.invalidateQueries({ queryKey: ["world-status"] }));
  const running = !!job.data && !FINISHED.includes(job.data.status);
  const status = useQuery({ queryKey: ["world-status"], queryFn: api.worldStatus, retry: false, refetchInterval: running ? 10000 : false });
  const start = useMutation({ mutationFn: (preset: WorldSyncPresetName) => api.startWorldSync(preset), onSuccess: (j) => setJobId(j.id) });
  const presetLabel = (name: string) => presetName(t, name);
  const ws = status.data;
  // Online sources are only "active" when this user's setting and the server both allow online search.
  const onlineOff = !settings.online_search_enabled || ws?.online_enabled === false;
  const sourceLabel = (s: WorldSource) => (s.name === "index" ? t("set.source.index") : SOURCE_NAMES[s.name] ?? s.label);
  const sourceHint = (s: WorldSource) => (["index", ...Object.keys(SOURCE_NAMES)].includes(s.name) ? t(`set.source.hint.${s.name}` as TKey) : s.label);
  const sourceOn = (s: WorldSource) => s.available && !(s.online && onlineOff);
  const sourceState = (s: WorldSource) =>
    !s.available
      ? s.requires_key
        ? t("set.source.needsKey")
        : t("set.source.unavailable")
      : !s.online
        ? t("set.source.local")
        : onlineOff
          ? t("set.source.off")
          : t("set.source.active");
  const keyed = (ws?.sources ?? []).filter((s) => s.requires_key && s.online && s.available).map(sourceLabel);
  const last = ws?.last_sync;
  const lastDate = toDate(last?.finished_at);
  const lastStatus = last?.status && last.status !== "completed" && ["cancelled", "error", "interrupted"].includes(last.status) ? last.status : null;
  const elsewhere = ws?.running_sync ?? null;
  const startError = start.error instanceof ApiError && (start.error.status === 409 || start.error.status === 503)
    ? t(start.error.status === 409 ? "set.popular.err.running" : "set.popular.err.offline")
    : null;

  return (
    <Card>
      <CardHeader icon={<Globe2 className="h-4 w-4" />} title={t("set.online.title")} subtitle={t("set.online.subtitle")} />
      <div className="mt-2">
        <Row label={t("set.online.toggle")} hint={t("set.online.privacy") + (keyed.length ? ` ${t("set.online.privacyKeyed", { sources: keyed.join(", ") })}` : "")}>
          <Switch
            checked={settings.online_search_enabled}
            onCheckedChange={(v) => onChange({ online_search_enabled: v }).then(() => qc.invalidateQueries({ queryKey: ["world-status"] }))}
            label={t("set.online.toggle")}
          />
        </Row>
        {settings.online_search_enabled && ws && !ws.online_enabled && <p className="px-5 pb-3 text-xs text-brand">{t("set.online.serverOff")}</p>}
      </div>

      <div className="border-t border-border/40 px-5 pb-1 pt-4">
        <h4 className="flex items-center gap-2 text-sm font-semibold">
          <ListMusic className="h-4 w-4 text-brand" /> {t("set.popular.title")}
        </h4>
        <p className="mt-0.5 text-xs text-muted">{t("set.popular.subtitle")}</p>
      </div>
      {status.isError ? (
        <p className="px-5 pb-5 pt-2 text-sm text-muted">{t("set.popular.unavailable")}</p>
      ) : (
        <div>
          <Row label={t("set.popular.indexed")} hint={ws ? t("set.popular.size", { mb: (ws.db_bytes / 1e6).toLocaleString(lang, { maximumFractionDigits: 1 }) }) : undefined}>
            <span className="text-lg font-semibold tabular">{ws ? ws.songs.toLocaleString(lang) : "…"}</span>
          </Row>
          {ws?.melodies && (
            <Row
              label={t("set.melodies.label")}
              hint={ws.melodies.built ? t("set.melodies.hint", { melodies: ws.melodies.melodies.toLocaleString(lang) }) : undefined}
            >
              {ws.melodies.built ? (
                <span className="text-lg font-semibold tabular">{ws.melodies.songs.toLocaleString(lang)}</span>
              ) : (
                <span className="text-sm text-muted">{t("set.melodies.missing")}</span>
              )}
            </Row>
          )}
          <Row label={t("set.popular.lastSync")}>
            <span className="flex flex-wrap items-center justify-end gap-2 text-sm text-muted">
              {!ws
                ? "…"
                : last
                  ? t("set.popular.lastSyncValue", { preset: presetLabel(last.preset), date: lastDate ? lastDate.toLocaleString(lang) : "–", added: (last.added ?? 0).toLocaleString(lang) })
                  : t("set.popular.never")}
              {lastStatus && <Badge tone="brand">{t(`set.popular.lastStatus.${lastStatus}` as TKey)}</Badge>}
            </span>
          </Row>
          {ws && ws.sources.length > 0 && (
            <Row label={t("set.online.sources")}>
              <div className="flex flex-wrap gap-1.5">
                {ws.sources.map((s) => (
                  <Badge key={s.name} tone={sourceOn(s) ? "strong" : "neutral"} title={sourceHint(s)}>
                    {sourceLabel(s)} · {sourceState(s)}
                  </Badge>
                ))}
              </div>
            </Row>
          )}
          {ws && ws.presets.length > 0 && (
            <div className="border-t border-border/40 px-5 py-4">
              <div className="grid gap-2 sm:grid-cols-3">
                {ws.presets.map((p) => (
                  <Button
                    key={p.name}
                    variant={p.name === "quick" ? "primary" : "outline"}
                    className="h-auto flex-col items-start gap-0.5 whitespace-normal py-2 text-left"
                    disabled={running || start.isPending || !!elsewhere}
                    loading={start.isPending && start.variables === p.name}
                    onClick={() => start.mutate(p.name)}
                  >
                    <span>{presetLabel(p.name)}</span>
                    <span className="text-xs font-normal opacity-80">
                      {t("set.popular.presetMeta", { songs: p.approx_songs.toLocaleString(lang), time: approxDuration(p.approx_minutes) })}
                    </span>
                  </Button>
                ))}
              </div>
              <p className="mt-2 text-xs text-muted">{t("set.popular.note")}</p>
              {elsewhere && !running && (
                <p className="mt-2 text-xs text-brand" role="status">
                  {t("set.popular.runningElsewhere", { done: elsewhere.done.toLocaleString(lang), total: elsewhere.total.toLocaleString(lang) })}
                </p>
              )}
              {start.isError && <div className="mt-3">{startError ? <p role="alert" className="text-sm text-brand">{startError}</p> : <ErrorBox error={start.error} />}</div>}
              {job.data && (
                <SyncJobCard job={job.data} onCancel={() => api.cancelJob(job.data!.id).then(() => qc.invalidateQueries({ queryKey: ["job", job.data!.id] }))} />
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

export default function SettingsPage() {
  const { settings, update, t } = useAppSettings();
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const st = useQuery({ queryKey: ["catalog-status"], queryFn: api.catalogStatus });
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [clearing, setClearing] = useState(false);
  const [dlJob, setDlJob] = useState<string | null>(null);
  const job = useJob(dlJob, () => qc.invalidateQueries({ queryKey: ["models"] }));
  const [err, setErr] = useState<unknown>(null);
  const set = async (patch: Partial<UserSettings>) => {
    try {
      setErr(null);
      await update(patch);
      qc.invalidateQueries({ queryKey: ["health"] });
    } catch (e) {
      setErr(e);
    }
  };
  const clear = useMutation({ mutationFn: api.clearHistory, onSuccess: () => { setClearing(false); qc.invalidateQueries({ queryKey: ["history"] }); } });
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices?.().then((d) => setMics(d.filter((x) => x.kind === "audioinput"))).catch(() => undefined);
  }, []);
  const asrModel = models.data?.asr.models.find((m) => m.name === settings.asr_model);
  // "auto" with nothing cached: offer the recommended model (the server maps "auto" to it), else explain how.
  const autoMissing = settings.asr_model === "auto" && !!models.data && models.data.asr.auto?.cached === false;
  const missingModel = asrModel && !asrModel.cached ? settings.asr_model : autoMissing ? ASR_PREFERENCE[0] : null;
  const serverAsr = models.data?.asr.models ?? [];
  const isCached = (name: string) => serverAsr.some((m) => m.name === name && m.cached);
  const bestCached = ASR_PREFERENCE.find(isCached);
  const asrNames = [...ASR_MODELS, ...serverAsr.map((m) => m.name).filter((n) => !ASR_MODELS.includes(n as AsrModel))];
  const asrHint =
    settings.asr_model === "auto"
      ? bestCached
        ? t("set.asr.autoHint", { m: bestCached })
        : models.data
          ? t("set.asr.noneCached")
          : undefined
      : asrModel
        ? asrModel.cached
          ? "Downloaded"
          : "Not downloaded"
        : undefined;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-sm text-muted">Stored locally in the catalog folder (microphone choice stays in this browser).</p>
      </div>
      {err != null && <ErrorBox error={err} />}

      <Card>
        <CardHeader title="Interface" />
        <div className="mt-2">
          <Row label="Language"><Select value={settings.ui_language} onValueChange={(v) => set({ ui_language: v as any })} options={[{ value: "en", label: "English" }, { value: "es", label: "Español" }]} /></Row>
          <Row label="Theme"><Select value={settings.theme} onValueChange={(v) => set({ theme: v as any })} options={[{ value: "dark", label: "Dark" }, { value: "light", label: "Light" }, { value: "system", label: "System" }]} /></Row>
          <Row label="Default mode"><Select value={settings.default_mode} onValueChange={(v) => set({ default_mode: v as any })} options={["auto", "clip", "hum", "whistle", "lyrics", "combined"].map((m) => ({ value: m, label: m === "clip" ? "recording (clip)" : m }))} /></Row>
        </div>
      </Card>

      <Card>
        <CardHeader title="Recording" />
        <div className="mt-2">
          <Row label="Microphone" hint={mics.some((m) => m.label) ? undefined : "Device names appear after you grant microphone permission once."}>
            <Select value={settings.microphone_id ?? "default"} onValueChange={(v) => set({ microphone_id: v === "default" ? null : v })} options={[{ value: "default", label: "System default" }, ...mics.filter((m) => m.deviceId && m.deviceId !== "default").map((m, i) => ({ value: m.deviceId, label: m.label || `Microphone ${i + 1}` }))]} />
          </Row>
          <Row label={`Recording length: ${settings.min_record_seconds}–${settings.max_record_seconds} s`} hint="Minimum recommended and automatic stop">
            <Slider min={1} max={60} step={1} value={[settings.min_record_seconds, settings.max_record_seconds]} onValueChange={([a, b]) => set({ min_record_seconds: a, max_record_seconds: Math.max(b, 3) })} label="Recording length" />
          </Row>
        </div>
      </Card>

      <Card>
        <CardHeader title="Recognition" />
        <div className="mt-2">
          <Row label="Melody algorithm" hint="See the Research Lab for measured accuracy and latency of each one">
            <Select value={settings.melody_algorithm} onValueChange={(v) => set({ melody_algorithm: v })} options={(models.data?.melody_algorithms ?? []).map((a) => ({ value: a.name, label: a.label }))} />
          </Row>
          <Row label="Pitch extractor" hint="pYIN is the dependency-free default. In the synthetic benchmark CREPE was more robust to noisy humming at ~2× the cost (see docs/RESULTS.md)."><Select value={settings.pitch_extractor} onValueChange={(v) => set({ pitch_extractor: v })} options={(models.data?.pitch_extractors ?? []).map((e) => ({ value: e.name, label: e.label, disabled: !e.available }))} /></Row>
          <Row label="Fusion strategy" hint="How melody and lyrics evidence are combined"><Select value={settings.fusion_method} onValueChange={(v) => set({ fusion_method: v as any })} options={[{ value: "weighted", label: "Weighted (reliability-aware)" }, { value: "rrf", label: "Reciprocal Rank Fusion" }, { value: "combmnz", label: "CombMNZ" }]} /></Row>
          <Row label={`Results shown: ${settings.top_k}`}><Slider min={1} max={20} step={1} value={[settings.top_k]} onValueChange={([v]) => set({ top_k: v })} label="Number of results" /></Row>
          <Row label="Inference backend" hint={`Detected PyTorch device: ${models.data?.inference_device ?? "…"}. Whisper always runs on CPU/CUDA (no Apple MPS support in CTranslate2). Set MELODYAI_DEVICE to force a device.`}>
            <Badge tone="strong">{models.data?.inference_device ?? "…"}</Badge>
          </Row>
        </div>
      </Card>

      <Card>
        <CardHeader title="Speech recognition (lyrics from audio)" subtitle="faster-whisper, running locally. Typed lyrics always work without it." />
        <div className="mt-2">
          <Row label="Enable speech recognition"><Switch checked={settings.asr_enabled} onCheckedChange={(v) => set({ asr_enabled: v })} label="Enable speech recognition" /></Row>
          <Row label="Whisper model" hint={asrHint}>
            <div className="flex gap-2">
              <Select
                value={settings.asr_model}
                onValueChange={(v) => set({ asr_model: v as AsrModel })}
                ariaLabel="Whisper model"
                options={[
                  { value: "auto", label: `${t("set.asr.auto")}${bestCached ? ` → ${bestCached}` : ""}` },
                  ...asrNames.map((n) => ({ value: n, label: `${n}${isCached(n) ? " ✓" : ""}` })),
                ]}
              />
              {missingModel && (
                <Button variant="outline" disabled={!models.data?.asr.downloads_allowed} aria-label={t("set.asr.download", { m: missingModel })} title={models.data?.asr.downloads_allowed ? t("set.asr.download", { m: missingModel }) : "Disabled by the server (MELODYAI_ALLOW_MODEL_DOWNLOAD)"} onClick={async () => setDlJob((await api.downloadAsr(settings.asr_model)).id)}>
                  <Download className="h-4 w-4" />
                </Button>
              )}
            </div>
            {!models.data?.asr.downloads_allowed && missingModel && <p className="mt-1 text-xs text-muted">{t("set.asr.makeHint", { m: missingModel })}</p>}
            {job.data && <Progress className="mt-2" value={job.data.progress} indeterminate={job.data.status === "running"} />}
          </Row>
          <Row label="Speech language"><Select value={settings.asr_language ?? "auto"} onValueChange={(v) => set({ asr_language: v === "auto" ? null : (v as any) })} options={[{ value: "auto", label: "Auto-detect" }, { value: "en", label: "English" }, { value: "es", label: "Español" }]} /></Row>
        </div>
      </Card>

      <OnlineSearchCard onChange={set} />

      <Card>
        <CardHeader title="Licensed catalog provider (optional)" subtitle="Find commercial songs that are not in your local catalog, e.g. what you hear on Spotify or TikTok, through a provider that licenses that catalog." />
        <div className="mt-2">
          {(models.data?.providers ?? []).map((p) => (
            <Row key={p.name} label={p.label} hint={p.configured ? `Configured (${p.host}) · recordings: ${p.music ? "yes" : "no"} · humming: ${p.humming ? "yes" : "no"}` : p.invalid_host ? "Invalid host in MELODYAI_ACRCLOUD_HOST" : "Not configured: set MELODYAI_ACRCLOUD_HOST, _ACCESS_KEY and _ACCESS_SECRET in .env and restart the server (see docs/SCALING.md)."}>
              <Badge tone={p.configured ? "strong" : "neutral"}>{p.configured ? "configured" : "not configured"}</Badge>
            </Row>
          ))}
          <Row label="Send recordings to the provider" hint="Privacy: when enabled, up to 12 s of each recording leave this computer and are processed by the provider under its terms. Off by default.">
            <Switch checked={settings.external_provider_enabled} onCheckedChange={(v) => set({ external_provider_enabled: v })} label="Send recordings to the provider" />
          </Row>
          <Row label="When to ask the provider" hint="Fallback: only when your local catalog has no confident match (less data shared, lower cost).">
            <Select value={settings.provider_policy} onValueChange={(v) => set({ provider_policy: v as any })} options={[{ value: "fallback", label: "Only if not found locally" }, { value: "always", label: "Always" }]} />
          </Row>
          {settings.external_provider_enabled && !models.data?.providers_enabled && (
            <p className="px-5 pb-4 text-xs text-brand">Enabled, but no provider is configured on the server, so nothing will be sent.</p>
          )}
        </div>
      </Card>

      <Card>
        <CardHeader title="Privacy & data" />
        <div className="mt-2">
          <Row label="Store recognition history" hint="Only results and derived pitch data, never audio unless allowed below"><Switch checked={settings.store_history} onCheckedChange={(v) => set({ store_history: v })} label="Store history" /></Row>
          <Row label="Recording retention" hint="'Keep with consent' shows a per-recording checkbox; nothing is kept by default">
            <Select value={settings.retention} onValueChange={(v) => set({ retention: v as any })} options={[{ value: "discard", label: "Never keep recordings" }, { value: "keep_with_consent", label: "Keep with explicit consent" }]} />
          </Row>
          <Row label="Delete history"><Button variant="danger" onClick={() => setClearing(true)}><Trash2 className="h-4 w-4" /> Delete history & kept recordings</Button></Row>
          <Row label="Catalog location" hint="Set MELODYAI_STORE_DIR before starting the server to change it">
            <span className="font-mono text-xs text-muted">{st.data ? `${st.data.songs} songs · features ${st.data.feature_version}` : "…"}</span>
          </Row>
          <Row label="Server limits"><span className="text-xs text-muted">{health.data ? `${health.data.limits.max_upload_mb} MB / ${health.data.limits.max_query_seconds} s per query · ffmpeg ${health.data.ffmpeg ? "available" : "missing"}` : "…"}</span></Row>
        </div>
      </Card>

      <Dialog open={clearing} onOpenChange={setClearing} title="Delete recognition history?" description="All stored results and any recordings you chose to keep will be permanently deleted.">
        {clear.isError && <ErrorBox error={clear.error} />}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setClearing(false)}>Cancel</Button>
          <Button variant="danger" loading={clear.isPending} onClick={() => clear.mutate()}>Delete</Button>
        </div>
      </Dialog>
      <Field label="" hint={`MelodyAI ${health.data?.version ?? ""}`}><span /></Field>
    </div>
  );
}
