import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download, Trash2 } from "lucide-react";
import { api } from "@/services/api";
import { useAppSettings } from "@/hooks/useAppSettings";
import { useJob } from "@/hooks/useJob";
import { Badge, Button, Card, CardHeader, Dialog, ErrorBox, Field, Progress, Select, Slider, Switch } from "@/components/ui/primitives";
import type { UserSettings } from "@/types/api";

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

export default function SettingsPage() {
  const { settings, update } = useAppSettings();
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
            <Badge tone="info">{models.data?.inference_device ?? "…"}</Badge>
          </Row>
        </div>
      </Card>

      <Card>
        <CardHeader title="Speech recognition (lyrics from audio)" subtitle="faster-whisper, running locally. Typed lyrics always work without it." />
        <div className="mt-2">
          <Row label="Enable speech recognition"><Switch checked={settings.asr_enabled} onCheckedChange={(v) => set({ asr_enabled: v })} label="Enable speech recognition" /></Row>
          <Row label="Whisper model" hint={asrModel ? (asrModel.cached ? "Downloaded" : "Not downloaded") : undefined}>
            <div className="flex gap-2">
              <Select value={settings.asr_model} onValueChange={(v) => set({ asr_model: v })} options={(models.data?.asr.models ?? []).map((m) => ({ value: m.name, label: `${m.name}${m.cached ? " ✓" : ""}` }))} />
              {asrModel && !asrModel.cached && (
                <Button variant="outline" disabled={!models.data?.asr.downloads_allowed} title={models.data?.asr.downloads_allowed ? "Download" : "Disabled by the server (MELODYAI_ALLOW_MODEL_DOWNLOAD)"} onClick={async () => setDlJob((await api.downloadAsr(settings.asr_model)).id)}>
                  <Download className="h-4 w-4" />
                </Button>
              )}
            </div>
            {!models.data?.asr.downloads_allowed && asrModel && !asrModel.cached && <p className="mt-1 text-xs text-muted">Run <code>make asr-model MODEL={settings.asr_model}</code> to download it.</p>}
            {job.data && <Progress className="mt-2" value={job.data.progress} indeterminate={job.data.status === "running"} />}
          </Row>
          <Row label="Speech language"><Select value={settings.asr_language ?? "auto"} onValueChange={(v) => set({ asr_language: v === "auto" ? null : (v as any) })} options={[{ value: "auto", label: "Auto-detect" }, { value: "en", label: "English" }, { value: "es", label: "Español" }]} /></Row>
        </div>
      </Card>

      <Card>
        <CardHeader title="Licensed catalog provider (optional)" subtitle="Find commercial songs that are not in your local catalog, e.g. what you hear on Spotify or TikTok, through a provider that licenses that catalog." />
        <div className="mt-2">
          {(models.data?.providers ?? []).map((p) => (
            <Row key={p.name} label={p.label} hint={p.configured ? `Configured (${p.host}) · recordings: ${p.music ? "yes" : "no"} · humming: ${p.humming ? "yes" : "no"}` : p.invalid_host ? "Invalid host in MELODYAI_ACRCLOUD_HOST" : "Not configured: set MELODYAI_ACRCLOUD_HOST, _ACCESS_KEY and _ACCESS_SECRET in .env and restart the server (see docs/SCALING.md)."}>
              <Badge tone={p.configured ? "ok" : "neutral"}>{p.configured ? "configured" : "not configured"}</Badge>
            </Row>
          ))}
          <Row label="Send recordings to the provider" hint="Privacy: when enabled, up to 12 s of each recording leave this computer and are processed by the provider under its terms. Off by default.">
            <Switch checked={settings.external_provider_enabled} onCheckedChange={(v) => set({ external_provider_enabled: v })} label="Send recordings to the provider" />
          </Row>
          <Row label="When to ask the provider" hint="Fallback: only when your local catalog has no confident match (less data shared, lower cost).">
            <Select value={settings.provider_policy} onValueChange={(v) => set({ provider_policy: v as any })} options={[{ value: "fallback", label: "Only if not found locally" }, { value: "always", label: "Always" }]} />
          </Row>
          {settings.external_provider_enabled && !models.data?.providers_enabled && (
            <p className="px-5 pb-4 text-xs text-warn">Enabled, but no provider is configured on the server, so nothing will be sent.</p>
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
