import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import { AudioLines, Globe2, Lightbulb, Mic, MicOff, Music4, Pause, Play, Radio, Search, ShieldAlert, Square, Upload, Wind, Type, Layers, X } from "lucide-react";
import { useRecorder } from "@/hooks/useRecorder";
import { useAppSettings } from "@/hooks/useAppSettings";
import { api } from "@/services/api";
import type { Mode, RecognitionResult } from "@/types/api";
import { fileToWav } from "@/lib/wav";
import { Button, Card, CardHeader, ErrorBox, Input, Progress, SegmentedControl } from "@/components/ui/primitives";
import { LiveWaveform } from "@/visualizations/canvas";
import { ResultsView } from "@/features/recognition/ResultsView";
import { cn, formatTime } from "@/lib/utils";
import { stopPreview, togglePreview, usePreviewActive } from "@/lib/preview";
import { DEMO } from "@/lib/demo";

const MODE_ICONS: Record<Mode, JSX.Element> = {
  auto: <AudioLines className="h-4 w-4" />,
  clip: <Radio className="h-4 w-4" />,
  hum: <Music4 className="h-4 w-4" />,
  whistle: <Wind className="h-4 w-4" />,
  lyrics: <Type className="h-4 w-4" />,
  combined: <Layers className="h-4 w-4" />,
};

function MicButton({ state, level, onClick, disabled }: { state: "idle" | "recording" | "busy"; level: number; onClick: () => void; disabled?: boolean }) {
  const { t } = useAppSettings();
  const rec = state === "recording";
  return (
    <div className="relative grid place-items-center" style={{ width: 200, height: 200 }}>
      {rec && (
        <>
          <span className="absolute h-36 w-36 animate-ripple rounded-full border border-brand/50" />
          <span className="absolute rounded-full bg-brand/10 transition-all duration-100" style={{ width: 144 + level * 56, height: 144 + level * 56 }} />
        </>
      )}
      <button
        onClick={onClick}
        disabled={disabled || state === "busy"}
        aria-label={rec ? t("rec.stop") : t("rec.tap")}
        className={cn(
          "relative z-10 grid h-36 w-36 place-items-center rounded-full border transition-colors duration-200 focus-visible:ring-4",
          rec ? "border-brand bg-brand" : "border-border bg-elevated hover:border-brand/60",
          state === "busy" && "opacity-60",
        )}
      >
        {rec ? <Square className="h-10 w-10 fill-brand-fg text-brand-fg" /> : <Mic className="h-12 w-12 text-brand" />}
      </button>
    </div>
  );
}

function PracticePanel() {
  const { data } = useQuery({ queryKey: ["songs", ""], queryFn: () => api.songs() });
  // Shared player: a result preview and a practice melody never play at the same time.
  const active = usePreviewActive();
  const playing = active?.startsWith("song:") ? active.slice(5) : null;
  useEffect(() => () => stopPreview(), []);
  const songs = (data?.items ?? []).filter((s) => s.index_status === "indexed").slice(0, 48);
  if (!data) return null;
  if (!songs.length)
    return (
      <Card className="p-6 text-center text-sm text-muted">
        The catalog is empty. <Link to="/library" className="text-brand hover:underline">Import the demo catalog</Link> to start.
      </Card>
    );
  const toggle = (id: string) => togglePreview(`song:${id}`, api.songAudioUrl(id));
  return (
    <Card>
      <CardHeader title="Practice with the catalog" subtitle="Listen to a melody, then hum or whistle any part of it — the chorus in the middle works too." />
      <div className="grid max-h-80 grid-cols-1 gap-1 overflow-y-auto p-3 scrollbar-thin sm:grid-cols-2 lg:grid-cols-3">
        {songs.map((s) => (
          <button key={s.id} onClick={() => toggle(s.id)} className="group flex items-center gap-3 rounded-lg px-2 py-2 text-left hover:bg-elevated/60" aria-label={`Play ${s.title}`}>
            <span
              className={cn(
                "grid h-9 w-9 shrink-0 place-items-center rounded-lg border transition-colors",
                playing === s.id ? "border-brand bg-brand text-brand-fg" : "border-border bg-elevated text-muted group-hover:border-brand/50 group-hover:text-brand",
              )}
            >
              {playing === s.id ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            </span>
            <div className="min-w-0">
              <div className="truncate text-sm font-medium">{s.title}</div>
              <div className="truncate text-xs text-muted">
                {s.artist}
                {s.has_lyrics && " · lyrics"}
              </div>
            </div>
          </button>
        ))}
      </div>
    </Card>
  );
}

function History() {
  const { data } = useQuery({ queryKey: ["history"], queryFn: () => api.history(8) });
  if (!data?.length) return null;
  return (
    <Card>
      <CardHeader title="Recent searches" subtitle="Results only — recordings are not kept unless you allowed it" />
      <ul className="p-3">
        {data.map((h) => {
          const found = !!h.top_song && h.status !== "no_match";
          return (
            <li key={h.id}>
              <Link to={`/r/${h.id}`} className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm hover:bg-elevated/60">
                <span
                  title={h.status.replace("_", " ")}
                  className={cn("h-1.5 w-1.5 shrink-0 rounded-full", h.status === "match" ? "bg-highlight" : h.status === "uncertain" ? "bg-brand" : "bg-muted/40")}
                />
                <span className={cn("min-w-0 flex-1 truncate", !found && "text-muted")}>{found ? h.top_song!.title : "No reliable match"}</span>
                <span className="font-mono text-[11px] text-muted">{(h.resolved_mode ?? h.mode).replace("_", " ")}</span>
                <span className="font-mono text-[11px] text-muted tabular">{new Date(h.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

export default function RecognizePage() {
  const { t, settings } = useAppSettings();
  const qc = useQueryClient();
  // Words leave the server only when the setting is on *and* the server is online (not MELODYAI_OFFLINE).
  const worldStatus = useQuery({ queryKey: ["world-status"], queryFn: api.worldStatus, retry: false, staleTime: 60_000 });
  const searchesOnline = settings.online_search_enabled && worldStatus.data?.online_enabled !== false;
  const keyedSources = (worldStatus.data?.sources ?? []).filter((s) => s.requires_key && s.available && s.online).map((s) => s.label.split(" (")[0]);
  const [mode, setMode] = useState<Mode>(settings.default_mode);
  const [text, setText] = useState("");
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [retain, setRetain] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [shortWarning, setShortWarning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => setMode(settings.default_mode), [settings.default_mode]);

  const m = useMutation({
    mutationFn: async (p: { audio?: Blob | null; name?: string; text?: string }) => {
      abortRef.current = new AbortController();
      return api.recognize({ mode, audio: p.audio, audioName: p.name, text: p.text, retainAudio: retain }, abortRef.current.signal);
    },
    onSuccess: (r) => {
      setResult(r);
      qc.invalidateQueries({ queryKey: ["history"] });
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    },
  });

  const onComplete = useCallback(
    (r: { blob: Blob; duration: number }) => {
      setShortWarning(r.duration < settings.min_record_seconds);
      m.mutate({ audio: r.blob, name: "recording.wav", text: mode === "combined" || mode === "lyrics" ? text : undefined });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [mode, text, settings.min_record_seconds, retain],
  );
  const rec = useRecorder({ maxSeconds: settings.max_record_seconds, deviceId: settings.microphone_id, onComplete });

  const recording = rec.state === "recording" || rec.state === "requesting";
  const busy = m.isPending;
  const uploadFile = async (f: File) => {
    setResult(null);
    m.mutate({ audio: await fileToWav(f), name: f.name.replace(/\.[^.]+$/, "") + ".wav", text: mode === "combined" ? text : undefined });
  };
  const textOnly = () => {
    setResult(null);
    m.mutate({ text });
  };
  const cancel = () => {
    if (recording) rec.cancel();
    abortRef.current?.abort();
  };

  const modes: Mode[] = ["auto", "clip", "hum", "whistle", "lyrics", "combined"];
  return (
    <div className="space-y-8">
      <section
        className={cn("card relative overflow-hidden px-4 pb-8 pt-10 text-center sm:px-10", dragging && "ring-2 ring-brand")}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) uploadFile(f);
        }}
      >
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">{t("rec.headline")}</h1>
        <p className="mx-auto mt-2 max-w-xl text-muted">{t("rec.sub")}</p>
        <div className="mt-6 flex justify-center">
          <SegmentedControl
            ariaLabel="Recognition mode"
            value={mode}
            onChange={(v) => setMode(v)}
            options={modes.map((v) => ({ value: v, label: t(`mode.${v}` as any), icon: MODE_ICONS[v], hint: t(`mode.${v}.hint` as any) }))}
          />
        </div>
        <p className="mt-2 text-xs text-muted">{t(`mode.${mode}.hint` as any)}</p>
        {mode !== "clip" && (
          <p className="mx-auto mt-3 flex max-w-xl items-start justify-center gap-1.5 rounded-xl border border-border bg-elevated/50 px-3 py-2 text-left text-xs text-fg/90">
            <Lightbulb className="mt-px h-3.5 w-3.5 shrink-0 text-highlight" />
            {t("rec.tip")}
          </p>
        )}

        <div className="mt-4 flex flex-col items-center">
          <MicButton
            state={busy ? "busy" : recording ? "recording" : "idle"}
            level={rec.level}
            // The static demo does not record: it shows the example result of the selected mode.
            onClick={() => (DEMO ? (setResult(null), m.mutate({})) : recording ? rec.stop() : (setResult(null), rec.start()))}
          />
          <div className="h-24 w-full max-w-xl">
            <AnimatePresence>
              {recording && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-2">
                  <LiveWaveform analyser={rec.analyser} active={rec.state === "recording"} height={56} />
                  <div className="flex items-center justify-center gap-4 text-sm">
                    <span className="font-mono tabular text-fg">{formatTime(rec.elapsed)}</span>
                    <span className="text-muted">/ {formatTime(settings.max_record_seconds)}</span>
                    <div className="h-1.5 w-24 overflow-hidden rounded-full bg-elevated" aria-label="Input volume">
                      <div className={cn("h-full transition-all", rec.level > 0.9 ? "bg-brand" : "bg-highlight")} style={{ width: `${rec.level * 100}%` }} />
                    </div>
                    <Button size="sm" variant="ghost" onClick={cancel}>
                      <X className="h-4 w-4" /> {t("rec.cancel")}
                    </Button>
                  </div>
                  {rec.elapsed < settings.min_record_seconds && <p className="text-xs text-muted">{t("rec.min", { s: settings.min_record_seconds })}</p>}
                </motion.div>
              )}
              {!recording && !busy && <p className="pt-2 text-sm text-muted">{t(DEMO ? "demo.tap" : "rec.tap")}</p>}
            </AnimatePresence>
          </div>
          {(rec.state === "denied" || rec.state === "error") && (
            <div className="mt-2 flex items-center gap-2 text-sm text-brand">
              <MicOff className="h-4 w-4" /> {rec.state === "denied" ? t("rec.denied") : rec.error}
            </div>
          )}
        </div>

        <div className="mx-auto mt-4 grid max-w-2xl gap-3 sm:grid-cols-[1fr_auto]">
          {(mode === "lyrics" || mode === "combined" || mode === "auto") && (
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (text.trim()) textOnly();
              }}
            >
              <Input value={text} onChange={(e) => setText(e.target.value)} placeholder={t("rec.typeLyrics")} aria-label="Lyrics text" maxLength={2000} />
              <Button type="submit" variant="secondary" disabled={!text.trim() || busy || mode === "combined"} title={mode === "combined" ? "In combined mode the text is sent with your recording" : undefined}>
                <Search className="h-4 w-4" />
                <span className="hidden sm:inline">{t("rec.searchText")}</span>
              </Button>
            </form>
          )}
          <div className={cn("flex items-center justify-center gap-2", !(mode === "lyrics" || mode === "combined" || mode === "auto") && "sm:col-span-2")}>
            <input ref={fileRef} type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm" className="hidden" onChange={(e) => e.target.files?.[0] && uploadFile(e.target.files[0])} />
            <Button variant="outline" onClick={() => (DEMO ? (setResult(null), m.mutate({})) : fileRef.current?.click())} disabled={busy || recording}>
              <Upload className="h-4 w-4" /> {t("rec.upload")}
            </Button>
          </div>
        </div>
        <p className="mt-2 text-xs text-muted">{t("rec.drop")}</p>
        {settings.external_provider_enabled && (
          <p className="mx-auto mt-3 flex max-w-xl items-center justify-center gap-1.5 text-xs text-brand">
            <ShieldAlert className="h-3.5 w-3.5" />
            {settings.provider_policy === "always" ? "Every recording" : "Recordings not found locally"} are also sent (≤12 s) to the licensed catalog provider.
          </p>
        )}
        {searchesOnline && (
          <p className="mx-auto mt-3 flex max-w-xl items-center justify-center gap-1.5 text-xs text-muted">
            <Globe2 className="h-3.5 w-3.5 shrink-0" />
            {t("rec.onlinePrivacy")}
            {keyedSources.length > 0 && ` ${t("set.online.privacyKeyed", { sources: keyedSources.join(", ") })}`}
          </p>
        )}
        {settings.retention === "keep_with_consent" && (
          <label className="mt-3 inline-flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={retain} onChange={(e) => setRetain(e.target.checked)} /> Keep this recording on this computer (consent)
          </label>
        )}
      </section>

      <div ref={resultRef} className="scroll-mt-24">
        {busy && (
          <Card className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">{t("rec.processing")}</div>
                <div className="mt-1 text-xs text-muted">
                  {[t("rec.stage.decode"), t("rec.stage.pitch"), t("rec.stage.match"), ...(searchesOnline ? [t("rec.stage.online")] : []), t("rec.stage.rank")].join(" → ")}
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={cancel}>
                {t("rec.cancel")}
              </Button>
            </div>
            <Progress indeterminate className="mt-4" />
          </Card>
        )}
        {m.isError && !(m.error as any)?.name?.includes("Abort") && <ErrorBox error={m.error} />}
        {shortWarning && result && <p className="mb-3 text-xs text-brand">{t("rec.min", { s: settings.min_record_seconds })}</p>}
        {result && !busy && DEMO && <p className="mb-3 text-xs text-muted">{t("demo.example")}</p>}
        {result && !busy && <ResultsView result={result} onAgain={() => { setResult(null); window.scrollTo({ top: 0, behavior: "smooth" }); }} />}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[2fr_1fr]">
        <PracticePanel />
        <History />
      </div>
    </div>
  );
}
