import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, ExternalLink, Globe2, HelpCircle, Music2, Pause, Play, Radio, RotateCcw, Type, Waves } from "lucide-react";
import type { ProviderResult, RecognitionResult, ResultItem } from "@/types/api";
import { api } from "@/services/api";
import { useAppSettings } from "@/hooks/useAppSettings";
import { Badge, Button, Card, CardHeader, InfoTip, Stat } from "@/components/ui/primitives";
import { AlignedContours, CostMatrix, PitchContour } from "@/visualizations/canvas";
import { cn, externalSearchLinks, fixed, formatTime, hueFromString, pct } from "@/lib/utils";

export function Artwork({ id, size = 64 }: { id: string; size?: number }) {
  const h = hueFromString(id);
  return (
    <div
      aria-hidden
      className="relative shrink-0 overflow-hidden rounded-xl shadow-lg"
      style={{ width: size, height: size, background: `radial-gradient(circle at 30% 25%, hsl(${h} 80% 66%), transparent 60%), linear-gradient(135deg, hsl(${h} 70% 50%), hsl(${(h + 45) % 360} 75% 28%))` }}
    >
      <Music2 className="absolute inset-0 m-auto text-white/80" style={{ width: size * 0.38, height: size * 0.38 }} />
    </div>
  );
}

function ScoreBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-elevated">
      <div className="h-full rounded-full bg-gradient-to-r from-accent to-accent2" style={{ width: `${Math.max(2, value * 100)}%` }} />
    </div>
  );
}

function MatchKind({ item }: { item: ResultItem }) {
  const { t } = useAppSettings();
  const m = !!item.engines.melody;
  const l = !!item.engines.lyrics;
  const fp = item.engines.fingerprint;
  return (
    <span className="flex flex-wrap gap-1.5">
      {fp && (
        <Badge tone="ok">
          <Radio className="h-3 w-3" /> {t("res.recording")} {fixed(fp.score, 2)}
        </Badge>
      )}
      {m && (
        <Badge tone="accent">
          <Waves className="h-3 w-3" /> {t("res.melody")} {fixed(item.engines.melody!.score, 2)}
        </Badge>
      )}
      {l && (
        <Badge tone="info">
          <Type className="h-3 w-3" /> {t("res.lyrics")} {fixed(item.engines.lyrics!.score, 2)}
        </Badge>
      )}
    </span>
  );
}

function PlayButton({ item }: { item: ResultItem }) {
  const { t } = useAppSettings();
  const audio = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  if (!item.song.has_audio) return null;
  const toggle = () => {
    if (!audio.current) {
      audio.current = new Audio(api.songAudioUrl(item.song_id));
      audio.current.onended = () => setPlaying(false);
    }
    if (playing) {
      audio.current.pause();
      setPlaying(false);
    } else {
      if (item.match_start != null) audio.current.currentTime = Math.max(0, item.match_start - 0.5);
      audio.current.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    }
  };
  return (
    <Button variant="secondary" size="sm" onClick={toggle} aria-pressed={playing}>
      {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      {t("res.listen")}
      {item.match_start != null && <span className="text-muted">@ {formatTime(item.match_start)}</span>}
    </Button>
  );
}

const isDemo = (item: ResultItem) => (item.song.source ?? "").startsWith("MelodyAI procedural");

function EditInfo({ speed, pitch }: { speed?: number; pitch?: number }) {
  if (speed == null) return null;
  const sped = Math.abs(speed - 1) >= 0.03;
  const shifted = pitch != null && Math.abs(pitch) >= 0.4;
  if (!sped && !shifted) return <p className="text-sm text-muted">Original speed and pitch.</p>;
  return (
    <p className="text-sm text-muted">
      Your clip is{" "}
      {sped && <span className="text-fg">{speed > 1 ? `${fixed(speed, 2)}× faster (sped up)` : `${fixed(1 / speed, 2)}× slower (slowed)`}</span>}
      {sped && shifted && " and "}
      {shifted && <span className="text-fg">{pitch! > 0 ? "+" : ""}{fixed(pitch, 1)} semitones</span>} relative to the original.
    </p>
  );
}

const LINK_LABELS: Record<string, string> = { spotify: "Spotify", youtube: "YouTube", deezer: "Deezer", musicbrainz: "MusicBrainz" };

export function ExternalResults({ external, localStatus }: { external: ProviderResult[]; localStatus: string }) {
  const { t } = useAppSettings();
  if (!external?.length) return null;
  return (
    <Card className={cn(localStatus !== "match" && external.some((e) => e.status === "match") && "ring-1 ring-accent2/50")}>
      <CardHeader icon={<Globe2 className="h-4 w-4" />} title={t("res.external")}
        subtitle={t("res.externalNote", { p: external[0].provider, s: fixed(external[0].sent_seconds, 0) })} />
      <div className="space-y-3 p-5">
        {external.map((r, k) => (
          <div key={k}>
            <div className="label mb-2">{r.provider} · {r.kind === "humming" ? "humming search" : "recording fingerprint"} · {fixed(r.elapsed_s, 2)} s</div>
            {r.status !== "match" && <p className="text-sm text-muted">{r.status === "no_result" ? "No result in the provider's catalog." : r.message}</p>}
            <ol className="space-y-2">
              {r.matches.slice(0, 5).map((m, i) => (
                <li key={i} className="flex flex-wrap items-center gap-3 rounded-xl border border-border/50 px-3 py-2">
                  <Artwork id={`${m.title}${m.artists.join()}`} size={36} />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium">{m.title}</div>
                    <div className="truncate text-xs text-muted">{m.artists.join(", ")}{m.album ? ` · ${m.album}` : ""}{m.release_date ? ` · ${m.release_date.slice(0, 4)}` : ""}</div>
                  </div>
                  <span className="text-xs text-muted" title={m.score_kind}>{m.score_kind.includes("70-100") ? fixed(m.score, 0) : fixed(m.score, 2)}</span>
                  <span className="flex gap-1">
                    {Object.entries(m.links).map(([k, url]) => (
                      <a key={k} href={url} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-accent hover:bg-elevated">
                        {LINK_LABELS[k] ?? k} <ExternalLink className="h-3 w-3" />
                      </a>
                    ))}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        ))}
      </div>
    </Card>
  );
}

export function ResultsView({ result, onAgain }: { result: RecognitionResult; onAgain?: () => void }) {
  const { t } = useAppSettings();
  const top = result.results[0];
  const d = result.decision;
  const status = result.status;
  const statusUi = {
    match: { icon: CheckCircle2, tone: "ok" as const, text: t("res.match") },
    uncertain: { icon: HelpCircle, tone: "warn" as const, text: t("res.uncertain") },
    no_match: { icon: AlertTriangle, tone: "bad" as const, text: t("res.no_match") },
  }[status];
  const StatusIcon = statusUi.icon;
  const lyr = result.engines.lyrics;
  const mel = result.engines.melody;
  const showTop = top && status !== "no_match";

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="space-y-5">
      <Card className="overflow-hidden">
        <div className={cn("flex flex-wrap items-center justify-between gap-3 border-b border-border/60 px-5 py-3", status === "match" ? "bg-ok/5" : status === "uncertain" ? "bg-warn/5" : "bg-bad/5")}>
          <div className="flex items-center gap-2">
            <StatusIcon className={cn("h-5 w-5", { "text-ok": status === "match", "text-warn": status === "uncertain", "text-bad": status === "no_match" })} />
            <span className="font-medium" role="status">{statusUi.text}</span>
            {d.confidence && <Badge tone={statusUi.tone}>{d.confidence} confidence</Badge>}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted">
            <span>{result.resolved_mode} · {result.resolved_mode === "clip" ? "fingerprint" : mel?.algorithm ?? "lyrics"} · {fixed(result.elapsed_s, 2)} s</span>
            {onAgain && (
              <Button size="sm" variant="ghost" onClick={onAgain}>
                <RotateCcw className="h-3.5 w-3.5" /> {t("res.again")}
              </Button>
            )}
          </div>
        </div>

        {showTop ? (
          <div className="grid gap-6 p-5 md:grid-cols-[1fr_260px]">
            <div className="flex gap-4">
              <Artwork id={top.song_id} size={88} />
              <div className="min-w-0 space-y-2">
                <h2 className="truncate text-2xl font-semibold tracking-tight">{top.song.title}</h2>
                <p className="text-muted">
                  {top.song.artist}
                  {top.song.album ? ` · ${top.song.album}` : ""}
                  {top.song.year ? ` · ${top.song.year}` : ""}
                </p>
                <MatchKind item={top} />
                {top.engines.fingerprint && <EditInfo speed={top.engines.fingerprint.speed} pitch={top.engines.fingerprint.pitch_semitones} />}
                {top.match_start != null && top.match_end != null && (
                  <p className="text-sm text-muted">
                    {t("res.matchedAt")}: <span className="font-mono text-fg">{formatTime(top.match_start)}–{formatTime(top.match_end)}</span>
                    {top.song.duration_s ? <span> / {formatTime(top.song.duration_s)}</span> : null}
                  </p>
                )}
                {top.engines.lyrics?.matched_text && (
                  <blockquote className="border-l-2 border-accent2/60 pl-3 text-sm italic text-fg/90">“{top.engines.lyrics.matched_text}”</blockquote>
                )}
                <div className="flex flex-wrap gap-2 pt-1">
                  <PlayButton item={top} />
                  {top.song.listen_url && (
                    <a className="inline-flex items-center gap-1 text-sm text-accent hover:underline" href={top.song.listen_url} target="_blank" rel="noreferrer noopener">
                      Open <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  )}
                  {!isDemo(top) &&
                    externalSearchLinks(top.song.title, top.song.artist).map((l) => (
                      <a key={l.name} className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-muted hover:bg-elevated hover:text-fg" href={l.url} target="_blank" rel="noreferrer noopener">
                        {l.name} <ExternalLink className="h-3 w-3" />
                      </a>
                    ))}
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <div>
                <div className="label flex items-center gap-1">
                  {t("res.score")} <InfoTip content={t("res.scoreHelp")} />
                </div>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="text-3xl font-semibold tabular">{fixed(top.score, 3)}</span>
                  <span className="text-xs text-muted">margin {fixed(d.margin, 3)}</span>
                </div>
                <ScoreBar value={top.score} />
              </div>
              {d.calibrated_probability != null && (
                <div>
                  <div className="label flex items-center gap-1">
                    {t("res.calibrated")} <InfoTip content={t("res.calibratedHelp")} />
                  </div>
                  <div className="mt-1 text-lg font-semibold tabular">{pct(d.calibrated_probability, 0)}</div>
                </div>
              )}
              {d.thresholds && (
                <p className="text-[11px] leading-relaxed text-muted">
                  Accept ≥ {fixed(d.thresholds.accept, 2)} ({d.threshold_source === "calibrated" ? "tuned on synthetic dev queries" : "default"}); uncertain ≥ {fixed(d.thresholds.uncertain, 2)}.
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="p-6 text-sm text-muted">
            <p>{t("res.noMatchHelp")}</p>
            {mel?.reason && <p className="mt-2">Melody engine: {mel.reason}</p>}
            {lyr?.reason && <p className="mt-1">Lyrics engine: {lyr.reason}</p>}
          </div>
        )}
        {result.warnings?.length > 0 && (
          <div className="border-t border-border/60 px-5 py-3 text-xs text-warn">{result.warnings.join(" ")}</div>
        )}
      </Card>

      {result.external && <ExternalResults external={result.external} localStatus={status} />}

      {result.results.length > (showTop ? 1 : 0) && (
        <Card>
          <CardHeader title={showTop ? t("res.alternatives") : "Closest (below the acceptance threshold)"} subtitle="Ranked by the same relative score" />
          <ol className="divide-y divide-border/50 px-2 pb-2 pt-3">
            {result.results.slice(showTop ? 1 : 0).map((r) => (
              <li key={r.song_id} className="flex items-center gap-3 rounded-xl px-3 py-2.5 hover:bg-elevated/50">
                <span className="w-5 text-right text-xs text-muted tabular">{r.rank}</span>
                <Artwork id={r.song_id} size={36} />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{r.song.title}</div>
                  <div className="truncate text-xs text-muted">{r.song.artist}</div>
                </div>
                <div className="hidden sm:block">
                  <MatchKind item={r} />
                </div>
                <div className="w-28">
                  <div className="text-right text-xs tabular text-muted">{fixed(r.score, 3)}</div>
                  <ScoreBar value={r.score} />
                </div>
              </li>
            ))}
          </ol>
        </Card>
      )}

      {lyr?.transcript && (
        <Card className="p-5">
          <div className="label">{t("res.transcript")} (Whisper {lyr.transcript.model}{lyr.transcript.language ? `, ${lyr.transcript.language}` : ""})</div>
          <p className="mt-1 text-sm">{lyr.transcript.speech_detected ? `“${lyr.transcript.text}”` : `No words detected (${lyr.transcript.reason}).`}</p>
        </Card>
      )}

      {result.alignment && showTop && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Card>
            <CardHeader
              title={t("res.comparison")}
              subtitle={`Aligned with subsequence DTW · your tempo ×${fixed(result.alignment.tempo_scale, 2)} · transposed ${result.alignment.query_transposition_semitones > 0 ? "+" : ""}${fixed(result.alignment.query_transposition_semitones, 1)} st · distance ${fixed(result.alignment.distance, 3)}`}
            />
            <div className="p-4">
              <AlignedContours alignment={result.alignment} />
              <div className="mt-2 flex gap-4 text-xs text-muted">
                <span className="flex items-center gap-1"><span className="h-0.5 w-4 bg-accent" /> your query (warped)</span>
                <span className="flex items-center gap-1"><span className="h-1 w-4 bg-muted/70" /> reference melody</span>
              </div>
            </div>
          </Card>
          <Card>
            <CardHeader title="DTW cost matrix" subtitle="Accumulated alignment cost (darker = cheaper) and the optimal warping path" />
            <div className="p-4">
              <CostMatrix alignment={result.alignment} />
            </div>
          </Card>
        </div>
      )}

      {result.query && (
        <Card>
          <CardHeader title={t("res.quality")} subtitle={`Pitch extractor: ${result.query.pitch_track?.extractor ?? "–"} · profile: ${mel?.profile ?? "–"}`} />
          <div className="grid grid-cols-2 gap-3 p-5 sm:grid-cols-4">
            <Stat label="Voiced" value={`${fixed(Number(result.query.quality.voiced_duration_s ?? 0), 1)} s`} hint="Seconds of stable pitched sound" />
            <Stat label="Pitch confidence" value={pct(Number(result.query.quality.pitch_confidence ?? 0))} hint="Mean pYIN voicing probability over voiced frames" />
            <Stat label="Notes" value={String(result.query.quality.n_notes ?? 0)} />
            <Stat label="Est. SNR" value={`${fixed(Number(result.query.quality.snr_db ?? 0), 0)} dB`} hint="Rough estimate from loud vs quiet frames" />
          </div>
          {result.query.pitch_track && (
            <div className="px-5 pb-5">
              <PitchContour
                track={result.query.pitch_track}
                notes={result.query.notes.onsets.map((o, i) => ({ onset: o, duration: result.query!.notes.durations[i], midi: result.query!.notes.pitches[i] }))}
                height={180}
              />
            </div>
          )}
        </Card>
      )}
    </motion.div>
  );
}
