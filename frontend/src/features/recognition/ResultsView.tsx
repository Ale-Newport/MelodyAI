import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, ExternalLink, Globe2, HelpCircle, Mic, Music2, Pause, Play, Radio, RotateCcw, Type, VolumeX, Waves } from "lucide-react";
import type { ProviderResult, RecognitionResult, ResultItem, SongSummary, WorldEngineResult } from "@/types/api";
import { api } from "@/services/api";
import { useAppSettings } from "@/hooks/useAppSettings";
import { Badge, Button, Card, CardHeader, InfoTip, Stat } from "@/components/ui/primitives";
import { AlignedContours, CostMatrix, PitchContour } from "@/visualizations/canvas";
import { cn, externalSearchLinks, fixed, formatTime, hueFromString, musicLinks, pct, safeUrl } from "@/lib/utils";
import { stopPreview, togglePreview, usePreviewFailed, usePreviewPlaying } from "@/lib/preview";
import type { TKey } from "@/lib/i18n";

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

/**
 * Album art when the song has one (open-world songs), else the generated artwork. `remote={false}` never loads
 * the image: covers are on Deezer's CDN, and fetching one tells Deezer which song was just recognised, which must
 * not happen while online search is off (or the server is offline).
 */
export function Cover({ url, id, size = 64, remote = true }: { url?: string | null; id: string; size?: number; remote?: boolean }) {
  const [failed, setFailed] = useState<string | null>(null);
  const src = remote ? safeUrl(url) : null;
  if (!src || failed === src) return <Artwork id={id} size={size} />;
  return (
    <img
      src={src}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      referrerPolicy="no-referrer"
      onError={() => setFailed(src)}
      className="shrink-0 rounded-xl object-cover shadow-lg"
      style={{ width: size, height: size }}
    />
  );
}

/** Songs found by the open-world lyrics search (LRCLIB / Deezer / the popular-songs index). */
const isWorldItem = (item: ResultItem) => !!item.engines.world || item.song_id.startsWith("world:") || item.song.reference_type === "online";

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
  const worldMel = item.engines.world_melody;
  const world = item.engines.world ?? (isWorldItem(item) && !worldMel ? { score: item.score } : null);
  return (
    <span className="flex flex-wrap gap-1.5">
      {worldMel && (
        <Badge tone="accent">
          <Waves className="h-3 w-3" /> {t("res.melodyOnline")} {fixed(worldMel.score, 2)}
        </Badge>
      )}
      {world && (
        <Badge tone="info">
          <Globe2 className="h-3 w-3" /> {t("res.lyricsOnline")} {fixed(world.score, 2)}
        </Badge>
      )}
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
  const key = `song:${item.song_id}`;
  const playing = usePreviewPlaying(key);
  if (!item.song.has_audio) return null;
  const toggle = () =>
    togglePreview(key, api.songAudioUrl(item.song_id), item.match_start != null ? Math.max(0, item.match_start - 0.5) : null);
  return (
    <Button variant="secondary" size="sm" onClick={toggle} aria-pressed={playing}>
      {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
      {t("res.listen")}
      {item.match_start != null && <span className="text-muted">@ {formatTime(item.match_start)}</span>}
    </Button>
  );
}

/**
 * Play/pause for a 30 s streaming preview (never autoplays; starting one stops any other). With a Deezer id the
 * server resolves a fresh preview URL (Deezer signs them for ~15 minutes, so stored ones — e.g. in the history —
 * expire); a preview that cannot play says so instead of failing silently.
 */
function PreviewButton({ url, deezerId, title, compact }: { url?: string | null; deezerId?: number | null; title: string; compact?: boolean }) {
  const { t } = useAppSettings();
  const src = deezerId ? api.worldPreviewUrl(deezerId) : safeUrl(url);
  const key = `preview:${deezerId ? `deezer:${deezerId}` : src}`;
  const playing = usePreviewPlaying(key);
  const failed = usePreviewFailed(key);
  if (!src) return null;
  const icon = playing ? <Pause className="h-4 w-4" /> : failed ? <VolumeX className="h-4 w-4" /> : <Play className="h-4 w-4" />;
  const hint = failed ? t("res.previewUnavailable") : playing ? t("res.pausePreview") : t("res.preview");
  if (compact)
    return (
      <Button variant="ghost" size="icon" className={cn("h-8 w-8", failed && "text-muted")} onClick={() => togglePreview(key, src)} aria-pressed={playing} aria-label={t("res.previewOf", { title })} title={hint}>
        {icon}
      </Button>
    );
  return (
    <Button variant="secondary" size="sm" className={cn(failed && "text-muted")} onClick={() => togglePreview(key, src)} aria-pressed={playing} title={t("res.previewOf", { title })}>
      {icon} {hint}
    </Button>
  );
}

function MusicLinks({ song, small }: { song: SongSummary; small?: boolean }) {
  return (
    <>
      {musicLinks(song.title, song.artist, song.links).map((l) => (
        <a
          key={l.key}
          href={l.url}
          target="_blank"
          rel="noreferrer noopener"
          className={cn("inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs hover:bg-elevated", small ? "text-accent" : "border border-border/60 text-fg")}
        >
          {l.name} <ExternalLink className="h-3 w-3" />
        </a>
      ))}
    </>
  );
}

/** Matched lyric lines ("line / line") as a quotation. */
function LyricQuote({ text }: { text: string }) {
  const lines = text.split(/\s+\/\s+|\n/).map((l) => l.trim()).filter(Boolean);
  return (
    <blockquote className="border-l-2 border-accent2/60 pl-3 text-sm italic text-fg/90">
      “{lines.map((l, i) => (
        <span key={i}>
          {i > 0 && <br />}
          {l}
        </span>
      ))}”
    </blockquote>
  );
}

const SOURCE_LABELS: Record<string, string> = { lrclib: "LRCLIB", deezer: "Deezer", genius: "Genius", claude: "Claude" };

function useSourceLabel() {
  const { t } = useAppSettings();
  return (name: string) => (name === "index" ? t("res.source.index") : SOURCE_LABELS[name] ?? name);
}

/** Steps of the search that are not sources proposing songs (e.g. lyric lookups): not listed as sources. */
const NOT_SOURCES = new Set(["lyrics_fetch"]);

/** Small print: which sources answered the open-world search, and how many songs each proposed. */
function WorldSources({ world }: { world: WorldEngineResult }) {
  const { t } = useAppSettings();
  const label = useSourceLabel();
  const entries = Object.entries(world.sources ?? {}).filter(([name, s]) => !NOT_SOURCES.has(name) && (s.status === "ok" || s.status === "error"));
  if (!entries.length) return null;
  return (
    <p className="border-t border-border/60 px-5 py-2 text-[11px] text-muted">
      {world.online === false ? t("res.indexOnly") : t("res.onlineSources")}:{" "}
      {entries.map(([name, s], i) => (
        <span key={name} title={s.message || undefined} className={cn(s.status === "error" && "text-warn")}>
          {i > 0 && " · "}
          {label(name)} {s.status === "ok" ? `(${s.n})` : "✗"}
        </span>
      ))}
      {" · "}
      {fixed(world.elapsed_s, 1)} s
    </p>
  );
}

function Hint({ icon, tone, children }: { icon: ReactNode; tone?: "accent" | "warn"; children: ReactNode }) {
  return (
    <p className={cn("flex items-start gap-2 text-sm", tone === "accent" ? "text-fg" : tone === "warn" ? "text-warn" : "text-muted")}>
      <span className={cn("mt-0.5 shrink-0", tone === "accent" ? "text-accent" : undefined)}>{icon}</span>
      <span>{children}</span>
    </p>
  );
}

/** Open-world candidates that are not already listed: "also found online" next to a local match. */
function AlsoFoundOnline({ world, shown, below, remote }: { world: WorldEngineResult; shown: Set<string>; below: boolean; remote: boolean }) {
  const { t } = useAppSettings();
  const items = (world.candidates ?? []).filter((c) => !shown.has(c.song_id)).slice(0, 3);
  if (!items.length) return null;
  return (
    <Card>
      <CardHeader icon={<Globe2 className="h-4 w-4" />} title={below ? t("res.closeOnline") : t("res.alsoOnline")} subtitle={below ? t("res.closeOnlineSub") : t("res.alsoOnlineSub")} />
      <ol className="space-y-2 p-4">
        {items.map((c) => (
          <li key={c.song_id} className="flex flex-wrap items-center gap-3 rounded-xl border border-border/50 px-3 py-2">
            <Cover url={c.cover_url} id={c.song_id} size={40} remote={remote} />
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-medium">{c.title}</div>
              <div className="truncate text-xs text-muted">
                {c.artist}
                {c.album ? ` · ${c.album}` : ""}
              </div>
              {c.matched_text && <div className="truncate text-xs italic text-fg/80">“{c.matched_text}”</div>}
            </div>
            <Badge tone="info" title={t("res.lyricsOnline")}>
              <Globe2 className="h-3 w-3" /> {fixed(c.score, 2)}
            </Badge>
            <PreviewButton compact url={c.preview_url} deezerId={c.deezer_id} title={c.title} />
            <span className="flex flex-wrap gap-1">
              <MusicLinks small song={{ id: c.song_id, title: c.title, artist: c.artist, links: c.links }} />
            </span>
          </li>
        ))}
      </ol>
    </Card>
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

/** Known `world.message` values of failed searches, translated (anything else is an unexpected error). */
const WORLD_ERRORS: Record<string, TKey> = {
  "timed out": "res.err.timeout",
  "open-world search is not configured": "res.err.notConfigured",
  "no result": "res.err.noResult",
};

export function ResultsView({ result, onAgain }: { result: RecognitionResult; onAgain?: () => void }) {
  const { t, settings } = useAppSettings();
  const sourceLabel = useSourceLabel();
  // Leaving the results (new search, another page) stops any preview that is playing.
  useEffect(() => () => stopPreview(), []);
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
  const world = result.engines.world;
  const showTop = top && status !== "no_match";
  const topIsWorld = !!top && (result.resolved_mode === "lyrics_online" || result.resolved_mode === "melody_online" || isWorldItem(top));
  const matchedText = (topIsWorld && top?.engines.world?.matched_text) || top?.engines.lyrics?.matched_text;

  // What the recogniser understood: Whisper's transcript, or the text that was typed.
  const tr = lyr?.transcript;
  const noSpeech = lyr?.status === "no_speech" || (!!tr && !tr.speech_detected);
  const heard =
    tr && tr.speech_detected && tr.text.trim()
      ? t("res.understood", { text: tr.text.trim() })
      : noSpeech
        ? t("res.noWordsHeard")
        : lyr?.source === "text" && lyr.query_text
          ? t("res.searched", { text: lyr.query_text })
          : null;

  // Whether *this* result searched online comes from the result itself (it may be an old one from the history):
  // `world.online`, and `world.offline_reason` for why not (older servers: the current setting decides).
  const offlineReason = world
    ? world.online
      ? ""
      : world.offline_reason || (settings.online_search_enabled ? "server_offline" : "setting_off")
    : settings.online_search_enabled
      ? ""
      : "setting_off";
  // Remote cover art only for results that were searched online, while online search is still on.
  const remoteArt = settings.online_search_enabled && world?.online !== false;
  const worldError = world?.status === "error" ? (WORLD_ERRORS[world.message] ? t(WORLD_ERRORS[world.message]) : t("res.err.unexpected")) : "";
  const hints = status === "match" ? [] : [
    noSpeech && <Hint key="sing" icon={<Mic className="h-4 w-4" />} tone="accent">{t("res.singWords")}</Hint>,
    status === "no_match" && offlineReason === "setting_off" && (
      <Hint key="off" icon={<Globe2 className="h-4 w-4" />}>
        {t("res.onlineOff")}{" "}
        <Link to="/settings" className="text-accent hover:underline">{t("res.openSettings")}</Link>
      </Hint>
    ),
    status === "no_match" && offlineReason === "server_offline" && <Hint key="server" icon={<Globe2 className="h-4 w-4" />}>{t("res.serverOffline")}</Hint>,
    status === "no_match" && world?.online && world.status === "no_candidates" && world.query_text && (
      <Hint key="none" icon={<Globe2 className="h-4 w-4" />}>{t("res.noOnlineCandidates", { text: world.query_text })}</Hint>
    ),
    worldError && (
      <Hint key="err" icon={<AlertTriangle className="h-4 w-4" />} tone="warn">
        <span title={world?.message || undefined}>{t("res.onlineError", { msg: worldError })}</span>
      </Hint>
    ),
  ].filter(Boolean);
  const showAlso = !!world?.candidates?.length && !(showTop && topIsWorld);

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="space-y-5">
      <Card className="overflow-hidden">
        <div className={cn("flex flex-wrap items-center justify-between gap-3 border-b border-border/60 px-5 py-3", status === "match" ? "bg-ok/5" : status === "uncertain" ? "bg-warn/5" : "bg-bad/5")}>
          <div className="flex items-center gap-2">
            <StatusIcon className={cn("h-5 w-5", { "text-ok": status === "match", "text-warn": status === "uncertain", "text-bad": status === "no_match" })} />
            <span className="font-medium" role="status">{statusUi.text}</span>
            {d.confidence && <Badge tone={statusUi.tone}>{t(`res.confidence.${d.confidence}` as TKey)}</Badge>}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted">
            <span>{result.resolved_mode} · {result.resolved_mode === "clip" ? "fingerprint" : result.resolved_mode === "lyrics_online" ? t("res.mode.worldLyrics") : result.resolved_mode === "melody_online" ? t("res.mode.worldMelody") : mel?.algorithm ?? "lyrics"} · {fixed(result.elapsed_s, 2)} s</span>
            {onAgain && (
              <Button size="sm" variant="ghost" onClick={onAgain}>
                <RotateCcw className="h-3.5 w-3.5" /> {t("res.again")}
              </Button>
            )}
          </div>
        </div>

        {heard && (
          <div className="flex items-start gap-2 border-b border-border/60 px-5 py-3 text-sm">
            <Mic className="mt-0.5 h-4 w-4 shrink-0 text-accent2" />
            <p>
              <span className={cn(noSpeech && "text-muted")} title={noSpeech ? tr?.reason || lyr?.reason : undefined}>{heard}</span>
              {tr?.model && (
                <span className="ml-2 text-xs text-muted">
                  Whisper {tr.model}
                  {tr.language ? ` · ${tr.language}` : ""}
                </span>
              )}
            </p>
          </div>
        )}

        {showTop ? (
          <div className="grid gap-6 p-5 md:grid-cols-[1fr_260px]">
            <div className="flex gap-4">
              <Cover url={top.song.cover_url} id={top.song_id} size={88} remote={remoteArt} />
              <div className="min-w-0 space-y-2">
                <h2 className="truncate text-2xl font-semibold tracking-tight">{top.song.title}</h2>
                <p className="text-muted">
                  {top.song.artist}
                  {top.song.album ? ` · ${top.song.album}` : ""}
                  {top.song.year ? ` · ${top.song.year}` : ""}
                </p>
                <MatchKind item={top} />
                {topIsWorld && top.engines.world?.sources?.length ? (
                  <p className="text-[11px] text-muted">{t("res.foundVia", { sources: top.engines.world.sources.map(sourceLabel).join(" + ") })}</p>
                ) : null}
                {top.engines.fingerprint && <EditInfo speed={top.engines.fingerprint.speed} pitch={top.engines.fingerprint.pitch_semitones} />}
                {top.match_start != null && top.match_end != null && (
                  <p className="text-sm text-muted">
                    {t("res.matchedAt")}: <span className="font-mono text-fg">{formatTime(top.match_start)}–{formatTime(top.match_end)}</span>
                    {top.song.duration_s ? <span> / {formatTime(top.song.duration_s)}</span> : null}
                  </p>
                )}
                {matchedText && <LyricQuote text={matchedText} />}
                <div className="flex flex-wrap gap-2 pt-1">
                  <PlayButton item={top} />
                  <PreviewButton url={top.song.preview_url} deezerId={top.song.deezer_id} title={top.song.title} />
                  {top.song.listen_url && (
                    <a className="inline-flex items-center gap-1 text-sm text-accent hover:underline" href={top.song.listen_url} target="_blank" rel="noreferrer noopener">
                      Open <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  )}
                  {topIsWorld && <MusicLinks song={top.song} />}
                  {!topIsWorld &&
                    !isDemo(top) &&
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
        {hints.length > 0 && <div className="space-y-2 border-t border-border/60 px-5 py-3">{hints}</div>}
        {world && world.status !== "disabled" && world.status !== "no_query" && <WorldSources world={world} />}
        {result.warnings?.length > 0 && (
          <div className="border-t border-border/60 px-5 py-3 text-xs text-warn">{result.warnings.join(" ")}</div>
        )}
      </Card>

      {showAlso && world && <AlsoFoundOnline world={world} shown={new Set(result.results.map((r) => r.song_id))} below={!showTop} remote={remoteArt} />}

      {result.external && <ExternalResults external={result.external} localStatus={status} />}

      {result.results.length > (showTop ? 1 : 0) && (
        <Card>
          <CardHeader title={showTop ? t("res.alternatives") : "Closest (below the acceptance threshold)"} subtitle="Ranked by the same relative score" />
          <ol className="divide-y divide-border/50 px-2 pb-2 pt-3">
            {result.results.slice(showTop ? 1 : 0).map((r) => (
              <li key={r.song_id} className="flex items-center gap-3 rounded-xl px-3 py-2.5 hover:bg-elevated/50">
                <span className="w-5 text-right text-xs text-muted tabular">{r.rank}</span>
                <Cover url={r.song.cover_url} id={r.song_id} size={36} remote={remoteArt} />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{r.song.title}</div>
                  <div className="truncate text-xs text-muted">{r.song.artist}</div>
                </div>
                <PreviewButton compact url={r.song.preview_url} deezerId={r.song.deezer_id} title={r.song.title} />
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
