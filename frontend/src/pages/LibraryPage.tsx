import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Copy, Database, FileMusic, Pencil, Play, Pause, Plus, RefreshCw, Trash2, Upload, Wand2 } from "lucide-react";
import { api } from "@/services/api";
import type { Song } from "@/types/api";
import { useAppSettings } from "@/hooks/useAppSettings";
import { useJob } from "@/hooks/useJob";
import { Badge, Button, Card, Dialog, ErrorBox, Field, Input, Progress, Select, Skeleton, Stat, Textarea } from "@/components/ui/primitives";
import { Artwork } from "@/features/recognition/ResultsView";
import { formatTime } from "@/lib/utils";

const STATUS_TONE = { indexed: "ok", pending: "neutral", processing: "info", error: "bad", stale: "warn" } as const;
const REF_TYPES = ["midi", "monophonic_audio", "vocal_audio", "instrumental_audio", "polyphonic_audio"];

function AddSongDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const qc = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [lyricsFile, setLyricsFile] = useState<File | null>(null);
  const [f, setF] = useState({ title: "", artist: "", album: "", year: "", language: "", reference_type: "", license: "", source: "", listen_url: "", lyrics_text: "", lyrics_language: "en" });
  const m = useMutation({
    mutationFn: () => {
      const fd = new FormData();
      fd.set("reference", file!);
      Object.entries(f).forEach(([k, v]) => v && fd.set(k, v));
      if (lyricsFile) fd.set("lyrics_file", lyricsFile);
      return api.addSong(fd);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["songs"] });
      qc.invalidateQueries({ queryKey: ["catalog-status"] });
      onOpenChange(false);
      setFile(null);
    },
  });
  const set = (k: keyof typeof f) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => setF({ ...f, [k]: e.target.value });
  const isMidi = file && /\.midi?$/i.test(file.name);
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Add a song" description="MIDI melodies or audio references. Only import material you are allowed to use." wide>
      <form
        className="grid gap-4 sm:grid-cols-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (file && f.title) m.mutate();
        }}
      >
        <div className="sm:col-span-2">
          <Field label="Reference file" hint="MIDI (.mid) or audio (WAV, MP3, FLAC, OGG, M4A). Audio references are transcribed with pYIN; polyphonic audio uses Basic Pitch + skyline (experimental).">
            <Input type="file" accept=".mid,.midi,audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} required />
          </Field>
        </div>
        <Field label="Title" htmlFor="title"><Input id="title" value={f.title} onChange={set("title")} required maxLength={300} /></Field>
        <Field label="Artist" htmlFor="artist"><Input id="artist" value={f.artist} onChange={set("artist")} maxLength={300} /></Field>
        <Field label="Album" htmlFor="album"><Input id="album" value={f.album} onChange={set("album")} /></Field>
        <Field label="Year" htmlFor="year"><Input id="year" type="number" value={f.year} onChange={set("year")} /></Field>
        <Field label="Reference type" hint={isMidi ? "MIDI files are always 'midi'" : "What kind of audio is this?"}>
          <Select
            value={isMidi ? "midi" : f.reference_type || "monophonic_audio"}
            onValueChange={(v) => setF({ ...f, reference_type: v })}
            options={REF_TYPES.map((r) => ({ value: r, label: r.replace("_", " "), disabled: !!isMidi !== (r === "midi") }))}
          />
        </Field>
        <Field label="Language" htmlFor="lang"><Input id="lang" value={f.language} onChange={set("language")} placeholder="en, es…" /></Field>
        <Field label="License" htmlFor="license"><Input id="license" value={f.license} onChange={set("license")} placeholder="e.g. CC-BY-4.0, own work" /></Field>
        <Field label="Source" htmlFor="source"><Input id="source" value={f.source} onChange={set("source")} placeholder="Where does this file come from?" /></Field>
        <div className="sm:col-span-2">
          <Field label="Listen link (optional)" htmlFor="url"><Input id="url" value={f.listen_url} onChange={set("listen_url")} placeholder="https://…" /></Field>
        </div>
        <div className="sm:col-span-2">
          <Field label="Lyrics (optional)" hint="Paste authorised lyrics, or attach a TXT / JSON file.">
            <Textarea rows={4} value={f.lyrics_text} onChange={set("lyrics_text")} />
            <div className="mt-2 flex gap-2">
              <Input type="file" accept=".txt,.json" onChange={(e) => setLyricsFile(e.target.files?.[0] ?? null)} />
              <Input className="w-24" value={f.lyrics_language} onChange={set("lyrics_language")} aria-label="Lyrics language" />
            </div>
          </Field>
        </div>
        {m.isError && <div className="sm:col-span-2"><ErrorBox error={m.error} /></div>}
        <div className="flex justify-end gap-2 sm:col-span-2">
          <Button type="button" variant="ghost" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button type="submit" variant="primary" loading={m.isPending} disabled={!file || !f.title}>Import & index</Button>
        </div>
      </form>
    </Dialog>
  );
}

function BulkDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const qc = useQueryClient();
  const [manifest, setManifest] = useState<File | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const m = useMutation({
    mutationFn: () => {
      const fd = new FormData();
      fd.set("manifest", manifest!);
      Array.from(files ?? []).forEach((f) => fd.append("files", f));
      return api.bulkImport(fd);
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["songs"] }),
  });
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Bulk import" description="A CSV or JSON manifest plus the files it references." wide>
      <div className="space-y-4">
        <pre className="overflow-x-auto rounded-xl bg-elevated/60 p-3 text-xs text-muted">title,artist,year,language,license,source,reference_type,reference_file,lyrics_file{"\n"}My Tune,Me,2024,en,own work,home studio,midi,my_tune.mid,my_tune.txt</pre>
        <Field label="Manifest (CSV / JSON)"><Input type="file" accept=".csv,.json" onChange={(e) => setManifest(e.target.files?.[0] ?? null)} /></Field>
        <Field label="Files" hint="Select all referenced MIDI/audio/lyrics files (max 200)"><Input type="file" multiple onChange={(e) => setFiles(e.target.files)} /></Field>
        {m.isError && <ErrorBox error={m.error} />}
        {m.data && (
          <ul className="max-h-48 space-y-1 overflow-y-auto text-sm scrollbar-thin">
            {m.data.results.map((r, i) => (
              <li key={i} className="flex items-center justify-between gap-2">
                <span className="truncate">{r.title}</span>
                <Badge tone={r.status === "indexed" ? "ok" : r.status === "duplicate" ? "warn" : "bad"}>{r.status}{r.error ? `: ${r.error}` : ""}</Badge>
              </li>
            ))}
          </ul>
        )}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={() => onOpenChange(false)}>Close</Button>
          <Button variant="primary" loading={m.isPending} disabled={!manifest || !files?.length} onClick={() => m.mutate()}>Import</Button>
        </div>
      </div>
    </Dialog>
  );
}

function EditDialog({ song, onClose }: { song: Song | null; onClose: () => void }) {
  const qc = useQueryClient();
  const detail = useQuery({ queryKey: ["song", song?.id], queryFn: () => api.song(song!.id), enabled: !!song });
  const [patch, setPatch] = useState<Record<string, string>>({});
  const [lyrics, setLyrics] = useState<string | null>(null);
  const [lyricsLang, setLyricsLang] = useState<string | null>(null);
  const save = useMutation({
    mutationFn: async () => {
      const d = detail.data!;
      const p: Record<string, unknown> = { ...patch };
      if (p.year !== undefined) p.year = p.year ? Number(p.year) : null;
      if (Object.keys(p).length) await api.updateSong(d.id, p as any);
      if (lyrics !== null || lyricsLang !== null)
        await api.setLyrics(d.id, { text: lyrics ?? d.lyrics?.text ?? "", language: lyricsLang ?? d.lyrics?.language ?? "en" });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["songs"] });
      setPatch({});
      setLyrics(null);
      setLyricsLang(null);
      onClose();
    },
  });
  const d = detail.data;
  const val = (k: keyof Song) => patch[k] ?? (d?.[k] != null ? String(d[k]) : "");
  const field = (k: keyof Song, label: string) => (
    <Field label={label} htmlFor={`e-${k}`}>
      <Input id={`e-${k}`} value={val(k)} onChange={(e) => setPatch({ ...patch, [k]: e.target.value })} />
    </Field>
  );
  return (
    <Dialog open={!!song} onOpenChange={(o) => !o && onClose()} title={song?.title ?? ""} description="Metadata, provenance and lyrics" wide>
      {!d ? (
        <Skeleton className="h-60" />
      ) : (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            {field("title", "Title")}
            {field("artist", "Artist")}
            {field("album", "Album")}
            {field("year", "Year")}
            {field("language", "Language")}
            {field("composition_id", "Composition ID")}
            {field("license", "License")}
            {field("source", "Source")}
          </div>
          <div className="grid gap-2 rounded-xl bg-elevated/40 p-3 text-xs text-muted sm:grid-cols-2">
            <span>Reference: {d.reference_type} ({d.reference_filename})</span>
            <span>Status: {d.index_status} · features {d.feature_version}</span>
            <span>{d.n_notes} notes · {d.n_windows} windows · {formatTime(d.duration_s)}</span>
            <span className="truncate" title={d.content_hash}>SHA-256 {d.content_hash.slice(0, 16)}…</span>
            {d.index_error && <span className="text-bad sm:col-span-2">{d.index_error}</span>}
            {typeof d.extra?.melody_method === "string" && <span className="sm:col-span-2">Melody extraction: {String(d.extra.melody_method)}</span>}
          </div>
          <Field label="Lyrics" hint={d.lyrics?.source ? `Source: ${d.lyrics.source} · License: ${d.lyrics.license ?? "–"}` : "Only add lyrics you are authorised to use."}>
            <Textarea rows={7} value={lyrics ?? d.lyrics?.text ?? ""} onChange={(e) => setLyrics(e.target.value)} />
            <Input className="mt-2 w-28" value={lyricsLang ?? d.lyrics?.language ?? "en"} onChange={(e) => setLyricsLang(e.target.value)} aria-label="Lyrics language" />
          </Field>
          {save.isError && <ErrorBox error={save.error} />}
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>Cancel</Button>
            <Button variant="primary" loading={save.isPending} onClick={() => save.mutate()}>Save</Button>
          </div>
        </div>
      )}
    </Dialog>
  );
}

export default function LibraryPage() {
  const { t } = useAppSettings();
  const qc = useQueryClient();
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("all");
  const [sort, setSort] = useState("created");
  const [adding, setAdding] = useState(false);
  const [bulk, setBulk] = useState(false);
  const [editing, setEditing] = useState<Song | null>(null);
  const [deleting, setDeleting] = useState<Song | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const songs = useQuery({ queryKey: ["songs", q, status], queryFn: () => api.songs(q, status === "all" ? "" : status) });
  const st = useQuery({ queryKey: ["catalog-status"], queryFn: api.catalogStatus });
  const dups = useQuery({ queryKey: ["duplicates"], queryFn: api.duplicates });
  const job = useJob(jobId, () => {
    qc.invalidateQueries({ queryKey: ["songs"] });
    qc.invalidateQueries({ queryKey: ["catalog-status"] });
    qc.invalidateQueries({ queryKey: ["health"] });
  });
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["songs"] });
    qc.invalidateQueries({ queryKey: ["catalog-status"] });
  };
  const del = useMutation({ mutationFn: (id: string) => api.deleteSong(id), onSuccess: () => { setDeleting(null); refresh(); qc.invalidateQueries({ queryKey: ["health"] }); } });
  const reindex = useMutation({ mutationFn: (id: string) => api.reindexSong(id), onSuccess: refresh });

  const items = useMemo(() => {
    const list = [...(songs.data?.items ?? [])];
    if (sort === "title") list.sort((a, b) => a.title.localeCompare(b.title));
    if (sort === "artist") list.sort((a, b) => a.artist.localeCompare(b.artist));
    if (sort === "duration") list.sort((a, b) => (b.duration_s ?? 0) - (a.duration_s ?? 0));
    return list;
  }, [songs.data, sort]);
  const play = (id: string) => {
    audio.current?.pause();
    if (playing === id) return setPlaying(null);
    audio.current = new Audio(api.songAudioUrl(id));
    audio.current.onended = () => setPlaying(null);
    audio.current.play().then(() => setPlaying(id)).catch(() => setPlaying(null));
  };
  const running = job.data && !["completed", "failed", "cancelled", "interrupted"].includes(job.data.status);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{t("lib.title")}</h1>
          <p className="text-sm text-muted">Every song keeps its source, license and content hash. Indexing is incremental.</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={async () => setJobId((await api.importDemo()).id)} disabled={!!running}>
            <Wand2 className="h-4 w-4" /> {t("lib.demo")}
          </Button>
          <Button variant="outline" onClick={async () => setJobId((await api.reindexAll(false)).id)} disabled={!!running}>
            <RefreshCw className="h-4 w-4" /> Reindex pending
          </Button>
          <Button variant="outline" onClick={() => setBulk(true)}>
            <Upload className="h-4 w-4" /> {t("lib.bulk")}
          </Button>
          <Button variant="primary" onClick={() => setAdding(true)}>
            <Plus className="h-4 w-4" /> {t("lib.add")}
          </Button>
        </div>
      </div>

      {job.data && (
        <Card className="p-4">
          <div className="mb-2 flex justify-between text-sm">
            <span>{job.data.kind.replace("_", " ")}: {job.data.status} {job.data.message ? `— ${job.data.message}` : ""}</span>
            <span className="tabular text-muted">{Math.round(job.data.progress * 100)}%</span>
          </div>
          <Progress value={job.data.progress} />
          {job.data.error && <p className="mt-2 text-sm text-bad">{job.data.error}</p>}
        </Card>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Songs" value={st.data?.songs ?? "–"} />
        <Stat label="Indexed" value={st.data?.by_status.indexed ?? 0} />
        <Stat label="With lyrics" value={st.data?.with_lyrics ?? "–"} />
        <Stat label="Melody windows" value={st.data?.melody_index.windows.toLocaleString() ?? "–"} hint={`Overlapping windows of ${st.data?.melody_index.window_lengths.join(", ")} notes`} />
        <Stat label="Features" value={st.data?.feature_version ?? "–"} hint="Feature-extraction version; songs from older versions are marked stale" />
      </div>

      {!!dups.data?.length && (
        <Card className="border-warn/40 p-4">
          <div className="flex items-center gap-2 text-sm text-warn">
            <Copy className="h-4 w-4" /> {dups.data.length} possible duplicate group(s) (same title+artist or composition id). Identical files are always rejected at import.
          </div>
        </Card>
      )}

      <Card>
        <div className="flex flex-wrap items-center gap-3 p-4">
          <Input className="max-w-xs" value={q} onChange={(e) => setQ(e.target.value)} placeholder={t("lib.search")} aria-label="Search songs" />
          <Select className="w-40" value={status} onValueChange={setStatus} ariaLabel="Filter by status" options={["all", "indexed", "pending", "processing", "error", "stale"].map((s) => ({ value: s, label: s === "all" ? "All statuses" : s }))} />
          <Select className="w-40" value={sort} onValueChange={setSort} ariaLabel="Sort" options={[{ value: "created", label: "Date added" }, { value: "title", label: "Title" }, { value: "artist", label: "Artist" }, { value: "duration", label: "Duration" }]} />
          <span className="ml-auto text-xs text-muted">{songs.data?.total ?? 0} songs</span>
        </div>
        {songs.isLoading && <div className="space-y-2 p-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-12" />)}</div>}
        {songs.isError && <div className="p-4"><ErrorBox error={songs.error} onRetry={() => songs.refetch()} /></div>}
        {songs.data && !items.length && (
          <div className="flex flex-col items-center gap-3 px-6 py-14 text-center text-muted">
            <Database className="h-10 w-10" />
            <p>{q || status !== "all" ? "No songs match your filters." : t("lib.empty")}</p>
          </div>
        )}
        {!!items.length && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted">
                <tr className="border-y border-border/60">
                  <th className="px-4 py-2 font-medium">Song</th>
                  <th className="hidden px-4 py-2 font-medium md:table-cell">Reference</th>
                  <th className="hidden px-4 py-2 font-medium xl:table-cell">Source / license</th>
                  <th className="px-4 py-2 font-medium">Status</th>
                  <th className="px-4 py-2 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {items.map((s) => (
                  <tr key={s.id} className="border-b border-border/40 hover:bg-elevated/30">
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-3">
                        <button onClick={() => play(s.id)} className="relative" aria-label={`Play ${s.title}`}>
                          <Artwork id={s.id} size={38} />
                          <span className="absolute inset-0 grid place-items-center rounded-xl bg-black/30 opacity-0 transition-opacity hover:opacity-100">
                            {playing === s.id ? <Pause className="h-4 w-4 text-white" /> : <Play className="h-4 w-4 text-white" />}
                          </span>
                        </button>
                        <div className="min-w-0">
                          <div className="truncate font-medium">{s.title}</div>
                          <div className="truncate text-xs text-muted">
                            {s.artist}
                            {s.year ? ` · ${s.year}` : ""} · {formatTime(s.duration_s)}
                            {s.has_lyrics && <Badge tone="info" className="ml-2 py-0">lyrics {s.lyrics_language}</Badge>}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="hidden px-4 py-2.5 text-xs text-muted md:table-cell">
                      <span className="inline-flex items-center gap-1"><FileMusic className="h-3.5 w-3.5" /> {s.reference_type}</span>
                      <div>{s.n_notes ?? "–"} notes</div>
                    </td>
                    <td className="hidden max-w-[220px] px-4 py-2.5 text-xs text-muted xl:table-cell">
                      <div className="truncate" title={s.source ?? ""}>{s.source ?? "–"}</div>
                      <div>{s.license ?? "license unknown"}</div>
                    </td>
                    <td className="px-4 py-2.5">
                      <Badge tone={STATUS_TONE[s.index_status]} title={s.index_error ?? undefined}>
                        {s.index_status === "error" && <AlertTriangle className="h-3 w-3" />}
                        {s.index_status}
                      </Badge>
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex justify-end gap-1">
                        <Button size="icon" variant="ghost" aria-label="Edit" onClick={() => setEditing(s)}><Pencil className="h-4 w-4" /></Button>
                        <Button size="icon" variant="ghost" aria-label="Reindex" onClick={() => reindex.mutate(s.id)} disabled={reindex.isPending}><RefreshCw className="h-4 w-4" /></Button>
                        <Button size="icon" variant="ghost" aria-label="Delete" onClick={() => setDeleting(s)}><Trash2 className="h-4 w-4 text-bad" /></Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <AddSongDialog open={adding} onOpenChange={setAdding} />
      <BulkDialog open={bulk} onOpenChange={setBulk} />
      <EditDialog song={editing} onClose={() => setEditing(null)} />
      <Dialog open={!!deleting} onOpenChange={(o) => !o && setDeleting(null)} title={`Delete “${deleting?.title}”?`} description="The song, its features, lyrics and reference file are removed from the catalog and all indexes.">
        {del.isError && <ErrorBox error={del.error} />}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setDeleting(null)}>Cancel</Button>
          <Button variant="danger" loading={del.isPending} onClick={() => deleting && del.mutate(deleting.id)}>
            <Trash2 className="h-4 w-4" /> Delete
          </Button>
        </div>
      </Dialog>
    </div>
  );
}
