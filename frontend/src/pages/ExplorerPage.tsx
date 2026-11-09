import { useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { AudioWaveform, Mic, Square, Upload } from "lucide-react";
import { api } from "@/services/api";
import type { AnalysisResult } from "@/types/api";
import { useRecorder } from "@/hooks/useRecorder";
import { useAppSettings } from "@/hooks/useAppSettings";
import { fileToWav } from "@/lib/wav";
import { Badge, Button, Card, CardHeader, ErrorBox, Progress, Select, Stat } from "@/components/ui/primitives";
import { AlignedContours, Chromagram, CostMatrix, LiveWaveform, PianoRoll, PitchContour, Spectrogram, Waveform } from "@/visualizations/canvas";
import { cn, fixed, formatTime, pct } from "@/lib/utils";
import { DEMO } from "@/lib/demo";

const LAYERS = [
  { id: "waveform", label: "Waveform" },
  { id: "spectrogram", label: "Spectrogram" },
  { id: "pitch", label: "Pitch contour" },
  { id: "pianoroll", label: "Piano roll" },
  { id: "chroma", label: "Chroma" },
  { id: "notes", label: "Notes & intervals" },
  { id: "dtw", label: "DTW alignment" },
] as const;
type Layer = (typeof LAYERS)[number]["id"];

export default function ExplorerPage() {
  const { settings } = useAppSettings();
  const [profile, setProfile] = useState("hum");
  const [extractor, setExtractor] = useState("pyin");
  const [songId, setSongId] = useState("auto");
  const [layers, setLayers] = useState<Set<Layer>>(new Set(["spectrogram", "pitch", "pianoroll", "dtw"]));
  const [audio, setAudio] = useState<Blob | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const songs = useQuery({ queryKey: ["songs", ""], queryFn: () => api.songs() });
  const m = useMutation({ mutationFn: (b: Blob) => api.analyze(b, profile, extractor, songId === "auto" ? undefined : songId) });
  const rec = useRecorder({
    maxSeconds: settings.max_record_seconds,
    deviceId: settings.microphone_id,
    onComplete: (r) => {
      setAudio(r.blob);
      m.mutate(r.blob);
    },
  });
  const recording = rec.state === "recording";
  const a: AnalysisResult | undefined = m.data;
  const toggle = (l: Layer) => setLayers((s) => {
    const n = new Set(s);
    n.has(l) ? n.delete(l) : n.add(l);
    return n;
  });
  const notesData = a ? { onsets: a.notes.map((n) => n.onset), durations: a.notes.map((n) => n.duration), pitches: a.notes.map((n) => n.midi), confidence: a.notes.map((n) => n.confidence) } : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Melody Explorer</h1>
        <p className="text-sm text-muted">Inspect what the engine actually hears: every plot below is computed from your recording.</p>
      </div>
      <Card className="p-5">
        <div className="flex flex-wrap items-end gap-4">
          <div className="w-36">
            <div className="label mb-1">Profile</div>
            <Select value={profile} onValueChange={setProfile} ariaLabel="Pitch profile" options={[{ value: "hum", label: "Hum / sing" }, { value: "whistle", label: "Whistle" }]} />
          </div>
          <div className="w-52">
            <div className="label mb-1">Pitch extractor</div>
            <Select value={extractor} onValueChange={setExtractor} ariaLabel="Pitch extractor" options={(models.data?.pitch_extractors ?? [{ name: "pyin", label: "pYIN", available: true }]).map((e) => ({ value: e.name, label: e.label + (e.available ? "" : " (not installed)"), disabled: !e.available }))} />
          </div>
          <div className="w-64">
            <div className="label mb-1">Compare with</div>
            <Select value={songId} onValueChange={setSongId} ariaLabel="Reference song" options={[{ value: "auto", label: "Best melody match" }, ...(songs.data?.items ?? []).filter((s) => s.index_status === "indexed").map((s) => ({ value: s.id, label: s.title }))]} />
          </div>
          <div className="ml-auto flex gap-2">
            {DEMO ? (
              // The static demo cannot analyse audio: it shows the captured analysis of an example hum.
              <Button variant="primary" onClick={() => m.mutate(new Blob())} disabled={m.isPending}>
                <AudioWaveform className="h-4 w-4" /> Analyse an example hum
              </Button>
            ) : (
              <Button variant={recording ? "danger" : "primary"} onClick={() => (recording ? rec.stop() : rec.start())} disabled={m.isPending}>
                {recording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                {recording ? `Stop (${formatTime(rec.elapsed)})` : "Record"}
              </Button>
            )}
            <input ref={fileRef} type="file" accept="audio/*" className="hidden" onChange={async (e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              const w = await fileToWav(f);
              setAudio(w);
              m.mutate(w);
            }} />
            {!DEMO && (
              <Button variant="outline" onClick={() => fileRef.current?.click()} disabled={m.isPending || recording}>
                <Upload className="h-4 w-4" /> Upload
              </Button>
            )}
            {audio && !recording && (
              <Button variant="ghost" onClick={() => m.mutate(audio)} disabled={m.isPending}>Re-analyse</Button>
            )}
          </div>
        </div>
        {recording && <div className="mt-4"><LiveWaveform analyser={rec.analyser} active height={48} /></div>}
        {rec.error && <p className="mt-3 text-sm text-brand">{rec.error}</p>}
        <div className="mt-4 flex flex-wrap gap-2" role="group" aria-label="Visible layers">
          {LAYERS.map((l) => (
            <button key={l.id} onClick={() => toggle(l.id)} aria-pressed={layers.has(l.id)} className={cn("rounded-md border px-2.5 py-1 text-xs transition-colors", layers.has(l.id) ? "border-brand/50 bg-brand/10 text-fg" : "border-border text-muted hover:text-fg")}>
              {l.label}
            </button>
          ))}
        </div>
      </Card>

      {m.isPending && <Card className="p-5"><div className="mb-3 text-sm">Analysing ({extractor})…</div><Progress indeterminate /></Card>}
      {m.isError && <ErrorBox error={m.error} />}
      {!a && !m.isPending && (
        <Card className="flex flex-col items-center gap-3 px-6 py-16 text-center text-muted">
          <AudioWaveform className="h-10 w-10" />
          <p>{DEMO ? "Analyse the example hum" : "Record or upload a hummed, whistled or sung melody"} to see its waveform, spectrum, pitch, notes and alignment.</p>
        </Card>
      )}

      {a && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
            <Stat label="Duration" value={`${fixed(a.waveform.duration_s, 1)} s`} />
            <Stat label="Voiced" value={`${fixed(Number(a.quality.voiced_duration_s ?? 0), 1)} s`} />
            <Stat label="Voiced ratio" value={pct(Number(a.quality.voiced_ratio ?? 0))} />
            <Stat label="Median f0" value={a.quality.median_f0_hz ? `${fixed(Number(a.quality.median_f0_hz), 0)} Hz` : "–"} />
            <Stat label="Notes" value={a.notes.length} />
            <Stat label="Pitch time" value={`${fixed((a.timings.pitch_s ?? 0) * 1000, 0)} ms`} />
          </div>
          {!a.informative && <p className="text-sm text-brand">Not enough stable pitched content for melody matching (need ≈1.5 s voiced and ≥3 notes).</p>}

          {layers.has("waveform") && (
            <Card><CardHeader title="Waveform" subtitle="Min/max envelope" /><div className="p-4"><Waveform min={a.waveform.min} max={a.waveform.max} duration={a.waveform.duration_s} /></div></Card>
          )}
          {layers.has("spectrogram") && (
            <Card><CardHeader title="Spectrogram" subtitle="Log-frequency STFT magnitude (dB) with the tracked f0 overlaid in cyan" /><div className="p-4"><Spectrogram data={a.spectrogram.data} hop={a.spectrogram.hop_s} freqs={a.spectrogram.freqs_hz} pitch={a.pitch_track} /></div></Card>
          )}
          {layers.has("pitch") && a.pitch_track && (
            <Card>
              <CardHeader title="Pitch contour" subtitle={`${a.pitch_track.extractor} f0 in MIDI semitones · bottom band: voiced (green) / unvoiced (red), opacity = voicing probability · boxes = segmented notes`} />
              <div className="p-4"><PitchContour track={a.pitch_track} notes={a.notes} height={260} /></div>
            </Card>
          )}
          {layers.has("pianoroll") && notesData && (
            <Card>
              <CardHeader title="Piano roll" subtitle="Notes segmented from your recording (time in seconds of the recording)" />
              <div className="p-4"><PianoRoll notes={notesData} /></div>
            </Card>
          )}
          {layers.has("chroma") && (
            <Card><CardHeader title="Chromagram" subtitle="CQT chroma (pitch-class energy) at 10 frames/s, used by the chroma DTW baseline" /><div className="p-4"><Chromagram data={a.chroma.data} rate={a.chroma.rate_hz} /></div></Card>
          )}
          {layers.has("notes") && (
            <Card>
              <CardHeader title="Notes & intervals" subtitle="Key-invariant interval sequence (semitones) used by the note-alignment matcher" />
              <div className="flex flex-wrap gap-1.5 p-4 font-mono text-xs">
                {a.notes.map((n, i) => (
                  <span key={i} className="flex items-center gap-1.5">
                    <Badge tone="strong" className="font-mono" title={`${fixed(n.onset, 2)} s, ${fixed(n.duration, 2)} s, conf ${fixed(n.confidence, 2)}`}>{n.name}</Badge>
                    {i < a.intervals.length && <span className={cn(a.intervals[i] > 0 ? "text-highlight" : a.intervals[i] < 0 ? "text-brand" : "text-muted")}>{a.intervals[i] > 0 ? "+" : ""}{fixed(a.intervals[i], 1)}</span>}
                  </span>
                ))}
              </div>
            </Card>
          )}
          {layers.has("dtw") && a.comparison?.alignment && (
            <div className="grid gap-5 lg:grid-cols-2">
              <Card>
                <CardHeader title={`Alignment with “${a.comparison.song?.title ?? a.comparison.song_id}”`} subtitle={`distance ${fixed(a.comparison.alignment.distance, 3)} · tempo ×${fixed(a.comparison.alignment.tempo_scale, 2)} · transposition ${fixed(a.comparison.alignment.query_transposition_semitones, 1)} st`} />
                <div className="p-4"><AlignedContours alignment={a.comparison.alignment} /></div>
              </Card>
              <Card>
                <CardHeader title="DTW accumulated cost & path" subtitle="Subsequence DTW over the reference region" />
                <div className="p-4"><CostMatrix alignment={a.comparison.alignment} /></div>
              </Card>
              <Card className="lg:col-span-2">
                <CardHeader title="Reference melody (piano roll)" subtitle="Notes of the compared song, as stored in the index" />
                <div className="p-4"><PianoRoll notes={a.comparison.reference_notes} height={160} /></div>
              </Card>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
