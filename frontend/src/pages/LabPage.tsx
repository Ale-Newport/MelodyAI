import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis, Scatter, ScatterChart, ZAxis } from "recharts";
import { Beaker, Brain, Cpu, Gauge, Mic, Play, Square, StopCircle, Upload } from "lucide-react";
import { api } from "@/services/api";
import type { EvalMetrics, Job, RecognitionResult } from "@/types/api";
import { useJob, FINISHED } from "@/hooks/useJob";
import { useRecorder } from "@/hooks/useRecorder";
import { useAppSettings } from "@/hooks/useAppSettings";
import { fileToWav } from "@/lib/wav";
import { Badge, Button, Card, CardHeader, ErrorBox, Field, Input, InfoTip, Progress, Select, Skeleton, Stat, Switch, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { cn, fixed, formatTime, pct } from "@/lib/utils";

function JobCard({ job, onCancel }: { job: Job; onCancel: () => void }) {
  const done = FINISHED.includes(job.status);
  return (
    <div className="rounded-xl border border-border/60 bg-elevated/40 p-4">
      <div className="mb-2 flex items-center justify-between gap-3 text-sm">
        <span className="flex items-center gap-2">
          <Badge tone={job.status === "completed" ? "ok" : job.status === "failed" ? "bad" : job.status === "cancelled" ? "warn" : "info"}>{job.status}</Badge>
          <span className="text-muted">{job.kind}</span>
          <span className="truncate text-xs text-muted">{job.message}</span>
        </span>
        {!done && (
          <Button size="sm" variant="danger" onClick={onCancel}>
            <StopCircle className="h-4 w-4" /> Cancel
          </Button>
        )}
      </div>
      <Progress value={job.progress} />
      {job.error && <p className="mt-2 text-xs text-bad">{job.error}</p>}
    </div>
  );
}

function heat(v: number | undefined) {
  if (v == null) return {};
  const h = 10 + v * 130;
  return { background: `hsl(${h} 65% 45% / ${0.15 + 0.55 * v})` };
}

function RunDetails({ m }: { m: EvalMetrics }) {
  const [alg, setAlg] = useState<string>(Object.keys(m.melody).includes("ensemble") ? "ensemble" : Object.keys(m.melody)[0]);
  const [cond, setCond] = useState("all");
  const [errorsOnly, setErrorsOnly] = useState(true);
  const algs = Object.keys(m.melody);
  const conds = Array.from(new Set(algs.flatMap((a) => Object.keys(m.melody[a].by_condition))));
  const pq = useQuery({
    queryKey: ["pq", m.run_id, alg, cond, errorsOnly],
    queryFn: () => api.experimentQueries(m.run_id, { algorithm: alg, condition: cond === "all" ? undefined : cond, only_errors: errorsOnly }),
  });
  const cal = m.melody[alg]?.rejection.calibration_test;
  const scoreDist = useMemo(() => (pq.data?.items ?? []).map((r: any) => ({ score: r.top_score, rank: r.rank ?? 11, cond: r.condition })), [pq.data]);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Dataset" value={<span className="text-sm">{m.dataset.type}</span>} hint={m.dataset.catalog} />
        <Stat label="Queries" value={m.dataset.queries} hint={`Conditions: ${m.dataset.conditions.join(", ")}`} />
        <Stat label="Index build" value={`${fixed(m.indexing.mean_ms_per_song, 0)} ms/song`} hint={`${m.indexing.windows} windows, ${m.indexing.window_matrix_mb} MB window matrix`} />
        <Stat label="Hardware" value={<span className="text-sm">{m.env.machine} · {m.env.cpu_count} CPU</span>} hint={`${m.env.platform} · ${m.env.memory_gb} GB · torch ${m.env.torch_version} (${m.env.torch_device})`} />
      </div>
      <p className="rounded-xl border border-warn/30 bg-warn/5 p-3 text-xs text-warn">
        Synthetic benchmark: queries are synthesised hums/whistles of original melodies. Results show robustness to controlled transformations, not accuracy on real users' recordings.
      </p>

      <Card>
        <CardHeader title="Algorithm comparison" subtitle="In-catalog retrieval (all conditions) and rejection on the held-out test half" />
        <div className="overflow-x-auto p-4">
          <table className="w-full text-sm tabular">
            <thead className="text-left text-xs text-muted">
              <tr>
                <th className="p-2">Algorithm</th><th className="p-2">Top-1</th><th className="p-2">Top-3</th><th className="p-2">Top-5</th><th className="p-2">MRR</th>
                <th className="p-2">Latency</th><th className="p-2">AUC vs OOC <InfoTip content="Probability that a correct top-1 score exceeds an out-of-catalog score" /></th>
                <th className="p-2">FAR @ thr <InfoTip content="False-accept rate on out-of-catalog/non-melodic test queries at the threshold tuned on the dev half (target 5%)" /></th>
                <th className="p-2">Accepted correct</th>
              </tr>
            </thead>
            <tbody>
              {algs.map((a) => {
                const x = m.melody[a];
                return (
                  <tr key={a} className={cn("border-t border-border/40", a === alg && "bg-accent/5")} onClick={() => setAlg(a)}>
                    <td className="cursor-pointer p-2 font-medium" title={m.algorithms[a]}>{a}</td>
                    <td className="p-2">{pct(x.overall.top1, 1)}</td>
                    <td className="p-2">{pct(x.overall.top3, 1)}</td>
                    <td className="p-2">{pct(x.overall.top5, 1)}</td>
                    <td className="p-2">{fixed(x.overall.mrr, 3)}</td>
                    <td className="p-2">{fixed(x.latency.mean_ms, 0)} ms</td>
                    <td className="p-2">{fixed(x.rejection.test.auc_correct_vs_ooc, 3)}</td>
                    <td className="p-2">{pct(x.rejection.test.false_accept_rate_ooc, 1)}</td>
                    <td className="p-2">{pct(x.rejection.test.accepted_correct_rate, 1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <CardHeader title="Top-1 by condition" subtitle="Each cell = one algorithm on one query condition" />
        <div className="overflow-x-auto p-4">
          <table className="w-full text-xs tabular">
            <thead><tr><th className="p-2 text-left text-muted">Algorithm</th>{conds.map((c) => <th key={c} className="p-2 text-muted">{c.replace(/_/g, " ")}</th>)}</tr></thead>
            <tbody>
              {algs.map((a) => (
                <tr key={a}>
                  <td className="p-2 font-medium">{a}</td>
                  {conds.map((c) => <td key={c} className="p-2 text-center" style={heat(m.melody[a].by_condition[c]?.top1)}>{pct(m.melody[a].by_condition[c]?.top1)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader title={`Calibration (${alg})`} subtitle={cal ? `Platt scaling fitted on dev, ECE ${fixed(cal.ece, 3)} · Brier ${fixed(cal.brier, 3)} on ${cal.n} test queries` : "No calibration data"} />
          <div className="h-64 p-4">
            {cal && (
              <ResponsiveContainer>
                <LineChart data={cal.reliability.map((b) => ({ conf: +b.confidence.toFixed(3), acc: b.accuracy, ideal: b.confidence, n: b.n }))}>
                  <CartesianGrid strokeOpacity={0.15} />
                  <XAxis dataKey="conf" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: "predicted", position: "insideBottom", offset: -2, fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <RTooltip contentStyle={{ background: "rgb(var(--surface))", border: "1px solid rgb(var(--border))", borderRadius: 12 }} />
                  <Line dataKey="ideal" stroke="#888" strokeDasharray="4 4" dot={false} name="perfect calibration" />
                  <Line dataKey="acc" stroke="rgb(var(--accent))" strokeWidth={2} name="observed accuracy" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </Card>
        <Card>
          <CardHeader title="Top score vs rank" subtitle={`Per query (${alg}${cond !== "all" ? `, ${cond}` : ""}${errorsOnly ? ", errors only" : ""}); rank 11 = not in top 10 / not in catalog`} />
          <div className="h-64 p-4">
            <ResponsiveContainer>
              <ScatterChart>
                <CartesianGrid strokeOpacity={0.15} />
                <XAxis dataKey="score" type="number" domain={[0, 1]} tick={{ fontSize: 11 }} name="top score" />
                <YAxis dataKey="rank" type="number" domain={[1, 11]} tick={{ fontSize: 11 }} name="rank of correct song" reversed />
                <ZAxis range={[30, 30]} />
                <RTooltip contentStyle={{ background: "rgb(var(--surface))", border: "1px solid rgb(var(--border))", borderRadius: 12 }} />
                <Scatter data={scoreDist} fill="rgb(var(--accent2))" fillOpacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader
          title="Per-query inspector"
          subtitle="Error analysis: which queries fail, what was returned instead and with which score"
          actions={
            <div className="flex flex-wrap items-center gap-2">
              <Select className="w-40" value={alg} onValueChange={setAlg} ariaLabel="Algorithm" options={algs.map((a) => ({ value: a, label: a }))} />
              <Select className="w-48" value={cond} onValueChange={setCond} ariaLabel="Condition" options={[{ value: "all", label: "All conditions" }, ...conds.map((c) => ({ value: c, label: c })), { value: "out_of_catalog", label: "out_of_catalog" }, { value: "non_melodic", label: "non_melodic" }]} />
              <label className="flex items-center gap-2 text-xs text-muted"><Switch checked={errorsOnly} onCheckedChange={setErrorsOnly} label="Errors only" /> errors only</label>
            </div>
          }
        />
        <div className="max-h-96 overflow-auto p-4 scrollbar-thin">
          {pq.isLoading ? <Skeleton className="h-40" /> : (
            <table className="w-full text-xs tabular">
              <thead className="sticky top-0 bg-surface text-left text-muted"><tr><th className="p-1.5">Query</th><th className="p-1.5">Target</th><th className="p-1.5">Returned</th><th className="p-1.5">Rank</th><th className="p-1.5">Score</th><th className="p-1.5">2nd</th><th className="p-1.5">Latency</th></tr></thead>
              <tbody>
                {(pq.data?.items ?? []).map((r: any) => (
                  <tr key={r.query + r.algorithm} className="border-t border-border/30">
                    <td className="p-1.5 font-mono">{r.query}</td>
                    <td className="p-1.5">{r.target ?? "—"}</td>
                    <td className={cn("p-1.5", r.target && r.top_song !== r.target && "text-bad")}>{r.top_song ?? "∅"}</td>
                    <td className="p-1.5">{r.rank ?? "–"}</td>
                    <td className="p-1.5">{fixed(r.top_score, 3)}</td>
                    <td className="p-1.5">{fixed(r.second_score, 3)}</td>
                    <td className="p-1.5">{fixed(r.latency_s * 1000, 0)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {pq.data && !pq.data.items.length && <p className="p-4 text-center text-sm text-muted">No queries for this filter.</p>}
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        {m.pitch_extractors && (
          <Card>
            <CardHeader title="Pitch extractors" subtitle={`Same queries, ${m.pitch_extractors.algorithm} matcher`} />
            <table className="m-4 w-[calc(100%-2rem)] text-sm tabular">
              <thead className="text-left text-xs text-muted"><tr><th>Extractor</th><th>Top-1</th><th>MRR</th><th>Pitch time</th></tr></thead>
              <tbody>{Object.entries(m.pitch_extractors.results).map(([k, v]) => <tr key={k} className="border-t border-border/40"><td className="py-1.5">{k}</td><td>{pct(v.overall.top1, 1)}</td><td>{fixed(v.overall.mrr, 3)}</td><td>{fixed(v.pitch_time.mean_ms, 0)} ms</td></tr>)}</tbody>
            </table>
          </Card>
        )}
        {m.lyrics && (
          <Card>
            <CardHeader title="Lyrics search" subtitle={`FTS5 + fuzzy token alignment · rejection FAR ${pct(m.lyrics.rejection.test.false_accept_rate_ooc, 0)} on distractors`} />
            <table className="m-4 w-[calc(100%-2rem)] text-sm tabular">
              <thead className="text-left text-xs text-muted"><tr><th>Condition</th><th>Top-1</th><th>MRR</th></tr></thead>
              <tbody>{Object.entries(m.lyrics.by_condition).map(([k, v]) => <tr key={k} className="border-t border-border/40"><td className="py-1.5">{k}</td><td>{pct(v.top1, 1)}</td><td>{fixed(v.mrr, 3)}</td></tr>)}</tbody>
            </table>
          </Card>
        )}
        {m.fusion && !m.fusion.skipped && (() => {
          const variants: Record<string, any> = m.fusion!.variants ?? { "4_words": m.fusion };
          const names = ["melody_only", "lyrics_only", "weighted", "rrf", "combmnz", "learned"].filter((k) => Object.values(variants).every((v: any) => v[k]));
          return (
            <Card>
              <CardHeader title="Fusion (sing + words)" subtitle={`${m.fusion!.n_queries} short/noisy hums, each paired with a few words (≈30% typos)`} />
              <table className="m-4 w-[calc(100%-2rem)] text-sm tabular">
                <thead className="text-left text-xs text-muted"><tr><th>Strategy</th>{Object.keys(variants).map((v) => <th key={v}>Top-1 ({v.replace("_", " ")})</th>)}</tr></thead>
                <tbody>
                  {names.map((k) => (
                    <tr key={k} className="border-t border-border/40">
                      <td className="py-1.5">{k}{k === "learned" && <InfoTip content="Trained on the dev half, evaluated on the test half (small sample)" />}</td>
                      {Object.values(variants).map((v: any, i) => <td key={i}>{pct(v[k].top1, 1)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          );
        })()}
        {m.polyphonic && (
          <Card>
            <CardHeader title="Reference modality" subtitle={`Same ${m.polyphonic.n_queries} hummed queries vs references derived differently (${m.polyphonic.songs} songs)`} />
            <table className="m-4 w-[calc(100%-2rem)] text-sm tabular">
              <thead className="text-left text-xs text-muted"><tr><th>Reference</th><th>Top-1</th><th>MRR</th><th>Note F1</th></tr></thead>
              <tbody>{Object.entries(m.polyphonic.modalities).map(([k, v]) => <tr key={k} className="border-t border-border/40"><td className="py-1.5">{k}</td><td>{pct(v.retrieval.top1, 1)}</td><td>{fixed(v.retrieval.mrr, 3)}</td><td>{fixed(v.note_f1_mean, 2)}</td></tr>)}</tbody>
            </table>
          </Card>
        )}
        {m.asr && (
          <Card>
            <CardHeader title="Speech → lyrics (ASR)" subtitle={m.asr.skipped ?? `${m.asr.model} · ${m.asr.tts}`} />
            {!m.asr.skipped && (
              <div className="grid grid-cols-2 gap-3 p-4">
                <Stat label="Top-1 (auto lang)" value={pct(m.asr.lyrics_from_speech_auto_language?.top1, 0)} />
                <Stat label="Top-1 (given lang)" value={pct(m.asr.lyrics_from_speech_given_language?.top1, 0)} />
                <Stat label="Mean WER" value={fixed(m.asr.mean_wer, 2)} />
                <Stat label="Hum → 'speech'" value={pct(m.asr.hum_false_speech_rate, 0)} hint="Share of hummed/whistled queries for which ASR produced words (hallucination guard)" />
              </div>
            )}
          </Card>
        )}
      </div>

      {!!m.figures?.length && (
        <Card>
          <CardHeader title="Figures" subtitle="Generated by the evaluation run (matplotlib)" />
          <div className="grid gap-4 p-4 md:grid-cols-2">
            {m.figures.map((f) => (
              <img key={f} src={api.figureUrl(m.run_id, f.replace("figures/", ""))} alt={f} className="w-full rounded-xl bg-white" loading="lazy" />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}

function BenchmarksTab() {
  const qc = useQueryClient();
  const runs = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const [sel, setSel] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, () => qc.invalidateQueries({ queryKey: ["experiments"] }));
  const evals = runs.data?.evaluations ?? [];
  useEffect(() => {
    if (!sel && evals.length) setSel(evals[0].run_id);
  }, [evals, sel]);
  const detail = useQuery({ queryKey: ["experiment", sel], queryFn: () => api.experiment(sel!), enabled: !!sel });
  const start = async (c: "quick" | "default") => setJobId((await api.evaluate(c)).id);
  return (
    <div className="space-y-5">
      <Card className="p-5">
        <div className="flex flex-wrap items-center gap-3">
          <Select className="w-80" value={sel ?? ""} onValueChange={setSel} ariaLabel="Evaluation run" placeholder="No runs yet" options={evals.map((r) => ({ value: r.run_id, label: `${r.run_id} (${r.dataset?.queries} queries)` }))} />
          <div className="ml-auto flex gap-2">
            <Button variant="outline" onClick={() => start("quick")} disabled={!!job.data && !FINISHED.includes(job.data.status)}><Play className="h-4 w-4" /> Quick benchmark</Button>
            <Button variant="primary" onClick={() => start("default")} disabled={!!job.data && !FINISHED.includes(job.data.status)}><Gauge className="h-4 w-4" /> Full benchmark</Button>
          </div>
        </div>
        <p className="mt-3 text-xs text-muted">Quick ≈ 1–2 min (16 songs). Full ≈ 25 min (48 songs, 10 conditions, extractors, lyrics, ASR, fusion, polyphonic). Runs in the background; you can leave this page.</p>
        {job.data && <div className="mt-4"><JobCard job={job.data} onCancel={() => api.cancelJob(job.data!.id)} /></div>}
      </Card>
      {runs.isLoading && <Skeleton className="h-64" />}
      {detail.isError && <ErrorBox error={detail.error} />}
      {detail.data && <RunDetails m={detail.data} />}
      {runs.data && !evals.length && <Card className="p-8 text-center text-muted">No benchmark runs yet. Start one above or run <code className="font-mono">make evaluate</code>.</Card>}
    </div>
  );
}

function CompareTab() {
  const { settings } = useAppSettings();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [selected, setSelected] = useState<string[]>(["dtw", "subseq_dtw", "chroma_dtw", "note_align", "ensemble"]);
  const [extractor, setExtractor] = useState("pyin");
  const [mode, setMode] = useState("hum");
  const [results, setResults] = useState<Record<string, RecognitionResult | { error: string }>>({});
  const [running, setRunning] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const run = async (blob: Blob) => {
    setRunning(true);
    setResults({});
    for (const a of selected) {
      try {
        const r = await api.recognize({ mode: mode as any, audio: blob, algorithm: a, pitchExtractor: extractor, topK: 3 });
        setResults((s) => ({ ...s, [a]: r }));
      } catch (e: any) {
        setResults((s) => ({ ...s, [a]: { error: e.message } }));
      }
    }
    setRunning(false);
  };
  const rec = useRecorder({ maxSeconds: settings.max_record_seconds, deviceId: settings.microphone_id, onComplete: (r) => run(r.blob) });
  const algs = models.data?.melody_algorithms ?? [];
  return (
    <div className="space-y-5">
      <Card className="p-5">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Algorithms to compare">
          {algs.map((a) => (
            <button key={a.name} title={a.label} aria-pressed={selected.includes(a.name)} onClick={() => setSelected((s) => (s.includes(a.name) ? s.filter((x) => x !== a.name) : [...s, a.name]))} className={cn("rounded-full border px-3 py-1 text-xs", selected.includes(a.name) ? "border-accent/50 bg-accent/15" : "border-border text-muted")}>
              {a.name}
            </button>
          ))}
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-3">
          <div className="w-36"><div className="label mb-1">Mode</div><Select value={mode} onValueChange={setMode} ariaLabel="Mode" options={[{ value: "hum", label: "Hum" }, { value: "whistle", label: "Whistle" }]} /></div>
          <div className="w-52"><div className="label mb-1">Pitch extractor</div><Select value={extractor} onValueChange={setExtractor} ariaLabel="Extractor" options={(models.data?.pitch_extractors ?? []).map((e) => ({ value: e.name, label: e.label, disabled: !e.available }))} /></div>
          <div className="ml-auto flex gap-2">
            <Button variant={rec.state === "recording" ? "danger" : "primary"} onClick={() => (rec.state === "recording" ? rec.stop() : rec.start())} disabled={running}>
              {rec.state === "recording" ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />} {rec.state === "recording" ? `Stop ${formatTime(rec.elapsed)}` : "Record query"}
            </Button>
            <input ref={fileRef} type="file" accept="audio/*" className="hidden" onChange={async (e) => e.target.files?.[0] && run(await fileToWav(e.target.files[0]))} />
            <Button variant="outline" onClick={() => fileRef.current?.click()} disabled={running}><Upload className="h-4 w-4" /> Upload</Button>
          </div>
        </div>
        {running && <Progress indeterminate className="mt-4" />}
      </Card>
      {!!Object.keys(results).length && (
        <Card>
          <CardHeader title="Side-by-side" subtitle="Same recording, each algorithm independently (scores are only comparable within one algorithm)" />
          <div className="overflow-x-auto p-4">
            <table className="w-full text-sm tabular">
              <thead className="text-left text-xs text-muted"><tr><th className="p-2">Algorithm</th><th className="p-2">Decision</th><th className="p-2">#1</th><th className="p-2">#2</th><th className="p-2">#3</th><th className="p-2">Matching time</th></tr></thead>
              <tbody>
                {selected.filter((a) => results[a]).map((a) => {
                  const r = results[a];
                  if ("error" in r) return <tr key={a}><td className="p-2">{a}</td><td colSpan={5} className="p-2 text-bad">{r.error}</td></tr>;
                  return (
                    <tr key={a} className="border-t border-border/40">
                      <td className="p-2 font-medium">{a}</td>
                      <td className="p-2"><Badge tone={r.status === "match" ? "ok" : r.status === "uncertain" ? "warn" : "neutral"}>{r.status}</Badge></td>
                      {[0, 1, 2].map((i) => <td key={i} className="p-2">{r.results[i] ? <span>{r.results[i].song.title} <span className="text-xs text-muted">{fixed(r.results[i].score, 3)}</span></span> : "–"}</td>)}
                      <td className="p-2">{fixed((r.engines.melody?.elapsed_s ?? 0) * 1000, 0)} ms</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

function TrainingTab() {
  const qc = useQueryClient();
  const exps = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const jobs = useQuery({ queryKey: ["jobs", "training"], queryFn: () => api.jobs("training"), refetchInterval: 3000 });
  const [cfg, setCfg] = useState({ config: "default" as "default" | "smoke", epochs: "", loss: "infonce", lr: "", resume: false });
  const start = useMutation({
    mutationFn: () => api.train({ config: cfg.config, epochs: cfg.epochs ? +cfg.epochs : undefined, loss: cfg.loss, lr: cfg.lr ? +cfg.lr : undefined, resume: cfg.resume }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs", "training"] }),
  });
  const active = jobs.data?.find((j) => !FINISHED.includes(j.status));
  useEffect(() => {
    if (!active) qc.invalidateQueries({ queryKey: ["experiments"] });
  }, [active, qc]);
  return (
    <div className="space-y-5">
      <Card className="p-5">
        <CardHeader title="Train the neural melody encoder" subtitle="Contrastive (InfoNCE / triplet) training on synthetic renditions of original compositions, split by composition. Runs in the background (MPS/CUDA if available)." icon={<Brain className="h-4 w-4" />} />
        <div className="mt-4 grid gap-4 px-5 sm:grid-cols-5">
          <Field label="Config"><Select value={cfg.config} onValueChange={(v) => setCfg({ ...cfg, config: v as any })} options={[{ value: "default", label: "default (≈20 min)" }, { value: "smoke", label: "smoke (seconds)" }]} /></Field>
          <Field label="Epochs"><Input type="number" min={1} max={200} placeholder="config" value={cfg.epochs} onChange={(e) => setCfg({ ...cfg, epochs: e.target.value })} /></Field>
          <Field label="Loss"><Select value={cfg.loss} onValueChange={(v) => setCfg({ ...cfg, loss: v })} options={[{ value: "infonce", label: "InfoNCE" }, { value: "triplet", label: "Triplet (hardest)" }]} /></Field>
          <Field label="Learning rate"><Input placeholder="config" value={cfg.lr} onChange={(e) => setCfg({ ...cfg, lr: e.target.value })} /></Field>
          <Field label="Resume"><div className="flex h-10 items-center"><Switch checked={cfg.resume} onCheckedChange={(v) => setCfg({ ...cfg, resume: v })} label="Resume" /></div></Field>
        </div>
        <div className="mt-4 flex justify-end px-5">
          <Button variant="primary" loading={start.isPending} disabled={!!active} onClick={() => start.mutate()}><Beaker className="h-4 w-4" /> Start training</Button>
        </div>
        {start.isError && <div className="px-5 pt-3"><ErrorBox error={start.error} /></div>}
        <div className="mt-4 space-y-2 px-5">{(jobs.data ?? []).slice(0, 3).map((j) => <JobCard key={j.id} job={j} onCancel={() => api.cancelJob(j.id).then(() => jobs.refetch())} />)}</div>
      </Card>
      {(exps.data?.training ?? []).map((r) => (
        <Card key={r.name}>
          <CardHeader title={r.name} subtitle={`${r.status} · ${r.epochs_run ?? r.history.length} epochs · best val MRR ${fixed(r.best_val_mrr, 3)} · ${r.device ?? ""} · ${r.n_params ? `${(r.n_params / 1e6).toFixed(2)}M params` : ""} ${r.train_seconds ? `· ${Math.round(r.train_seconds / 60)} min` : ""}`} />
          <div className="grid gap-4 p-4 md:grid-cols-2">
            <div className="h-56">
              <ResponsiveContainer>
                <LineChart data={r.history}>
                  <CartesianGrid strokeOpacity={0.15} />
                  <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <RTooltip contentStyle={{ background: "rgb(var(--surface))", border: "1px solid rgb(var(--border))", borderRadius: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line dataKey="loss" stroke="rgb(var(--accent))" dot={false} name="train loss" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="h-56">
              <ResponsiveContainer>
                <LineChart data={r.history}>
                  <CartesianGrid strokeOpacity={0.15} />
                  <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <RTooltip contentStyle={{ background: "rgb(var(--surface))", border: "1px solid rgb(var(--border))", borderRadius: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line dataKey="val_top1" stroke="rgb(var(--accent2))" dot={false} name="val top-1" />
                  <Line dataKey="val_mrr" stroke="rgb(var(--ok))" dot={false} name="val MRR" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
}

function ScaleTab() {
  const runs = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const id = runs.data?.scale?.[0]?.run_id;
  const m = useQuery({ queryKey: ["scale", id], queryFn: () => api.scaleRun(id!), enabled: !!id });
  if (runs.isLoading || m.isLoading) return <Skeleton className="h-64" />;
  if (!id) return <Card className="p-8 text-center text-muted">No scale benchmark yet. Run <code className="font-mono">make evaluate-scale</code> (≈40 min).</Card>;
  const d = m.data!;
  const mel = d.melody?.sizes ?? {};
  const fp = d.fingerprint?.sizes ?? {};
  const fpSizes = Object.keys(fp);
  const conds = fpSizes.length ? Object.keys(fp[fpSizes[0]].by_condition) : [];
  const curve = Object.entries(mel).map(([n, e]) => ({ n: Number(n), two: e.two_stage.top1, ann: e.neural_ann_only.top1, exh: e.exhaustive_note_align?.top1 ?? null, recall: e.candidate_recall }));
  return (
    <div className="space-y-5">
      <p className="rounded-xl border border-warn/30 bg-warn/5 p-3 text-xs text-warn">
        Synthetic: the 48 demo songs are hidden among procedurally generated distractors. Real catalogs are more diverse in timbre but contain covers, remixes and near-duplicates. Run {d.run_id}, {Math.round(d.elapsed_s / 60)} min on {d.env.machine}.
      </p>
      {!!curve.length && (
        <Card>
          <CardHeader title="Hum/whistle recognition vs catalog size" subtitle={`Two-stage = neural IVF-PQ (${d.melody!.nlist} lists, ${d.melody!.m} B/vector, nprobe ${d.melody!.nprobe}) → top-${d.melody!.n_candidates} songs → DTW + note alignment`} />
          <div className="h-64 p-4">
            <ResponsiveContainer>
              <LineChart data={curve}>
                <CartesianGrid strokeOpacity={0.15} />
                <XAxis dataKey="n" scale="log" type="number" domain={["dataMin", "dataMax"]} tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                <RTooltip contentStyle={{ background: "rgb(var(--surface))", border: "1px solid rgb(var(--border))", borderRadius: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="two" name="two-stage top-1" stroke="rgb(var(--accent))" strokeWidth={2} />
                <Line dataKey="ann" name="neural ANN only top-1" stroke="rgb(var(--accent2))" />
                <Line dataKey="recall" name="stage-1 candidate recall" stroke="rgb(var(--ok))" strokeDasharray="4 4" />
                <Line dataKey="exh" name="exhaustive note alignment top-1" stroke="#999" connectNulls={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="overflow-x-auto px-4 pb-4">
            <table className="w-full text-sm tabular">
              <thead className="text-left text-xs text-muted"><tr><th className="p-2">Songs</th><th className="p-2">Two-stage top-1</th><th className="p-2">MRR</th><th className="p-2">Candidate recall (@10 / 50 / 200 / 500)</th><th className="p-2">Latency (stage 1 + 2)</th><th className="p-2">Exhaustive top-1 / latency</th><th className="p-2">ANN index</th><th className="p-2">AUC vs OOC</th></tr></thead>
              <tbody>
                {Object.entries(mel).map(([n, e]) => (
                  <tr key={n} className="border-t border-border/40">
                    <td className="p-2 font-medium">{Number(n).toLocaleString()}</td>
                    <td className="p-2">{pct(e.two_stage.top1, 1)}</td>
                    <td className="p-2">{fixed(e.two_stage.mrr, 3)}</td>
                    <td className="p-2">{e.candidate_recall_at ? ["10", "50", "200", "500"].map((k) => pct(e.candidate_recall_at![k], 0)).join(" / ") : pct(e.candidate_recall, 1)}</td>
                    <td className="p-2">{fixed(e.latency_stage1.mean_ms, 0)} + {fixed(e.latency_stage2.mean_ms, 0)} ms</td>
                    <td className="p-2">{e.exhaustive_note_align ? `${pct(e.exhaustive_note_align.top1, 0)} / ${fixed(e.exhaustive_note_align.latency.mean_ms, 0)} ms` : "–"}</td>
                    <td className="p-2">{(e.ann_vectors / 1e6).toFixed(2)}M vec · {fixed(e.ann_index_mb, 0)} MB</td>
                    <td className="p-2">{fixed(e.auc_correct_vs_ooc, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
      {!!fpSizes.length && (
        <Card>
          <CardHeader title="Recording / TikTok-style clips vs index size" subtitle={`${d.fingerprint!.clip_seconds}-s clips; 'accepted' = top-1 correct and score ≥ ${d.fingerprint!.accept_threshold}`} />
          <div className="overflow-x-auto p-4">
            <table className="w-full text-xs tabular">
              <thead className="text-muted"><tr><th className="p-2 text-left">Edit</th>{fpSizes.map((n) => <th key={n} className="p-2">{n} rec. top-1 / accepted</th>)}</tr></thead>
              <tbody>
                {conds.map((c) => (
                  <tr key={c} className="border-t border-border/40">
                    <td className="p-2 font-medium">{c.replace(/_/g, " ")}</td>
                    {fpSizes.map((n) => <td key={n} className="p-2 text-center" style={heat(fp[n].by_condition[c].top1)}>{pct(fp[n].by_condition[c].top1, 0)} / {pct(fp[n].by_condition[c].accepted_correct, 0)}</td>)}
                  </tr>
                ))}
                <tr className="border-t border-border/40"><td className="p-2">false accepts: unknown recordings</td>{fpSizes.map((n) => <td key={n} className="p-2 text-center">{pct(fp[n].false_accept.out_of_catalog_recording, 1)}</td>)}</tr>
                <tr className="border-t border-border/40"><td className="p-2">false accepts: hummed queries</td>{fpSizes.map((n) => <td key={n} className="p-2 text-center">{pct(fp[n].false_accept.hummed_query, 1)}</td>)}</tr>
                <tr className="border-t border-border/40"><td className="p-2">latency (mean / p95)</td>{fpSizes.map((n) => <td key={n} className="p-2 text-center">{fixed(fp[n].latency.mean_ms, 0)} / {fixed(fp[n].latency.p95_ms, 0)} ms</td>)}</tr>
                <tr className="border-t border-border/40"><td className="p-2">index size</td>{fpSizes.map((n) => <td key={n} className="p-2 text-center">{fixed(fp[n].index.bytes / 1e6, 0)} MB · {fixed(fp[n].index.mb_per_audio_hour, 1)} MB/audio-hour</td>)}</tr>
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

function ModelsTab() {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const act = useMutation({ mutationFn: api.activateNeural, onSuccess: () => { qc.invalidateQueries({ queryKey: ["models"] }); qc.invalidateQueries({ queryKey: ["health"] }); } });
  const d = models.data;
  if (!d) return <Skeleton className="h-64" />;
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Card>
        <CardHeader title="Neural checkpoints" subtitle={d.neural.active ? `Active: ${d.neural.active.checkpoint} (${d.neural.active.device})` : "No neural model loaded — classical algorithms only"} icon={<Brain className="h-4 w-4" />} />
        <ul className="space-y-2 p-5">
          {d.neural.checkpoints.map((c) => (
            <li key={c.id} className="flex items-center justify-between gap-3 rounded-xl border border-border/50 px-3 py-2 text-sm">
              <span className="truncate font-mono text-xs">{c.id}</span>
              <span className="flex items-center gap-2">
                <Badge>{c.source}</Badge>
                {d.neural.active?.checkpoint === c.id ? <Badge tone="ok">active</Badge> : <Button size="sm" variant="outline" loading={act.isPending && act.variables === c.id} onClick={() => act.mutate(c.id)}>Load</Button>}
              </span>
            </li>
          ))}
          {!d.neural.checkpoints.length && <li className="text-sm text-muted">No checkpoints yet. Train one in the Training tab or with <code>make train</code>.</li>}
        </ul>
        {act.isError && <div className="px-5 pb-5"><ErrorBox error={act.error} /></div>}
      </Card>
      <Card>
        <CardHeader title="Inference backends" icon={<Cpu className="h-4 w-4" />} subtitle={`PyTorch device: ${d.inference_device}`} />
        <div className="space-y-4 p-5 text-sm">
          <div>
            <div className="label mb-2">Pitch extractors</div>
            {d.pitch_extractors.map((e) => <div key={e.name} className="flex justify-between py-1"><span>{e.label}</span><Badge tone={e.available ? "ok" : "neutral"}>{e.available ? "available" : "not installed"}</Badge></div>)}
          </div>
          <div>
            <div className="label mb-2">Whisper (faster-whisper) models</div>
            {d.asr.models.map((m) => <div key={m.name} className="flex justify-between py-1"><span>{m.name}</span><Badge tone={m.cached ? "ok" : "neutral"}>{m.cached ? "downloaded" : "not downloaded"}</Badge></div>)}
            <p className="mt-2 text-xs text-muted">{d.asr.note}</p>
          </div>
        </div>
      </Card>
    </div>
  );
}

export default function LabPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">AI Research Lab</h1>
        <p className="text-sm text-muted">Benchmark, compare and train recognition models. All numbers come from runs stored in <code className="font-mono text-xs">experiments/</code>.</p>
      </div>
      <Tabs defaultValue="bench">
        <TabsList>
          <TabsTrigger value="bench">Benchmarks</TabsTrigger>
          <TabsTrigger value="scale">Scale</TabsTrigger>
          <TabsTrigger value="compare">Compare algorithms</TabsTrigger>
          <TabsTrigger value="train">Training</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
        </TabsList>
        <TabsContent value="bench" className="mt-5"><BenchmarksTab /></TabsContent>
        <TabsContent value="scale" className="mt-5"><ScaleTab /></TabsContent>
        <TabsContent value="compare" className="mt-5"><CompareTab /></TabsContent>
        <TabsContent value="train" className="mt-5"><TrainingTab /></TabsContent>
        <TabsContent value="models" className="mt-5"><ModelsTab /></TabsContent>
      </Tabs>
    </div>
  );
}
