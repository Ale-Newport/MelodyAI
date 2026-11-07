import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { ResultsView } from "@/features/recognition/ResultsView";
import { SettingsProvider } from "@/hooks/useAppSettings";
import { TooltipProvider } from "@/components/ui/primitives";
import type { RecognitionResult } from "@/types/api";

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient();
  qc.setQueryData(["settings"], { ui_language: "en", theme: "dark" });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter>
        <SettingsProvider>
          <TooltipProvider>{ui}</TooltipProvider>
        </SettingsProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const base: RecognitionResult = {
  mode: "hum", resolved_mode: "hum", status: "no_match", message: "", score_kind: "relative",
  decision: { status: "no_match", reason: "No engine produced candidates" }, results: [], warnings: [], elapsed_s: 0.4,
  engines: { melody: { algorithm: "ensemble", status: "insufficient_melody", reason: "Not enough stable pitched content", reliability: 0, elapsed_s: 0, candidates: [] } },
};

describe("ResultsView", () => {
  it("shows the explicit no-match state and never a fake song", () => {
    wrap(<ResultsView result={base} />);
    expect(screen.getByRole("status")).toHaveTextContent("No sufficiently reliable match was found");
    expect(screen.getByText(/Not enough stable pitched content/)).toBeInTheDocument();
  });

  it("renders the top match with relative score wording", () => {
    const r: RecognitionResult = {
      ...base, status: "match",
      decision: { status: "match", confidence: "high", top_score: 0.81, margin: 0.4, thresholds: { accept: 0.5, margin: 0, uncertain: 0.4 }, threshold_source: "calibrated" },
      results: [
        { rank: 1, song_id: "a", score: 0.81, song: { id: "a", title: "Paper Boats", artist: "Ensemble", source: "MelodyAI procedural generator (seed=1)", has_audio: false }, engines: { melody: { rank: 1, score: 0.81, start_time: 12, end_time: 20 } }, match_start: 12, match_end: 20 },
        { rank: 2, song_id: "b", score: 0.3, song: { id: "b", title: "Copper Wires", artist: "Ensemble" }, engines: { melody: { rank: 2, score: 0.3 } } },
      ],
    };
    wrap(<ResultsView result={r} />);
    expect(screen.getByText("Paper Boats")).toBeInTheDocument();
    expect(screen.getByText("Relative score")).toBeInTheDocument();
    expect(screen.getByText("0:12–0:20")).toBeInTheDocument();
    expect(screen.getByText("Copper Wires")).toBeInTheDocument();
    expect(screen.queryByText("YouTube Music")).not.toBeInTheDocument(); // demo songs: no external search
  });
});

describe("Recording (clip) and licensed-provider results", () => {
  it("explains speed/pitch edits and shows provider matches with links", () => {
    const r: RecognitionResult = {
      ...base, mode: "clip", resolved_mode: "clip", status: "no_match",
      decision: { status: "no_match" },
      results: [],
      external: [{
        provider: "acrcloud", kind: "music", status: "match", message: "", elapsed_s: 0.8, sent_seconds: 12,
        matches: [{ title: "Hello", artists: ["Adele"], album: "25", release_date: "2015-10-23", score: 100,
          score_kind: "match confidence 70-100 (provider)", play_offset_s: 9, duration_s: 295, isrc: null,
          links: { spotify: "https://open.spotify.com/track/x" } }],
      }],
    };
    wrap(<ResultsView result={r} />);
    expect(screen.getByText("Found in a licensed catalog")).toBeInTheDocument();
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Spotify/ })).toHaveAttribute("href", "https://open.spotify.com/track/x");
  });

  it("describes a sped-up clip", () => {
    const r: RecognitionResult = {
      ...base, mode: "clip", resolved_mode: "clip", status: "match",
      decision: { status: "match", confidence: "high", top_score: 0.9, margin: 0.5 },
      results: [{ rank: 1, song_id: "a", score: 0.9, song: { id: "a", title: "Paper Boats", artist: "E", has_audio: false },
        engines: { fingerprint: { rank: 1, score: 0.9, speed: 1.25, pitch_semitones: 3.9, start_time: 10, end_time: 20 } },
        match_start: 10, match_end: 20 }],
    };
    wrap(<ResultsView result={r} />);
    expect(screen.getByText(/1.25× faster/)).toBeInTheDocument();
    expect(screen.getByText(/\+3.9 semitones/)).toBeInTheDocument();
  });
});
