import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { ResultsView } from "@/features/recognition/ResultsView";
import SettingsPage from "@/pages/SettingsPage";
import { SettingsProvider } from "@/hooks/useAppSettings";
import { TooltipProvider } from "@/components/ui/primitives";
import { stopPreview } from "@/lib/preview";
import { api } from "@/services/api";
import type { RecognitionResult, ResultItem, UserSettings, WorldCandidate, WorldEngineResult } from "@/types/api";

// All lyrics below are invented for the tests.

function wrap(ui: React.ReactNode, settings: Partial<UserSettings> = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  qc.setQueryData(["settings"], { ui_language: "es", theme: "dark", online_search_enabled: true, ...settings });
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
  mode: "auto", resolved_mode: "lyrics", status: "no_match", message: "", score_kind: "relative",
  decision: { status: "no_match" }, results: [], warnings: [], elapsed_s: 2.4, engines: {},
};

function candidate(n: number, extra: Partial<WorldCandidate> = {}): WorldCandidate {
  return {
    song_id: `world:w${n}`, title: `Canción ${n}`, artist: `Artista ${n}`, album: null, language: "es", duration_s: 200,
    cover_url: `https://cdn.example/${n}.jpg`, preview_url: `https://cdn.example/${n}.mp3`, rank: 500000 - n,
    links: { deezer: `https://www.deezer.com/track/${n}` }, score: 0.7 - n * 0.05, coverage: 0.8, char_similarity: 0.8,
    matched_text: `farol ${n} de la bahía`, line_range: [1, 1], sources: ["lrclib"], ...extra,
  };
}

function worldEngine(extra: Partial<WorldEngineResult> = {}): WorldEngineResult {
  return {
    status: "ok", query_text: "farolas de cristal sobre la bahía", online: true, elapsed_s: 2.1, message: "",
    sources: {
      index: { status: "ok", n: 3, elapsed_s: 0.05, message: "" },
      lrclib: { status: "ok", n: 2, elapsed_s: 1.2, message: "" },
      deezer: { status: "error", n: 0, elapsed_s: 8, message: "timeout" },
      genius: { status: "unavailable", n: 0, elapsed_s: 0, message: "" },
    },
    candidates: [], ...extra,
  };
}

const worldTop: ResultItem = {
  rank: 1, song_id: "world:wabc", score: 0.81,
  song: {
    id: "world:wabc", title: "Farolas de cristal", artist: "Los Inventados", album: "Bahía", language: "es",
    reference_type: "online", source: "LRCLIB + Deezer", duration_s: 213, has_audio: false,
    cover_url: "https://cdn.example/cover.jpg", preview_url: "https://cdn.example/preview.mp3",
    links: {
      deezer: "https://www.deezer.com/track/42",
      spotify: "https://open.spotify.com/search/Farolas%20de%20cristal",
      youtube: "https://www.youtube.com/results?search_query=Farolas+de+cristal",
    },
  },
  engines: { world: { rank: 1, score: 0.81, coverage: 0.86, matched_text: "farolas de cristal / sobre la bahía", line_range: [12, 13], sources: ["index", "lrclib"] } },
};

const transcript = {
  text: "farolas de cristal sobre la bahía", language: "es", language_probability: 0.97, speech_detected: true,
  reason: "", model: "large-v3-turbo", elapsed_s: 1.4,
};

const onlineMatch: RecognitionResult = {
  ...base, resolved_mode: "lyrics_online", status: "match",
  decision: { status: "match", confidence: "high", top_score: 0.81, margin: 0.3 },
  results: [worldTop, { ...worldTop, rank: 2, song_id: "world:wdef", score: 0.4,
    song: { ...worldTop.song, id: "world:wdef", title: "Otra farola", preview_url: "https://cdn.example/other.mp3", cover_url: null },
    engines: { world: { rank: 2, score: 0.4, sources: ["deezer"] } } }],
  engines: {
    lyrics: { source: "asr", status: "ok", transcript, candidates: [] },
    world: worldEngine({ candidates: [candidate(1)] }),
  },
};

let play: ReturnType<typeof vi.spyOn>;
let pause: ReturnType<typeof vi.spyOn>;
beforeEach(() => {
  play = vi.spyOn(HTMLMediaElement.prototype, "play").mockImplementation(() => Promise.resolve());
  pause = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => undefined);
});
afterEach(() => {
  stopPreview();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("Open-world (online lyrics) results", () => {
  it("shows cover, badge, matched lines, what Whisper understood, links and sources", () => {
    const { container } = wrap(<ResultsView result={onlineMatch} />);
    expect(screen.getByRole("heading", { name: "Farolas de cristal" })).toBeInTheDocument();
    expect(container.querySelector('img[src="https://cdn.example/cover.jpg"]')).not.toBeNull();
    expect(screen.getAllByText(/Letra · online/)[0]).toHaveTextContent("Letra · online 0.81");
    expect(screen.getByText("Te he entendido: «farolas de cristal sobre la bahía»")).toBeInTheDocument();
    expect(screen.getByText(/large-v3-turbo · es/)).toBeInTheDocument();
    const quote = container.querySelector("blockquote")!;
    expect(quote).toHaveTextContent("farolas de cristal");
    expect(quote).toHaveTextContent("sobre la bahía");
    expect(quote.querySelectorAll("br")).toHaveLength(1);
    expect(screen.getByText("encontrada en índice local + LRCLIB")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Deezer/ })).toHaveAttribute("href", "https://www.deezer.com/track/42");
    expect(screen.getByRole("link", { name: /Spotify/ })).toHaveAttribute("href", worldTop.song.links!.spotify);
    expect(screen.getByRole("link", { name: /YouTube/ })).toHaveAttribute("href", worldTop.song.links!.youtube);
    const sources = screen.getByText(/Búsqueda online:/);
    expect(sources).toHaveTextContent("índice local (3)");
    expect(sources).toHaveTextContent("LRCLIB (2)");
    expect(sources).toHaveTextContent("Deezer ✗");
    expect(sources).not.toHaveTextContent("Genius"); // unavailable sources did not answer
    // The online winner is already the answer: no "also found online" card.
    expect(screen.queryByText("También encontrado online")).not.toBeInTheDocument();
    expect(play).not.toHaveBeenCalled(); // never autoplays
  });

  it("falls back to the generated artwork when the cover fails to load", () => {
    const { container } = wrap(<ResultsView result={onlineMatch} />);
    const img = container.querySelector('img[src="https://cdn.example/cover.jpg"]')!;
    fireEvent.error(img);
    expect(container.querySelector('img[src="https://cdn.example/cover.jpg"]')).toBeNull();
  });

  it("plays the 30 s preview on click and stops it when another preview starts", () => {
    wrap(<ResultsView result={onlineMatch} />);
    const main = screen.getByRole("button", { name: /Escuchar 30 s/ });
    fireEvent.click(main);
    expect(play).toHaveBeenCalledTimes(1);
    expect(main).toHaveAttribute("aria-pressed", "true");
    expect(main).toHaveTextContent("Pausar");

    const other = screen.getByRole("button", { name: "Avance de 30 segundos de Otra farola" });
    fireEvent.click(other);
    expect(pause).toHaveBeenCalledTimes(1);
    expect(play).toHaveBeenCalledTimes(2);
    expect(other).toHaveAttribute("aria-pressed", "true");
    expect(main).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(other); // pause
    expect(other).toHaveAttribute("aria-pressed", "false");
  });

  it("only links to http(s) URLs", () => {
    const bad: RecognitionResult = {
      ...onlineMatch,
      results: [{ ...worldTop, song: { ...worldTop.song, links: { deezer: "javascript:alert(1)" }, preview_url: "data:audio/mp3;base64,AAAA" } }],
    };
    wrap(<ResultsView result={bad} />);
    expect(screen.queryByRole("link", { name: /Deezer/ })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Spotify/ }).getAttribute("href")).toMatch(/^https:\/\/open\.spotify\.com\/search\//);
    expect(screen.queryByRole("button", { name: /Escuchar 30 s/ })).not.toBeInTheDocument();
  });
});

describe("Also found online", () => {
  it("shows the top 3 online candidates next to a local winner without changing it", () => {
    const r: RecognitionResult = {
      ...base, resolved_mode: "hum", status: "match",
      decision: { status: "match", confidence: "high", top_score: 0.9, margin: 0.5 },
      results: [{ rank: 1, song_id: "a", score: 0.9, song: { id: "a", title: "Paper Boats", artist: "Ensemble", source: "MelodyAI procedural generator (seed=1)", has_audio: false },
        engines: { melody: { rank: 1, score: 0.9 } } }],
      engines: { world: worldEngine({ candidates: [candidate(1), candidate(2), candidate(3), candidate(4)] }) },
    };
    wrap(<ResultsView result={r} />, { ui_language: "en" });
    expect(screen.getByRole("heading", { name: "Paper Boats" })).toBeInTheDocument();
    expect(screen.getByText(/Melody 0.90/)).toBeInTheDocument();
    expect(screen.queryByText(/Lyrics · online/)).not.toBeInTheDocument(); // the local winner is not an online song
    const card = screen.getByText("Also found online").closest(".card") as HTMLElement;
    expect(within(card).getByText("Canción 1")).toBeInTheDocument();
    expect(within(card).getByText("Canción 3")).toBeInTheDocument();
    expect(within(card).queryByText("Canción 4")).not.toBeInTheDocument();
    expect(within(card).getAllByRole("link", { name: /Deezer/ })[0]).toHaveAttribute("href", "https://www.deezer.com/track/1");
    expect(within(card).getByRole("button", { name: "30-second preview of Canción 2" })).toBeInTheDocument();
  });

  it("calls them possible matches when nothing passed the threshold", () => {
    const r: RecognitionResult = { ...base, engines: { world: worldEngine({ candidates: [candidate(1)] }) } };
    wrap(<ResultsView result={r} />);
    expect(screen.getByText("Posibles coincidencias online")).toBeInTheDocument();
    expect(screen.getByText("Canción 1")).toBeInTheDocument();
  });
});

describe("No-match hints", () => {
  it("points to Settings when online search is disabled", () => {
    const r: RecognitionResult = { ...base, engines: { world: worldEngine({ status: "disabled", online: false, sources: {} }) } };
    wrap(<ResultsView result={r} />, { online_search_enabled: false });
    expect(screen.getByText(/La búsqueda online de canciones está desactivada/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Ir a Ajustes" })).toHaveAttribute("href", "/settings");
    expect(screen.queryByText(/Búsqueda online:/)).not.toBeInTheDocument();
  });

  it("shows the hint when the setting is off even without a world result", () => {
    wrap(<ResultsView result={base} />, { online_search_enabled: false, ui_language: "en" });
    expect(screen.getByText(/Online song search is off/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open Settings" })).toBeInTheDocument();
  });

  it("does not show the hint when online search ran", () => {
    wrap(<ResultsView result={{ ...base, engines: { world: worldEngine({ status: "no_candidates" }) } }} />);
    expect(screen.queryByText(/está desactivada/)).not.toBeInTheDocument();
    expect(screen.getByText(/Ninguna canción conocida coincide con «farolas de cristal sobre la bahía»/)).toBeInTheDocument();
  });

  it("suggests singing the words when Whisper heard none", () => {
    const r: RecognitionResult = {
      ...base,
      engines: { lyrics: { source: "asr", status: "no_speech", candidates: [], transcript: { ...transcript, text: "", speech_detected: false, reason: "no speech" } } },
    };
    wrap(<ResultsView result={r} />);
    expect(screen.getByText("No he distinguido ninguna palabra.")).toBeInTheDocument();
    expect(screen.getByText(/Canta la letra que recuerdes/)).toBeInTheDocument();
  });
});

function mockServer(extra: Record<string, unknown> = {}, failures: Record<string, [number, string]> = {}) {
  const settings = { ui_language: "es", theme: "dark", online_search_enabled: true, asr_model: "auto" };
  const calls: { url: string; method: string; body: any }[] = [];
  const routes: Record<string, unknown> = {
    "/api/models": {
      pitch_extractors: [], melody_algorithms: [], default_algorithm: "ensemble", fusion_methods: [], inference_device: "cpu",
      asr: { package_installed: true, active: "auto", downloads_allowed: false, note: "",
        models: [{ name: "base", cached: true, installed_package: true }, { name: "large-v3-turbo", cached: true, installed_package: true }] },
      neural: { active: null, error: null, checkpoints: [] }, providers: [], providers_enabled: false,
    },
    "/api/health": { status: "ok", version: "test", songs: 0, indexed: 0, with_lyrics: 0, asr_ready: true, neural_model: null, device: "cpu", ffmpeg: true, limits: { max_upload_mb: 25, max_query_seconds: 60 } },
    "/api/catalog/status": { songs: 48, feature_version: "v1" },
    "/api/jobs?kind=world_sync": [],
    "/api/world/status": {
      songs: 12345, online_enabled: true, db_bytes: 52_400_000,
      last_sync: { preset: "quick", finished_at: 1_762_500_000, added: 1400, collected: 1500 },
      sources: [
        { name: "index", label: "Local index", online: false, requires_key: false, available: true },
        { name: "lrclib", label: "LRCLIB", online: true, requires_key: false, available: true },
        { name: "genius", label: "Genius", online: true, requires_key: true, available: false },
      ],
      presets: [
        { name: "quick", label: "Quick", approx_songs: 1500, approx_minutes: 20 },
        { name: "standard", label: "Standard", approx_songs: 6000, approx_minutes: 75 },
        { name: "large", label: "Large", approx_songs: 25000, approx_minutes: 300 },
      ],
    },
    "POST /api/world/sync": { id: "j1", kind: "world_sync", status: "queued", progress: 0, message: null, config: {}, result: null, error: null, created_at: null, started_at: null, finished_at: null },
    "/api/jobs/j1": { id: "j1", kind: "world_sync", status: "running", progress: 0.08, message: "Lyrics 120/1500 · +12 · ETA 30 min", config: {}, result: null, error: null, created_at: null, started_at: null, finished_at: null },
    ...extra,
  };
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    calls.push({ url, method, body });
    const route = method === "GET" ? url : `${method} ${url}`;
    if (failures[route]) return new Response(JSON.stringify({ detail: failures[route][1] }), { status: failures[route][0] });
    const data = method === "PUT" && url === "/api/settings" ? { ...settings, ...body } : routes[route];
    if (data === undefined) return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, settings };
}

describe("Settings: online search and popular-songs sync", () => {
  it("shows the index, sources and presets, starts a sync and follows its progress", async () => {
    const { calls } = mockServer();
    wrap(<SettingsPage />, { asr_model: "auto" });
    expect(await screen.findByText(/12[.,\s\u00a0\u202f]?345/)).toBeInTheDocument();
    expect(screen.getByText("52,4 MB en disco")).toBeInTheDocument();
    expect(screen.getByText("Canciones indexadas en local")).toBeInTheDocument();
    expect(screen.getByText(/Rápida · .* · \+1400 canciones/)).toBeInTheDocument();
    expect(screen.getByText("Índice local · local")).toBeInTheDocument();
    expect(screen.getByText("LRCLIB · activa")).toBeInTheDocument();
    expect(screen.getByText("Genius · necesita clave")).toBeInTheDocument();
    expect(screen.getByText(/solo se envían a LRCLIB y Deezer las palabras transcritas, nunca tu audio/)).toBeInTheDocument();
    expect(screen.getByText("Usa el mejor modelo ya descargado: large-v3-turbo.")).toBeInTheDocument();
    expect(screen.getByText("~25.000 canciones · ~5 h")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Rápida/ }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url === "/api/world/sync")).toBe(true));
    expect(calls.find((c) => c.url === "/api/world/sync")!.body).toEqual({ preset: "quick", limit: null });
    // the job's status and progress are shown in the UI language
    expect(await screen.findByText("Letras 120/1500 · +12 · quedan unos 30 min")).toBeInTheDocument();
    expect(screen.getByText("en curso")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Estándar/ })).toBeDisabled();
  });

  it("saves the online search toggle", async () => {
    const { calls } = mockServer();
    wrap(<SettingsPage />);
    const toggle = screen.getByRole("switch", { name: "Búsqueda online de canciones" });
    expect(toggle).toHaveAttribute("aria-checked", "true");
    fireEvent.click(toggle);
    await waitFor(() => expect(calls.some((c) => c.method === "PUT" && c.body?.online_search_enabled === false)).toBe(true));
  });

  it("explains when the server has no world endpoints", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Not Found" }), { status: 404 })));
    wrap(<SettingsPage />);
    expect(await screen.findByText(/este servidor todavía no ofrece la búsqueda online/)).toBeInTheDocument();
  });
});

describe("World API client", () => {
  it("encodes the search text and sends sync presets as JSON", async () => {
    const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await api.worldSearch("la niña & el mar", "es");
    // POST + JSON: the words stay out of URLs (and access logs), and another site cannot send it unpreflighted
    expect(fetchMock.mock.calls[0][0]).toBe("/api/world/search");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({ q: "la niña & el mar", language: "es", top_k: 5 });
    expect(api.worldPreviewUrl(42)).toBe("/api/world/preview/42");
    await api.startWorldSync("standard", 200);
    expect(fetchMock.mock.calls[1][1]?.method).toBe("POST");
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toEqual({ preset: "standard", limit: 200 });
    await api.worldSongs("", 50, 25);
    expect(fetchMock.mock.calls[2][0]).toBe("/api/world/songs?offset=50&limit=25");
  });
});


describe("Review fixes: contract, privacy and translations", () => {
  it("does not list internal steps (lyrics lookups) as sources", () => {
    const world = worldEngine({ candidates: [candidate(1)], sources: { ...worldEngine().sources, lyrics_fetch: { status: "ok", n: 1, elapsed_s: 0.4, message: "" } } });
    wrap(<ResultsView result={{ ...onlineMatch, engines: { ...onlineMatch.engines, world } }} />, { ui_language: "en" });
    const sources = screen.getByText(/Online search:/);
    expect(sources).toHaveTextContent("LRCLIB (2)");
    expect(sources).not.toHaveTextContent("lyrics_fetch");
  });

  it("tells an offline server from the setting, and never claims an online search that did not happen", () => {
    const offline = worldEngine({
      status: "no_candidates", online: false, offline_reason: "server_offline",
      sources: { index: { status: "ok", n: 0, elapsed_s: 0.01, message: "" }, lrclib: { status: "skipped", n: 0, elapsed_s: 0, message: "server offline" } },
    });
    wrap(<ResultsView result={{ ...base, engines: { world: offline } }} />, { ui_language: "en" });
    expect(screen.getByText(/This server runs offline/)).toBeInTheDocument();
    expect(screen.queryByText(/No known song matched/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Online song search is off/)).not.toBeInTheDocument();
    expect(screen.getByText(/Searched offline:/)).toHaveTextContent("local index (0)");
  });

  it("says online search was off for that result only, without the 'no known song' hint", () => {
    const off = worldEngine({ status: "no_candidates", online: false, offline_reason: "setting_off", sources: {} });
    wrap(<ResultsView result={{ ...base, engines: { world: off } }} />, { ui_language: "en", online_search_enabled: true });
    expect(screen.getByText(/Online song search is off/)).toBeInTheDocument();
    expect(screen.queryByText(/No known song matched/)).not.toBeInTheDocument();
    // a history result that did search online is not "off" because the setting is off now
    const ran = worldEngine({ status: "no_candidates" });
    wrap(<ResultsView result={{ ...base, engines: { world: ran } }} />, { ui_language: "en", online_search_enabled: false });
    expect(screen.getAllByText(/Online song search is off/)).toHaveLength(1);
  });

  it("translates the error of a failed online search", () => {
    wrap(<ResultsView result={{ ...base, engines: { world: worldEngine({ status: "error", message: "timed out" }) } }} />);
    expect(screen.getByText("La búsqueda online ha fallado: ha tardado demasiado")).toBeInTheDocument();
  });

  it("names the open-world mode in the UI language", () => {
    wrap(<ResultsView result={onlineMatch} />);
    expect(screen.getByText(/lyrics_online · letra \(búsqueda abierta\)/)).toBeInTheDocument();
  });

  it("does not load remote cover art when online search is off", () => {
    const { container } = wrap(<ResultsView result={onlineMatch} />, { online_search_enabled: false });
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByRole("heading", { name: "Farolas de cristal" })).toBeInTheDocument();
  });

  it("plays previews through the server by Deezer id and says when one is unavailable", async () => {
    const withId: RecognitionResult = { ...onlineMatch, results: [{ ...worldTop, song: { ...worldTop.song, deezer_id: 42 } }] };
    play.mockImplementation(function (this: HTMLMediaElement) {
      return Promise.reject(Object.assign(new Error("403"), { name: "NotSupportedError" }));
    });
    wrap(<ResultsView result={withId} />, { ui_language: "en" });
    fireEvent.click(screen.getByRole("button", { name: /Play 30 s/ }));
    expect((play.mock.contexts[0] as HTMLMediaElement).src).toMatch(/\/api\/world\/preview\/42$/);
    expect(await screen.findByText("Preview unavailable")).toBeInTheDocument();
  });

  it("shows sources by name, 'off' while online search is off, and the last sync's status", async () => {
    mockServer({
      "/api/world/status": {
        songs: 10, online_enabled: false, server_online: true, db_bytes: 1000, presets: [],
        last_sync: { preset: "large", status: "cancelled", finished_at: 1_762_500_000, added: 30, collected: 25000 },
        sources: [
          { name: "index", label: "Local lyrics index", online: false, requires_key: false, available: true },
          { name: "lrclib", label: "LRCLIB (title search)", online: true, requires_key: false, available: true },
          { name: "deezer", label: "Deezer (sung lines and titles)", online: true, requires_key: false, available: true },
        ],
      },
    });
    wrap(<SettingsPage />, { online_search_enabled: false });
    expect(await screen.findByText("LRCLIB · desactivada")).toBeInTheDocument();
    expect(screen.getByText("Deezer · desactivada")).toBeInTheDocument();
    expect(screen.getByText("Índice local · local")).toBeInTheDocument();
    expect(screen.queryByText(/búsqueda por título|title search/)).not.toBeInTheDocument();
    expect(screen.getByText("detenida antes de terminar")).toBeInTheDocument();
  });

  it("blocks presets while another process syncs and translates refusals", async () => {
    mockServer({
      "/api/world/status": {
        songs: 10, online_enabled: true, server_online: true, db_bytes: 1000, last_sync: null, sources: [],
        presets: [{ name: "quick", label: "Quick", approx_songs: 1500, approx_minutes: 20 }],
        running_sync: { preset: "standard", started_at: 1, updated_at: 2, total: 1900, done: 120 },
      },
    });
    wrap(<SettingsPage />, { ui_language: "en" });
    expect(await screen.findByText(/already running in another process .*120\/1,900/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Quick/ })).toBeDisabled();
  });

  it("explains a 409 from the server in the UI language", async () => {
    mockServer({}, { "POST /api/world/sync": [409, "A popular-songs sync is already running"] });
    wrap(<SettingsPage />);
    fireEvent.click(await screen.findByRole("button", { name: /Rápida/ }));
    expect(await screen.findByText("Ya hay una sincronización de canciones populares en marcha.")).toBeInTheDocument();
  });

  it("offers the recommended Whisper model when 'auto' has nothing downloaded", async () => {
    const { calls } = mockServer({
      "/api/models": {
        pitch_extractors: [], melody_algorithms: [], default_algorithm: "ensemble", fusion_methods: [], inference_device: "cpu",
        asr: { package_installed: true, active: "auto", downloads_allowed: true, note: "",
          auto: { name: "auto", installed_package: true, cached: false, resolved: null },
          models: [{ name: "base", cached: false, installed_package: true }] },
        neural: { active: null, error: null, checkpoints: [] }, providers: [], providers_enabled: false,
      },
      "POST /api/models/asr/download": { id: "d1", kind: "asr_download", status: "queued", progress: 0, message: null, config: {}, result: null, error: null, created_at: null, started_at: null, finished_at: null },
    });
    wrap(<SettingsPage />, { asr_model: "auto", ui_language: "en" });
    const button = await screen.findByRole("button", { name: "Download large-v3-turbo" });
    fireEvent.click(button);
    await waitFor(() => expect(calls.find((c) => c.url === "/api/models/asr/download")?.body).toEqual({ model: "auto" }));
  });
});
