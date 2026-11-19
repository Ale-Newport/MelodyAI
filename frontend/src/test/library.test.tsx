import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import LibraryPage from "@/pages/LibraryPage";
import { SettingsProvider, useAppSettings } from "@/hooks/useAppSettings";
import { TooltipProvider } from "@/components/ui/primitives";
import { stopPreview, usePreviewActive } from "@/lib/preview";
import { api } from "@/services/api";
import type { Song } from "@/types/api";

function wrap(ui: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
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

const song = {
  id: "demo-01", title: "Paper Boats", artist: "MelodyAI Folk Machine", album: null, year: 2026, language: "en",
  reference_type: "midi", source: "demo", license: "CC0", duration_s: 48, has_audio: true, content_hash: "x",
  index_status: "indexed", has_lyrics: false, extra: {},
} as unknown as Song;

let play: ReturnType<typeof vi.spyOn>;
let pause: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  pause = vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => undefined);
  vi.spyOn(api, "songs").mockResolvedValue({ items: [song], total: 1 } as never);
  vi.spyOn(api, "catalogStatus").mockResolvedValue({
    songs: 1, by_status: { indexed: 1 }, with_lyrics: 0, feature_version: "1",
    melody_index: { songs: 1, windows: 1, window_lengths: [8], version: 1 }, lyrics_index: { songs: 0 },
    neural_index: null, stats: {},
  } as never);
  vi.spyOn(api, "duplicates").mockResolvedValue([]);
});

afterEach(() => {
  stopPreview();
  vi.restoreAllMocks();
});

describe("Library", () => {
  it("stops the song it is playing when the page is left (#11)", async () => {
    const view = wrap(<LibraryPage />);
    fireEvent.click(await screen.findByRole("button", { name: "Play Paper Boats" }));
    expect(play).toHaveBeenCalledTimes(1);
    pause.mockClear();
    view.unmount();
    expect(pause).toHaveBeenCalled();
  });

  it("does not stop a preview another page started", async () => {
    function Probe() {
      return <span data-testid="active">{usePreviewActive() ?? "none"}</span>;
    }
    const { togglePreview } = await import("@/lib/preview");
    act(() => togglePreview("practice:demo-02", "/api/songs/demo-02/audio"));
    const view = wrap(<><LibraryPage /><Probe /></>);
    await screen.findByRole("button", { name: "Play Paper Boats" });
    view.unmount();
    const probe = wrap(<Probe />);
    expect(probe.getByTestId("active").textContent).toBe("practice:demo-02");
  });

  it("starts the next song with an empty form, and sends MIDI files as 'midi' (#12)", async () => {
    const add = vi.spyOn(api, "addSong").mockResolvedValue(song);
    wrap(<LibraryPage />);
    await screen.findByRole("button", { name: "Play Paper Boats" });

    const fill = (title: string, ref: File, lyrics?: File) => {
      const inputs = () => Array.from(document.querySelectorAll<HTMLInputElement>('input[type="file"]'));
      fireEvent.change(inputs()[0], { target: { files: [ref] } });
      fireEvent.change(screen.getByLabelText("Title"), { target: { value: title } });
      if (lyrics) fireEvent.change(inputs()[1], { target: { files: [lyrics] } });
      // jsdom's constraint validation cannot see files set this way, so submit the form directly.
      fireEvent.submit(screen.getByRole("button", { name: "Import & index" }).closest("form")!);
    };

    fireEvent.click(screen.getByRole("button", { name: /add/i }));
    fill("First", new File(["MThd"], "first.mid"), new File(["la la"], "first.txt"));
    await waitFor(() => expect(add).toHaveBeenCalledTimes(1));
    const first = add.mock.calls[0][0] as FormData;
    expect(first.get("reference_type")).toBe("midi");
    expect((first.get("lyrics_file") as File).name).toBe("first.txt");
    await waitFor(() => expect(screen.queryByRole("button", { name: "Import & index" })).not.toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /add/i }));
    expect((screen.getByLabelText("Title") as HTMLInputElement).value).toBe("");
    fill("Second", new File(["RIFF"], "second.wav"));
    await waitFor(() => expect(add).toHaveBeenCalledTimes(2));
    const second = add.mock.calls[1][0] as FormData;
    expect(second.get("lyrics_file")).toBeNull();
    expect(second.get("title")).toBe("Second");
    expect(second.get("reference_type")).toBe("monophonic_audio");
  });
});

describe("Microphone selection (#13)", () => {
  it("applies a newly chosen microphone immediately", async () => {
    vi.spyOn(api, "saveSettings").mockImplementation(async (p) => ({ ui_language: "en", ...p }) as never);
    let ctx: ReturnType<typeof useAppSettings> | null = null;
    function Probe() {
      ctx = useAppSettings();
      return <span data-testid="mic">{ctx.settings.microphone_id ?? "default"}</span>;
    }
    wrap(<Probe />);
    expect(screen.getByTestId("mic").textContent).toBe("default");
    await act(async () => {
      await ctx!.update({ microphone_id: "usb-mic" });
    });
    expect(screen.getByTestId("mic").textContent).toBe("usb-mic");
  });
});
