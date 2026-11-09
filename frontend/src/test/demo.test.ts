import { demoRequest } from "@/services/demo";
import { ApiError } from "@/services/api";

const FILES: Record<string, unknown> = {
  "/demo/api/examples.json": { hum: "h1", clip: "c1", lyrics: "l1" },
  "/demo/api/recognitions/h1.json": { id: "h1" },
  "/demo/api/recognitions/c1.json": { id: "c1" },
  "/demo/api/recognitions/l1.json": { id: "l1" },
  "/demo/api/songs.json": {
    items: [
      { id: "demo-00", title: "Paper Boats", artist: "Synth Ensemble", index_status: "indexed" },
      { id: "demo-01", title: "Paper Harbor", artist: "Folk Machine", index_status: "error" },
      { id: "demo-02", title: "Satellite", artist: "Night Choir", index_status: "indexed" },
    ],
  },
  "/demo/api/experiments/run/queries.json": [
    { algorithm: "dtw", condition: "hum_clean", target: "a", rank: 1 },
    { algorithm: "dtw", condition: "hum_noisy", target: "b", rank: 3 },
    { algorithm: "ensemble", condition: "hum_noisy", target: null, rank: null },
  ],
};

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => (url in FILES ? new Response(JSON.stringify(FILES[url])) : new Response("", { status: 404 }))),
  );
});
afterEach(() => vi.unstubAllGlobals());

const form = (fields: Record<string, string | Blob>) => {
  const fd = new FormData();
  Object.entries(fields).forEach(([k, v]) => fd.set(k, v));
  return { method: "POST", body: fd };
};

describe("static demo transport", () => {
  it("replays the example of the selected mode", async () => {
    const audio = new Blob(["x"]);
    expect(await demoRequest<{ id: string }>("/api/recognize", form({ mode: "clip", audio }))).toEqual({ id: "c1" });
    expect(await demoRequest<{ id: string }>("/api/recognize", form({ mode: "whistle", audio }))).toEqual({ id: "h1" });
    // Typed words without audio are a lyrics search, whatever the mode.
    expect(await demoRequest<{ id: string }>("/api/recognize", form({ mode: "auto", text: "paper boats" }))).toEqual({ id: "l1" });
  });

  it("filters the catalog like the server", async () => {
    const r = await demoRequest<{ items: { id: string }[]; total: number }>("/api/songs?limit=500&q=paper&status=indexed");
    expect(r.items.map((s) => s.id)).toEqual(["demo-00"]);
    expect(r.total).toBe(1);
  });

  it("filters benchmark queries and keeps only errors when asked", async () => {
    const r = await demoRequest<{ items: unknown[]; total: number }>("/api/experiments/run/queries?algorithm=dtw&only_errors=true&limit=500");
    expect(r.items).toEqual([{ algorithm: "dtw", condition: "hum_noisy", target: "b", rank: 3 }]);
  });

  it("refuses actions that need the local server", async () => {
    await expect(demoRequest("/api/catalog/demo", { method: "POST" })).rejects.toBeInstanceOf(ApiError);
    await expect(demoRequest("/api/recognize", form({ mode: "hum", algorithm: "dtw", audio: new Blob(["x"]) }))).rejects.toThrow(/static demo/);
  });
});
