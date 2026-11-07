"""End-to-end: a fresh store, the demo catalog imported through the API job, then recognition
of a whistled mid-song passage, typed lyrics and a combined query, exactly as the web UI does it."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import ServerSettings
from app.main import create_app
from melodyai.audio.io import wav_bytes
from melodyai.demo.generator import generate_catalog
from melodyai.demo.queries import CONDITIONS, excerpt, perturb, synth_query

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    import os

    os.environ["MELODYAI_DATA_DIR"] = str(root / "data")  # demo files generated here, not in the repo
    app = create_app(ServerSettings(store_dir=str(root / "store"), load_neural_model=False, serve_frontend=False))
    with TestClient(app) as c:
        job = c.post("/api/catalog/demo").json()
        done = c.app.state.melody.jobs.wait(job["id"], 600)
        assert done["status"] == "completed", done
        assert done["result"]["added"] == 48
        yield c
    os.environ.pop("MELODYAI_DATA_DIR", None)


def test_demo_catalog_ready(client):
    h = client.get("/api/health").json()
    assert h["indexed"] == 48 and h["with_lyrics"] == 20
    assert client.get("/api/songs/demo-03/audio").status_code == 200


def test_whistled_chorus_from_the_middle(client):
    m = generate_catalog(48)[21]
    rng = np.random.default_rng(3)
    spec = CONDITIONS["whistle"]
    ex, start = excerpt(m.notes, rng, 6, 9, m.sections, "chorus")
    pn, _ = perturb(ex, spec, rng)
    r = client.post("/api/recognize", data={"mode": "auto"},
                    files={"audio": ("w.wav", wav_bytes(synth_query(pn, spec, rng), 16000), "audio/wav")}).json()
    assert r["engines"]["melody"]["profile"] == "whistle"
    assert r["results"][0]["song_id"] == "demo-21" and r["status"] == "match"
    # The chorus repeats (song form A-B-A-B), so any chorus occurrence is a correct location.
    t = r["results"][0]["match_start"]
    choruses = [sec for sec in m.sections if sec["name"] == "chorus"]
    assert any(sec["start"] - 2.0 <= t <= sec["end"] for sec in choruses), (t, start, choruses)


def test_lyrics_in_spanish_with_errors(client):
    r = client.post("/api/recognize/lyrics", data={"text": "corazon de papel mojao no me pidas que te escriva"}).json()
    assert r["results"][0]["song"]["title"] == "Corazón de Papel" and r["status"] == "match"


def test_combined_query_beats_a_too_short_hum(client):
    m = generate_catalog(48)[2]  # "The Lighthouse Keeper" (lyrics slot 1)
    rng = np.random.default_rng(11)
    pn, _ = perturb(m.notes.slice(4, 8), CONDITIONS["hum_short"], rng)
    audio = wav_bytes(synth_query(pn, CONDITIONS["hum_short"], rng), 16000)
    r = client.post("/api/recognize/combined", data={"text": "turn the lanter slow"},
                    files={"audio": ("q.wav", audio, "audio/wav")}).json()
    assert r["results"][0]["song_id"] == "demo-02"
