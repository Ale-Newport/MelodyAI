"""API input validation: pitch extractors (#17), bulk import limits (#16) and user-facing import errors."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import lab, recognition
from app.core.config import ServerSettings
from app.main import create_app
from melodyai.audio.io import wav_bytes
from melodyai.catalog.importers import notes_to_midi_bytes
from melodyai.demo.generator import generate_catalog
from melodyai.demo.queries import CONDITIONS, perturb, synth_query


@pytest.fixture(scope="module")
def melodies():
    return generate_catalog(3, base_seed=2000)


@pytest.fixture()
def client(tmp_path, melodies):
    app = create_app(ServerSettings(store_dir=str(tmp_path / "store"), load_neural_model=False, serve_frontend=False,
                                    max_bulk_mb=0.05))
    with TestClient(app) as c:
        for i, m in enumerate(melodies):
            r = c.post("/api/songs", files={"reference": (f"s{i}.mid", notes_to_midi_bytes(m.notes), "audio/midi")},
                       data={"title": f"Song {i}", "license": "CC0-1.0"})
            assert r.status_code == 201, r.text
        yield c


def hum(m, seed=0):
    rng = np.random.default_rng(seed)
    pn, _ = perturb(m.notes.slice(6, 20), CONDITIONS["hum_clean"], rng)
    return {"audio": ("q.wav", wav_bytes(synth_query(pn, CONDITIONS["hum_clean"], rng), 16000), "audio/wav")}


@pytest.fixture()
def only_pyin(monkeypatch):
    """A base install without the [pitch] extra: CREPE and Basic Pitch are not available."""
    for module in (lab, recognition):
        monkeypatch.setattr(module, "available_extractors", lambda: {"pyin"})


def test_settings_refuse_unavailable_or_unknown_extractors(client, only_pyin):
    for name in ("crepe", "no-such-extractor"):
        r = client.put("/api/settings", json={"pitch_extractor": name})
        assert r.status_code == 422 and "not available" in r.json()["detail"]
    assert client.get("/api/settings").json()["pitch_extractor"] == "pyin"


def test_recognize_with_an_unavailable_extractor_is_a_422(client, melodies, only_pyin):
    r = client.post("/api/recognize/hum", files=hum(melodies[0]), data={"pitch_extractor": "crepe"})
    assert r.status_code == 422 and "not available" in r.json()["detail"]


def test_saved_extractor_that_is_no_longer_installed_falls_back_to_pyin(client, melodies, monkeypatch):
    assert client.put("/api/settings", json={"pitch_extractor": "pyin"}).status_code == 200
    state = client.app.state.melody
    state.user_settings.update({"pitch_extractor": "crepe"})        # saved while the [pitch] extra was installed
    state.apply_user_settings()
    for module in (lab, recognition):
        monkeypatch.setattr(module, "available_extractors", lambda: {"pyin"})
    r = client.post("/api/recognize/hum", files=hum(melodies[1]))
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["song"]["title"] == "Song 1"


def test_bulk_import_total_size_is_limited(client, melodies):
    big = b"\0" * 40_000
    files = [("manifest", ("m.csv", b"title,reference_file\nA,a.mid\nB,b.mid\n", "text/csv")),
             ("files", ("a.mid", big, "audio/midi")), ("files", ("b.mid", big, "audio/midi"))]
    r = client.post("/api/catalog/import", files=files)
    assert r.status_code == 413 and "too large" in r.json()["detail"]


def test_import_errors_do_not_echo_library_exceptions(client):
    files = [("manifest", ("m.csv", b"title,reference_file\nBroken,broken.mid\n", "text/csv")),
             ("files", ("broken.mid", b"MThd\x00\x00\x00\x06garbage", "audio/midi"))]
    res = client.post("/api/catalog/import", files=files).json()["results"]
    assert res == [{"title": "Broken", "status": "error", "error": "Invalid MIDI file: it could not be parsed"}]
    r = client.post("/api/catalog/import", files=[("manifest", ("m.json", b'{"songs": [', "application/json")),
                                                  ("files", ("x.mid", b"x", "audio/midi"))])
    assert r.status_code == 422 and r.json()["detail"].startswith("Invalid manifest: Invalid manifest JSON (line 1")
    r = client.post("/api/catalog/import", files=[("manifest", ("m.csv", b"\xff\xfe\x00bad", "text/csv")),
                                                  ("files", ("x.mid", b"x", "audio/midi"))])
    assert r.status_code == 422 and "UTF-8" in r.json()["detail"]
