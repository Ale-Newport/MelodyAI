"""FastAPI integration tests (real engine, temporary store, no network, no neural model)."""


import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import ServerSettings
from app.main import create_app
from melodyai.audio.io import wav_bytes
from melodyai.catalog.importers import notes_to_midi_bytes
from melodyai.demo.generator import generate_catalog
from melodyai.demo.lyrics_bank import LYRICS
from melodyai.demo.queries import CONDITIONS, perturb, synth_query


@pytest.fixture(scope="module")
def melodies():
    return generate_catalog(7, base_seed=1000)


@pytest.fixture(scope="module")
def client(tmp_path_factory, melodies):
    store = tmp_path_factory.mktemp("apistore")
    app = create_app(ServerSettings(store_dir=str(store), load_neural_model=False, serve_frontend=False,
                                    max_upload_mb=5))
    with TestClient(app) as c:
        for i, m in enumerate(melodies[:6]):
            files = {"reference": (f"s{i}.mid", notes_to_midi_bytes(m.notes), "audio/midi")}
            data = {"title": f"Song {i}", "artist": "API Tester", "license": "CC0-1.0"}
            if i < 3:
                data["lyrics_text"] = LYRICS[i]["text"]
            r = c.post("/api/songs", files=files, data=data)
            assert r.status_code == 201, r.text
        yield c


def hum_wav(m, seed=0, start=8, n=14, cond="hum_clean"):
    rng = np.random.default_rng(seed)
    pn, _ = perturb(m.notes.slice(start, start + n), CONDITIONS[cond], rng)
    return wav_bytes(synth_query(pn, CONDITIONS[cond], rng), 16000)


def test_health_and_models(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["songs"] == 6 and h["indexed"] == 6 and h["with_lyrics"] == 3
    m = client.get("/api/models").json()
    names = {a["name"] for a in m["melody_algorithms"]}
    assert {"dtw", "subseq_dtw", "chroma_dtw", "note_align", "ensemble"} <= names
    assert any(e["name"] == "pyin" and e["available"] for e in m["pitch_extractors"])


def test_recognize_hum(client, melodies):
    r = client.post("/api/recognize/hum", files={"audio": ("q.wav", hum_wav(melodies[4]), "audio/wav")})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["status"] == "match" and res["results"][0]["song"]["title"] == "Song 4"
    assert res["score_kind"] in ("relative", "calibrated_probability")
    assert res["query"]["pitch_track"]["midi"] and res["alignment"]["path"]
    assert res["id"]
    # stored in history, without audio
    rec = client.get(f"/api/recognitions/{res['id']}").json()
    assert rec["results"][0]["song_id"] == res["results"][0]["song_id"]
    hist = client.get("/api/recognitions").json()
    assert hist[0]["id"] == res["id"] and hist[0]["audio_retained"] is False


def test_recognize_other_algorithms(client, melodies):
    for alg in ("dtw", "note_align", "chroma_dtw"):
        r = client.post("/api/recognize", data={"mode": "hum", "algorithm": alg},
                        files={"audio": ("q.wav", hum_wav(melodies[2], seed=3), "audio/wav")})
        assert r.status_code == 200, r.text
        assert r.json()["results"][0]["song"]["title"] == "Song 2", alg
    r = client.post("/api/recognize", data={"mode": "hum", "algorithm": "nope"},
                    files={"audio": ("q.wav", hum_wav(melodies[2]), "audio/wav")})
    assert r.status_code == 422


def test_recognize_lyrics_text_and_no_match(client):
    r = client.post("/api/recognize/lyrics", data={"text": "the lighthouse keeper counts the ships at nite"}).json()
    assert r["status"] == "match" and r["results"][0]["song"]["title"] == "Song 1"
    assert "lighthouse keeper" in r["engines"]["lyrics"]["candidates"][0]["matched_text"].lower()
    r = client.post("/api/recognize/lyrics", data={"text": "dancing in the rain until the morning light"}).json()
    assert r["status"] != "match"


def test_recognize_combined(client, melodies):
    r = client.post("/api/recognize/combined", data={"text": "every street a river after rain"},
                    files={"audio": ("q.wav", hum_wav(melodies[0], 5, 0, 8), "audio/wav")}).json()
    assert r["resolved_mode"] == "combined" and r["results"][0]["song"]["title"] == "Song 0"


def test_invalid_uploads(client):
    assert client.post("/api/recognize/hum", files={"audio": ("q.wav", b"garbage" * 100, "audio/wav")}).status_code == 422
    assert client.post("/api/recognize/hum", files={"audio": ("q.wav", b"", "audio/wav")}).status_code == 400
    big = b"0" * (6 * 1024 * 1024)
    assert client.post("/api/recognize/hum", files={"audio": ("q.wav", big, "audio/wav")}).status_code == 413
    assert client.post("/api/recognize", data={"mode": "auto"}).status_code == 422
    long = wav_bytes(np.zeros(16000 * 70, dtype=np.float32), 16000)
    assert client.post("/api/recognize/hum", files={"audio": ("q.wav", long, "audio/wav")}).status_code in (413, 422)


def test_noise_is_rejected(client):
    y = (0.2 * np.random.default_rng(0).standard_normal(16000 * 4)).astype(np.float32)
    r = client.post("/api/recognize/hum", files={"audio": ("q.wav", wav_bytes(y, 16000), "audio/wav")}).json()
    assert r["status"] == "no_match" and r["message"].startswith("No sufficiently reliable")


def test_song_crud_and_listing(client, melodies):
    lst = client.get("/api/songs", params={"q": "song 3"}).json()
    assert lst["total"] == 1
    sid = lst["items"][0]["id"]
    assert client.patch(f"/api/songs/{sid}", json={"album": "Edited"}).json()["album"] == "Edited"
    assert client.patch(f"/api/songs/{sid}", json={"listen_url": "javascript:alert(1)"}).status_code == 422
    s = client.put(f"/api/songs/{sid}/lyrics", json={"text": "brand new words for testing lyrics", "language": "en"}).json()
    assert s["has_lyrics"]
    r = client.post("/api/recognize/lyrics", data={"text": "brand new words for testing"}).json()
    assert r["results"][0]["song_id"] == sid
    assert client.get(f"/api/songs/{sid}/melody").json()["notes"]["pitches"]
    assert client.get(f"/api/songs/{sid}/audio").status_code == 404  # MIDI import without preview
    assert client.post(f"/api/songs/{sid}/reindex").json()["index_status"] == "indexed"
    dup = client.post("/api/songs", files={"reference": ("x.mid", notes_to_midi_bytes(melodies[3].notes), "audio/midi")},
                      data={"title": "dup"})
    assert dup.status_code == 409
    assert client.get("/api/songs/..%2Fetc").status_code in (400, 404)


def test_G_delete_song_via_api(client, melodies):
    sid = client.get("/api/songs", params={"q": "song 5"}).json()["items"][0]["id"]
    wav = hum_wav(melodies[5], 9)
    assert client.post("/api/recognize/hum", files={"audio": ("q.wav", wav, "audio/wav")}).json()["results"][0]["song_id"] == sid
    assert client.delete(f"/api/songs/{sid}").status_code == 200
    res = client.post("/api/recognize/hum", files={"audio": ("q.wav", wav, "audio/wav")}).json()
    assert all(r["song_id"] != sid for r in res["results"])
    assert client.get(f"/api/songs/{sid}").status_code == 404


def test_bulk_import_and_reindex_job(client, melodies):
    manifest = "title,artist,reference_file,lyrics_file\nBulk Song,Bulk,bulk.mid,bulk.txt\nMissing,X,none.mid,\n"
    files = [("manifest", ("metadata.csv", manifest.encode(), "text/csv")),
             ("files", ("bulk.mid", notes_to_midi_bytes(melodies[6].notes), "audio/midi")),
             ("files", ("bulk.txt", b"some bulk imported words\nsecond line here", "text/plain"))]
    res = client.post("/api/catalog/import", files=files).json()["results"]
    assert res[0]["status"] == "indexed" and res[1]["status"] == "error"
    job = client.post("/api/catalog/index", json={"all": True}).json()
    state = client.app.state.melody
    done = state.jobs.wait(job["id"], 120)
    assert done["status"] == "completed" and done["result"]["reindexed"] >= 5
    assert client.get("/api/catalog/status").json()["by_status"] == {"indexed": done["result"]["reindexed"]}


def test_analyze_endpoint(client, melodies):
    r = client.post("/api/audio/analyze", data={"profile": "hum"},
                    files={"audio": ("q.wav", hum_wav(melodies[1], 4), "audio/wav")})
    assert r.status_code == 200, r.text
    a = r.json()
    assert a["informative"] and len(a["notes"]) >= 8
    assert len(a["spectrogram"]["data"]) == 160 and a["waveform"]["max"]
    assert len(a["chroma"]["data"]) == 12
    assert a["comparison"]["song"]["title"] == "Song 1"
    assert a["comparison"]["alignment"]["accumulated_cost"]


def test_settings_roundtrip(client):
    s = client.get("/api/settings").json()
    assert s["retention"] == "discard"
    s2 = client.put("/api/settings", json={"melody_algorithm": "dtw", "ui_language": "es"}).json()
    assert s2["melody_algorithm"] == "dtw" and s2["ui_language"] == "es"
    assert client.app.state.melody.engine.settings.melody_algorithm == "dtw"
    assert client.put("/api/settings", json={"theme": "purple"}).status_code == 422
    assert client.put("/api/settings", json={"min_record_seconds": 25, "max_record_seconds": 10}).status_code == 422
    client.put("/api/settings", json={"melody_algorithm": "ensemble"})


def test_audio_retention_requires_consent_setting(client, melodies):
    wav = hum_wav(melodies[0], 11)
    r = client.post("/api/recognize/hum", data={"retain_audio": "true"},
                    files={"audio": ("q.wav", wav, "audio/wav")}).json()
    assert r["audio_retained"] is False  # retention policy is "discard"
    client.put("/api/settings", json={"retention": "keep_with_consent"})
    r = client.post("/api/recognize/hum", data={"retain_audio": "true"},
                    files={"audio": ("q.wav", wav, "audio/wav")}).json()
    assert r["audio_retained"] is True
    rec_dir = client.app.state.melody.catalog.root / "recordings"
    assert (rec_dir / f"{r['id']}.wav").exists()
    client.delete("/api/recognitions")
    assert not list(rec_dir.glob("*.wav"))
    client.put("/api/settings", json={"retention": "discard"})


def test_training_job_lifecycle(client, tmp_path, monkeypatch):
    """Smoke training job completes; a second one can be cancelled; unknown jobs 404."""
    monkeypatch.setenv("MELODYAI_EXPERIMENTS_DIR", str(tmp_path / "exp"))
    job = client.post("/api/training/jobs", json={"config": "smoke", "epochs": 1}).json()
    state = client.app.state.melody
    done = state.jobs.wait(job["id"], 300)
    assert done["status"] == "completed", done
    assert done["result"]["epochs_run"] == 1 and done["result"]["history"][0]["val_mrr"] > 0
    job2 = client.post("/api/training/jobs", json={"config": "smoke", "epochs": 50}).json()
    client.post(f"/api/training/jobs/{job2['id']}/cancel")
    assert state.jobs.wait(job2["id"], 300)["status"] == "cancelled"
    assert client.get("/api/training/jobs/doesnotexist").status_code == 404


def test_experiments_endpoints_validate_ids(client):
    assert client.get("/api/experiments").status_code == 200
    assert client.get("/api/experiments/../../etc").status_code in (400, 404)
    assert client.get("/api/experiments/20260101T000000Z-x/figures/..%2Fmetrics.json").status_code in (400, 404)
