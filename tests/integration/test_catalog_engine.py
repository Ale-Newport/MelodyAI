"""Catalog persistence, incremental indexing and engine behaviour on a real (temporary) store."""

import numpy as np
import pytest

from melodyai.audio.io import Audio, wav_bytes
from melodyai.audio.synth import render
from melodyai.catalog.importers import ImportErrorMelody, notes_to_midi_bytes
from melodyai.catalog.service import Catalog, DuplicateSongError, NotFoundError
from melodyai.demo.generator import generate_catalog
from melodyai.demo.lyrics_bank import LYRICS
from melodyai.demo.queries import CONDITIONS, perturb, synth_query
from melodyai.engine import EngineSettings, RecognitionEngine

N = 8


@pytest.fixture(scope="module")
def songs():
    return generate_catalog(N + 1, base_seed=1000)


@pytest.fixture()
def catalog(tmp_path, songs):
    cat = Catalog(tmp_path / "store")
    for i, m in enumerate(songs[:N]):
        lyr = {"text": LYRICS[i]["text"], "language": LYRICS[i]["language"]} if i < 4 else None
        cat.add_song({"id": f"song{i}", "title": f"Song {i}", "artist": "Tester", "license": "CC0-1.0"},
                     notes_to_midi_bytes(m.notes), f"song{i}.mid", lyrics=lyr)
    return cat


def hum(m, rng, start_note=10, n_notes=14):
    spec = CONDITIONS["hum_clean"]
    pn, _ = perturb(m.notes.slice(start_note, start_note + n_notes), spec, rng)
    return Audio(synth_query(pn, spec, rng), 16000)


def engine(cat):
    return RecognitionEngine(cat.melody_index, cat.lyrics_index, EngineSettings(asr_enabled=False))


def test_import_and_index(catalog):
    st = catalog.status()
    assert st["by_status"] == {"indexed": N} and st["with_lyrics"] == 4
    s = catalog.get_song("song0")
    assert s["n_notes"] > 20 and s["n_windows"] > 20 and s["feature_version"]
    assert s["lyrics"]["text"].startswith("Paper boats")


def test_duplicate_and_invalid_imports(catalog, songs):
    with pytest.raises(DuplicateSongError):
        catalog.add_song({"title": "again"}, notes_to_midi_bytes(songs[0].notes), "x.mid")
    with pytest.raises(ImportErrorMelody):
        catalog.add_song({"title": "bad"}, b"not a midi", "x.mid")
    with pytest.raises(ImportErrorMelody):
        catalog.add_song({"title": "bad"}, b"abc", "x.exe")
    assert catalog.status()["songs"] == N


def test_G_deleted_song_disappears_from_indexes_and_results(catalog, songs, rng):
    eng = engine(catalog)
    q = hum(songs[2], rng)
    assert eng.recognize(q, mode="hum")["results"][0]["song_id"] == "song2"
    catalog.delete_song("song2")
    res = eng.recognize(q, mode="hum")
    assert all(r["song_id"] != "song2" for r in res["results"])
    assert "song2" not in catalog.melody_index and "song2" not in catalog.lyrics_index
    assert not catalog.feature_path("song2").exists()
    with pytest.raises(NotFoundError):
        catalog.get_song("song2")
    # Lyrics of a deleted song are gone too.
    lyr = eng.recognize(text="turn the lantern turn the lantern slow", mode="lyrics")
    assert lyr["results"][0]["song_id"] == "song1"
    catalog.delete_song("song1")
    lyr = eng.recognize(text="turn the lantern turn the lantern slow", mode="lyrics")
    assert all(r["song_id"] != "song1" for r in lyr["results"])


def test_H_incremental_add_does_not_recompute_others(catalog, songs, rng):
    before = catalog.stats["features_computed"]
    mtimes = {p.name: p.stat().st_mtime_ns for p in catalog.features_dir.glob("*.npz")}
    catalog.add_song({"id": "new", "title": "New song"}, notes_to_midi_bytes(songs[N].notes), "new.mid")
    assert catalog.stats["features_computed"] == before + 1
    assert {p.name: p.stat().st_mtime_ns for p in catalog.features_dir.glob("*.npz") if p.name in mtimes} == mtimes
    assert engine(catalog).recognize(hum(songs[N], rng), mode="hum")["results"][0]["song_id"] == "new"


def test_persistence_and_stale_detection(catalog, tmp_path, monkeypatch):
    reopened = Catalog(tmp_path / "store")
    assert len(reopened.melody_index) == N and len(reopened.lyrics_index) == 4
    assert reopened.stats["features_computed"] == 0 and reopened.stats["features_loaded"] == N
    # A new feature version marks everything stale; reindex recomputes.
    import melodyai.catalog.service as svc

    monkeypatch.setattr(svc, "FEATURE_VERSION", "mel-test")
    stale = Catalog(tmp_path / "store")
    assert stale.status()["by_status"] == {"stale": N} and len(stale.melody_index) == 0
    assert stale.reindex()["reindexed"] == N
    assert stale.status()["by_status"] == {"indexed": N}


def test_audio_reference_import(tmp_path, songs, rng):
    """Monophonic audio references are transcribed with pYIN and become searchable."""
    cat = Catalog(tmp_path / "store2")
    for i in range(3):
        y = render(songs[i].notes, "voice", sr=16000, seed=i)
        s = cat.add_song({"title": f"audio {i}", "reference_type": "vocal_audio"}, wav_bytes(y, 16000), f"a{i}.wav")
        assert s["index_status"] == "indexed", s["index_error"]
    res = engine(cat).recognize(hum(songs[1], rng), mode="hum")
    assert cat.get_song(res["results"][0]["song_id"])["title"] == "audio 1"


def test_J_demo_mode_without_models_or_network(catalog, songs, rng):
    """Test J: hum + typed lyrics + combined work with ASR disabled and no neural model."""
    eng = engine(catalog)
    assert "neural" not in eng.algorithms
    r = eng.recognize(hum(songs[3], rng), mode="auto")
    assert r["status"] == "match" and r["results"][0]["song_id"] == "song3"
    assert r["engines"]["lyrics"]["status"] == "unavailable"
    assert r["alignment"]["path"] and r["alignment"]["accumulated_cost"]
    t = eng.recognize(text="paper boats carry me back home", mode="lyrics")
    assert t["results"][0]["song_id"] == "song0" and t["status"] == "match"
    # Combined: ambiguous short hum + lyrics -> fused result is the lyric song.
    c = eng.recognize(hum(songs[0], rng, 0, 8), text="every street a river after rain", mode="combined")
    assert c["resolved_mode"] == "combined" and c["results"][0]["song_id"] == "song0"
    assert set(c["results"][0]["engines"]) == {"melody", "lyrics"}


def test_E_engine_abstains_on_noise(catalog, rng):
    y = (0.2 * rng.standard_normal(16000 * 5)).astype(np.float32)
    r = engine(catalog).recognize(Audio(y, 16000), mode="hum")
    assert r["status"] == "no_match" and r["message"].startswith("No sufficiently reliable")
