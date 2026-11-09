"""Hummed melodies of real songs: vocal-track choice, Lakh MIDI index build, search and engine resolution.

No network and no Lakh download: the "Lakh" files are small MIDI files generated here from invented melodies
(demo generator), laid out like LMD-matched with invented Million Song Dataset ids, titles and artists.
"""

import io
import json

import numpy as np
import pretty_midi
import pytest

from melodyai.audio.io import Audio
from melodyai.catalog.importers import load_midi_melody
from melodyai.demo.generator import generate_catalog
from melodyai.demo.queries import CONDITIONS, excerpt, perturb, synth_query
from melodyai.engine import WORLD_MELODY_SOURCE, EngineSettings, RecognitionEngine, world_melody_item
from melodyai.features.extract import analyze_query
from melodyai.features.melody_track import rank_melody_tracks, vocal_track_from_lyrics
from melodyai.lyrics.search import LyricsIndex
from melodyai.world.lakh import build_lakh_index, plan_songs
from melodyai.world.melodies import WorldMelodyIndex, pack_notes, unpack_notes


def _inst(program, notes, name="", drum=False):
    inst = pretty_midi.Instrument(program=program, name=name, is_drum=drum)
    for s, e, p in notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(p), start=float(s), end=float(e)))
    return inst


def song_midi(melody, seed=0, lyrics=False, vocal_program=54, vocal_name="") -> pretty_midi.PrettyMIDI:
    """A band arrangement around an invented vocal line: bass, block chords, drums and the voice."""
    rng = np.random.default_rng(seed)
    pm = pretty_midi.PrettyMIDI(initial_tempo=110)
    end = float(melody.end_time) + 1
    beats = np.arange(0, end, 0.5)
    pm.instruments.append(_inst(33, [(b, b + 0.45, 36 + rng.integers(0, 7)) for b in beats], "Bass"))
    chords = [(b, b + 1.9, 48 + k) for b in np.arange(0, end, 2.0) for k in (0, 4, 7)]
    pm.instruments.append(_inst(0, chords, "Piano"))
    pm.instruments.append(_inst(0, [(b, b + 0.1, 38) for b in beats], "Drums", drum=True))
    voice = [(o, o + d, p) for o, d, p in zip(melody.onsets, melody.durations, melody.pitches)]
    pm.instruments.append(_inst(vocal_program, voice, vocal_name))
    if lyrics:
        pm.lyrics = [pretty_midi.Lyric("la", float(o)) for o in melody.onsets]
    return pm


@pytest.fixture(scope="module")
def songs():
    return generate_catalog(8, base_seed=5000)


def test_vocal_track_is_chosen_over_bass_piano_and_drums(songs):
    pm = song_midi(songs[0].notes, vocal_program=0, vocal_name="")       # even on a piano program, unnamed
    ranked = rank_melody_tracks(pm)
    assert ranked[0][0] == 3 and len(ranked) == 3                          # drums are never candidates
    assert vocal_track_from_lyrics(pm) is None
    pm2 = song_midi(songs[1].notes, lyrics=True)
    assert vocal_track_from_lyrics(pm2) == 3
    buf = io.BytesIO()
    pm.write(buf)
    m = load_midi_melody(buf.getvalue())
    assert m.track_index == 3 and len(m.notes) == len(songs[0].notes)


def test_notes_roundtrip(songs):
    n = songs[0].notes
    back = unpack_notes(pack_notes(n))
    assert np.allclose(back.onsets, n.onsets, atol=1e-3) and np.allclose(back.pitches, n.pitches)


@pytest.fixture(scope="module")
def lakh(tmp_path_factory, songs):
    """An LMD-matched-shaped folder of 8 invented songs, built into a melody index."""
    root = tmp_path_factory.mktemp("lakh")
    raw, dl = root / "raw", root / "download"
    dl.mkdir()
    scores, tracks = {}, []
    for i, s in enumerate(songs):
        trid = f"TRAB{i:02d}CD1234567890"
        folder = raw / "lmd_matched" / trid[2] / trid[3] / trid[4] / trid
        folder.mkdir(parents=True)
        md5 = f"{i:032x}"
        song_midi(s.notes, seed=i, lyrics=i % 3 == 0).write(str(folder / f"{md5}.mid"))
        scores[trid] = {md5: 0.9 - i * 0.01}
        tracks.append(f"{trid}<SEP>SOX{i}<SEP>Invented Band {i}<SEP>Invented Song {i}")
    tracks.append("TRZZ00000000000000<SEP>SOX<SEP>Nobody<SEP>Unmatched Song")
    (dl / "match_scores.json").write_text(json.dumps(scores))
    (dl / "unique_tracks.txt").write_text("\n".join(tracks) + "\n")
    meta = build_lakh_index(raw, dl, root / "world_melodies", workers=1, holdout=0)
    return root, meta


def test_plan_and_build(lakh):
    root, meta = lakh
    plan = plan_songs(root / "raw", root / "download" / "match_scores.json", root / "download" / "unique_tracks.txt")
    assert len(plan) == 8 and plan[0].title == "Invented Song 0" and plan[0].artist == "Invented Band 0"
    assert meta["songs"] == 8 and meta["melodies"] >= 8 and meta["windows"] > 0 and meta["encoder"]
    assert meta["selections"].get("lyrics", 0) >= 3
    idx = WorldMelodyIndex(root / "world_melodies")
    st = idx.stats()
    assert st["built"] and st["songs"] == 8 and st["source"].startswith("Lakh MIDI")


def test_reembed_keeps_the_melodies(lakh):
    root, meta = lakh
    again = build_lakh_index(root / "raw", root / "download", root / "world_melodies", workers=1, reembed=True)
    assert {k: again[k] for k in ("songs", "melodies", "windows")} == {k: meta[k] for k in ("songs", "melodies", "windows")}
    assert again["extract_s"] == 0.0 or again["extract_s"] < meta["extract_s"] + 5


def test_hummed_excerpt_finds_its_song(lakh, songs):
    root, _ = lakh
    looked_up = []

    def lookup(title, artist):
        looked_up.append((title, artist))
        return None

    idx = WorldMelodyIndex(root / "world_melodies", meta_lookup=lookup)
    rng = np.random.default_rng(3)
    spec = CONDITIONS["hum_clean"]
    hits = 0
    for i in (1, 4, 6):
        ex, _ = excerpt(songs[i].notes, rng, 6.0, 9.0)
        q = analyze_query(Audio(synth_query(perturb(ex, spec, rng)[0], spec, rng), 16000), with_chroma=False)
        res = idx.search(q, 5)
        assert res and len({m.key for m in res}) == len(res)               # one entry per song
        hits += res[0].title == f"Invented Song {i}"
        d = res[0].to_dict()
        assert d["song_id"].startswith("world:w") and d["links"]["youtube"].startswith("https://www.youtube.com/")
    assert hits >= 2 and looked_up


def test_unbuilt_or_broken_index_is_harmless(tmp_path):
    idx = WorldMelodyIndex(tmp_path / "nothing")
    assert not idx.built and not idx.warm() and idx.stats()["songs"] == 0
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "meta.json").write_text("{}")
    (tmp_path / "bad" / "melodies.db").write_text("not a database")
    bad = WorldMelodyIndex(tmp_path / "bad")
    assert bad.built and not bad.warm() and bad.error


# ---------------------------------------------------------------------------- engine
class FakeMelodies:
    built = True
    error = ""

    def __init__(self, scores):
        self.scores = scores

    def search(self, q, top_k=5):
        from melodyai.world.melodies import MelodyMatch

        return [MelodyMatch(f"world:w{i:015x}", f"k{i}", f"Invented Song {i}", "Invented Band", s, 1.0, 5.0, i,
                            None, "model", 0.5) for i, s in enumerate(self.scores)]


def engine(fake):
    from melodyai.indexing.melody_index import MelodyIndex

    return RecognitionEngine(MelodyIndex(), LyricsIndex(":memory:"), EngineSettings(asr_enabled=False),
                             world_melody=fake)


def hum(songs):
    rng = np.random.default_rng(5)
    spec = CONDITIONS["hum_clean"]
    pn, _ = perturb(songs[2].notes.slice(4, 20), spec, rng)
    return Audio(synth_query(pn, spec, rng), 16000)


def test_confident_real_song_melody_resolves_melody_online(songs):
    r = engine(FakeMelodies([0.95, 0.4])).recognize(hum(songs), mode="hum")
    wm = r["engines"]["world_melody"]
    assert wm["status"] == "ok" and wm["decision"]["status"] == "match" and wm["source"] == WORLD_MELODY_SOURCE
    assert r["resolved_mode"] == "melody_online" and r["status"] == "match"
    top = r["results"][0]
    assert top["song"]["title"] == "Invented Song 0" and top["song"]["source"] == WORLD_MELODY_SOURCE
    assert top["engines"]["world_melody"]["score"] == 0.95 and top["song"]["reference_type"] == "online"


def test_weak_real_song_melody_is_only_a_suggestion(songs):
    r = engine(FakeMelodies([0.62, 0.55])).recognize(hum(songs), mode="hum")
    assert r["engines"]["world_melody"]["decision"]["status"] == "uncertain"
    assert r["resolved_mode"] == "melody_online" and r["status"] == "uncertain"   # nothing local at all
    r = engine(FakeMelodies([0.3, 0.29])).recognize(hum(songs), mode="hum")
    assert r["resolved_mode"] != "melody_online" and r["status"] == "no_match"


def test_no_melody_index_means_no_engine_entry(songs):
    r = engine(None).recognize(hum(songs), mode="hum")
    assert "world_melody" not in r["engines"]
    item = world_melody_item(1, FakeMelodies([0.9]).search(None)[0].to_dict())
    assert item["song"]["source"] == WORLD_MELODY_SOURCE and item["engines"]["world_melody"]["rank"] == 1
