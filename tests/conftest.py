import os
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
os.environ.setdefault("MELODYAI_DEVICE", "cpu")
os.environ.setdefault("MELODYAI_OFFLINE", "1")  # apps built by tests never reach LRCLIB / Deezer (web cache only)
os.environ["MELODYAI_ENV_FILE"] = ""            # a developer's .env never leaks keys or settings into tests
# Keyed open-world sources stay off in tests, whatever the developer's shell exports (tests that swap in an
# online-capable mock web would otherwise send the test text to the real Claude / Genius APIs).
for _key in ("ANTHROPIC_API_KEY", "MELODYAI_ANTHROPIC_API_KEY", "MELODYAI_CLAUDE_SOURCE", "MELODYAI_GENIUS_TOKEN"):
    os.environ.pop(_key, None)

from melodyai.demo.generator import generate_catalog  # noqa: E402
from melodyai.features.extract import reference_from_notes  # noqa: E402
from melodyai.indexing.melody_index import MelodyIndex  # noqa: E402

N_TEST_SONGS = 16


@pytest.fixture(scope="session")
def melodies():
    return generate_catalog(N_TEST_SONGS, base_seed=1000)


@pytest.fixture(scope="session")
def melody_index(melodies):
    idx = MelodyIndex()
    for i, m in enumerate(melodies):
        idx.add(reference_from_notes(f"s{i}", m.notes))
    return idx


@pytest.fixture(autouse=True)
def _no_real_whisper(request, monkeypatch):
    """Engines and apps built by tests never load a real Whisper model from the developer's Hugging Face cache
    (slow, ~3 GB of RAM, and it leaks into WhisperTranscriber's class-level model cache), unless the test is
    marked ``asr``. Tests of the transcription path fake ``asr_ready`` / ``transcribe`` themselves."""
    if request.node.get_closest_marker("asr") is None:
        from melodyai import engine

        monkeypatch.setattr(engine, "asr_available", lambda: False)


@pytest.fixture()
def rng():
    return np.random.default_rng(1234)


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("MELODYAI_STORE_DIR", str(store))
    return store
