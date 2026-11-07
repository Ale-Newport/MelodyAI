import os
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")
os.environ.setdefault("MELODYAI_DEVICE", "cpu")

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


@pytest.fixture()
def rng():
    return np.random.default_rng(1234)


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("MELODYAI_STORE_DIR", str(store))
    return store
