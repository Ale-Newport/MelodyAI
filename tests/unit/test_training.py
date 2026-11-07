"""Training pipeline: leakage control (Test I), sampler, encoder, smoke training and checkpoint inference."""

import numpy as np
import pytest

from melodyai.demo.generator import _window_signatures, generate_catalog
from melodyai.training.data import (
    SPLIT_OFFSETS,
    PairSampler,
    check_leakage,
    compositions,
    grid_window,
    remove_leaky,
    simulate_tracker,
)

torch = pytest.importorskip("torch")


def test_I_splits_are_disjoint_by_composition():
    """Test I: no composition (or exact 8-gram passage) is shared between train and eval material."""
    train = compositions("train", 60)
    val = compositions("val", 20)
    demo = [m.notes for m in generate_catalog(16)]
    seeds = {k: v for k, v in SPLIT_OFFSETS.items()}
    assert len(set(seeds.values())) == len(seeds)  # disjoint seed ranges
    cleaned, report = remove_leaky(train, val + demo)
    assert check_leakage(cleaned, val + demo)["shared_8gram_compositions"] == 0
    # The check really detects leakage: inject a validation melody into the training set.
    leaky = cleaned + [val[3]]
    assert check_leakage(leaky, val)["shared_8gram_compositions"] >= 1
    assert report["n_train"] == 60


def test_grid_window_matches_index_grid():
    from melodyai.features.windows import DEFAULT_LENGTHS, hop_for

    for n, s, e in [(60, 7, 19), (60, 33, 40), (40, 0, 25), (12, 2, 9)]:
        gs, ge = grid_window(n, s, e)
        L = ge - gs
        assert L in [min(x, n) for x in DEFAULT_LENGTHS]
        assert gs % hop_for(L) == 0
        assert max(0, min(e, ge) - max(s, gs)) > 0  # overlaps the excerpt


def test_tracker_simulation_and_sampler_shapes():
    rng = np.random.default_rng(0)
    comps = compositions("train", 12)
    c = simulate_tracker(comps[0].slice(0, 10), rng)
    assert 10 < len(c) < 400 and np.isfinite(c).all()
    b = PairSampler(comps, 96, 1).batch(8)
    assert b.queries.shape == (8, 96) == b.refs.shape
    assert len(set(b.comp_ids.tolist())) == 8  # one window per composition -> no same-song negatives
    assert np.allclose(np.median(b.refs, axis=1), 0, atol=1.5)


def test_encoder_embeddings_are_normalised():
    from melodyai.models.encoder import MelodyEncoder

    m = MelodyEncoder(channels=32, dim=16, n_layers=1, n_heads=2).eval()
    z = m(torch.randn(4, 96))
    assert z.shape == (4, 16)
    assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)


@pytest.mark.slow
def test_smoke_training_and_checkpoint_inference(tmp_path, monkeypatch, melody_index):
    """Train for 2 tiny epochs, then load the checkpoint and run neural search on a catalog."""
    from melodyai.features.extract import QueryFeatures
    from melodyai.matching.neural import NeuralEmbedder, NeuralIndex, NeuralMatcher, NeuralRerank
    from melodyai.training.train import TrainConfig, train

    monkeypatch.setenv("MELODYAI_EXPERIMENTS_DIR", str(tmp_path))
    cfg = TrainConfig(name="t", n_train=40, n_val=10, channels=32, dim=32, n_layers=1, n_heads=2, batch_size=16,
                      steps_per_epoch=3, epochs=2, warmup_steps=1, pipeline_pool_comps=0, device="cpu", export=False)
    res = train(cfg, run_dir=tmp_path / "run")
    assert res["status"] == "completed" and res["epochs_run"] == 2
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "split_report.json").exists()
    emb = NeuralEmbedder(tmp_path / "run" / "best.pt", device="cpu")
    nidx = NeuralIndex(emb, melody_index, tmp_path / "cache", "exact")
    q = QueryFeatures.from_notes(next(iter(melody_index.refs.values())).notes.slice(5, 18))
    for alg in (NeuralMatcher(nidx), NeuralRerank(nidx)):
        c = alg.search(q, melody_index, 5)
        assert len(c) == 5 and all(0 <= x.score <= 1 for x in c)
    # Rerank uses DTW, so even a barely trained encoder finds an exact excerpt if it is retrieved.
    assert any(x.song_id == "s0" for x in NeuralRerank(nidx).search(q, melody_index, 5))
    assert len(list((tmp_path / "cache").glob("*/*.npy"))) == len(melody_index)  # per-song cache


def test_resume_keeps_history(tmp_path, monkeypatch):
    from melodyai.training.train import TrainConfig, train

    monkeypatch.setenv("MELODYAI_EXPERIMENTS_DIR", str(tmp_path))
    cfg = TrainConfig(name="r", n_train=20, n_val=6, channels=16, dim=16, n_layers=1, n_heads=2, batch_size=8,
                      steps_per_epoch=2, epochs=1, warmup_steps=1, pipeline_pool_comps=0, device="cpu", export=False,
                      val_queries_per_comp=1)
    train(cfg, run_dir=tmp_path / "run")
    cfg.epochs = 2
    res = train(cfg, run_dir=tmp_path / "run", resume=True)
    assert [h["epoch"] for h in res["history"]] == [1, 2]


def test_window_signature_helper():
    m = generate_catalog(1)[0]
    assert _window_signatures(m.notes).shape[1] == 8


def test_vector_indexes_exact_and_approximate():
    from melodyai.indexing.vector import NumpyFlat, NumpyIVF

    rng = np.random.default_rng(0)
    E = rng.normal(size=(5000, 32)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    flat, ivf = NumpyFlat(E), NumpyIVF(E, nprobe=12)
    recalls = []
    for k in range(20):
        q = E[k * 7] + 0.05 * rng.normal(size=32).astype(np.float32)
        s, i = flat.search(q, 10)
        brute = np.argsort(-(E @ q))[:10]
        assert np.array_equal(i, brute) and np.all(np.diff(s) <= 1e-6)
        recalls.append(len(set(ivf.search(q, 10)[1]) & set(brute)) / 10)
    assert np.mean(recalls) > 0.7  # approximate, but mostly right
