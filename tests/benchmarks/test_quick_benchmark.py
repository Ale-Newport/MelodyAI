"""Benchmark regression guard: the quick synthetic benchmark must run end to end, write all
artifacts and stay above conservative floors (these are NOT real-world accuracy figures)."""

import json

import pytest

from melodyai.config import ROOT_DIR

pytestmark = pytest.mark.slow


def test_quick_benchmark(tmp_path, monkeypatch):
    from melodyai.evaluation.report import build_report
    from melodyai.evaluation.runner import run_evaluation

    monkeypatch.setenv("MELODYAI_EXPERIMENTS_DIR", str(tmp_path))
    res = run_evaluation(ROOT_DIR / "configs" / "eval_quick.yaml")
    run = tmp_path / "runs" / res["run_id"]
    for f in ("metrics.json", "per_query.jsonl", "env.json", "config.yaml", "figures/top1_by_condition.png"):
        assert (run / f).exists(), f
    m = json.loads((run / "metrics.json").read_text())
    assert m["melody"]["ensemble"]["overall"]["top1"] >= 0.85
    assert m["melody"]["dtw"]["overall"]["top1"] >= 0.8
    assert m["melody"]["ensemble"]["rejection"]["test"]["auc_correct_vs_ooc"] >= 0.8
    assert m["lyrics"]["overall"]["top1"] >= 0.9
    report = build_report(run, tmp_path / "RESULTS.md")
    assert "Auto-generated" in report.read_text()
