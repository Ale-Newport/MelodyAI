"""Server configuration: ``.env`` entries reach the modules that read ``os.environ`` (keys, offline mode)."""

import os

from app.core.config import export_env_file


def test_export_env_file_sets_only_missing_melodyai_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MELODYAI_GENIUS_TOKEN=dummy-token\nMELODYAI_OFFLINE=0\nOTHER_VAR=x\n# comment\n")
    monkeypatch.delenv("MELODYAI_GENIUS_TOKEN", raising=False)
    monkeypatch.setenv("MELODYAI_OFFLINE", "1")
    monkeypatch.delenv("OTHER_VAR", raising=False)
    assert export_env_file(env) == ["MELODYAI_GENIUS_TOKEN"]
    assert os.environ["MELODYAI_GENIUS_TOKEN"] == "dummy-token"
    assert os.environ["MELODYAI_OFFLINE"] == "1"           # the process environment wins
    assert "OTHER_VAR" not in os.environ                   # only MELODYAI_* variables are exported
    monkeypatch.delenv("MELODYAI_GENIUS_TOKEN")


def test_export_env_file_disabled_or_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MELODYAI_ENV_FILE", "")
    assert export_env_file() == []
    assert export_env_file(tmp_path / "missing.env") == []
