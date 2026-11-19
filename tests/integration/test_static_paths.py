"""The web app's file handler and the experiment endpoints never serve files outside their directories."""

from fastapi.testclient import TestClient

import app.main as main_module
from app.core.config import ServerSettings


def test_spa_serves_only_files_inside_dist(tmp_path, monkeypatch):
    root = tmp_path / "root"
    dist = root / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "robots.txt").write_text("User-agent: *")
    (root / "frontend" / "secret.txt").write_text("secret")
    (root / "frontend" / "dist-evil").mkdir()
    (root / "frontend" / "dist-evil" / "x.txt").write_text("sibling")
    monkeypatch.setattr(main_module, "ROOT_DIR", root)
    app = main_module.create_app(ServerSettings(store_dir=str(tmp_path / "store"), load_neural_model=False,
                                                serve_frontend=True))
    with TestClient(app) as c:
        assert c.get("/robots.txt").text == "User-agent: *"
        assert c.get("/library").text == "<html>app</html>"                  # client-side route
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/..%2Fsecret.txt", "/../dist-evil/x.txt"):
            r = c.get(path)
            assert "secret" not in r.text and "sibling" not in r.text


def test_experiment_endpoints_refuse_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("MELODYAI_EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    app = main_module.create_app(ServerSettings(store_dir=str(tmp_path / "store"), load_neural_model=False,
                                                serve_frontend=False))
    with TestClient(app) as c:
        assert c.get("/api/experiments/..%2F..%2Fetc").status_code in (400, 404)
        assert c.get("/api/experiments/20260101T000000Z-x/figures/..%2Fmetrics.json").status_code in (400, 404)
        assert c.get("/api/experiments/20260101T000000Z-x").status_code == 404
