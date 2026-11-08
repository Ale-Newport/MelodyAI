# MelodyAI — developer commands. Requires: Python 3.11 (3.10–3.12), Node 18+, optionally uv and ffmpeg.
# Windows: use WSL, or run the equivalent commands shown in docs/INSTALLATION.md.
SHELL := /bin/bash
PY ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin
UV := $(shell command -v uv 2>/dev/null)
EXTRAS ?= ml,asr,pitch,dev
MODEL ?= large-v3-turbo
PRESET ?= quick

.PHONY: help setup setup-min demo-data dev backend frontend test test-fast test-frontend evaluate evaluate-quick \
        evaluate-scale evaluate-world train train-smoke report lint format build asr-model ingest world-sync \
        clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
ifdef UV
	uv venv --python 3.11 $(VENV)
else
	$(PY) -m venv $(VENV)
endif

setup: $(BIN)/python ## Install backend (all extras) + frontend dependencies
ifdef UV
	uv pip install --python $(BIN)/python -e ".[$(EXTRAS)]" -e backend
else
	$(BIN)/pip install -U pip && $(BIN)/pip install -e ".[$(EXTRAS)]" -e backend
endif
	cd frontend && npm install --no-fund --no-audit

setup-min: ## Minimal install: classical engine + API only (no torch / whisper / basic-pitch)
	$(MAKE) setup EXTRAS=dev

demo-data: ## Generate the original demo catalog (48 songs) and import it
	$(BIN)/python -m melodyai.cli demo-data

backend: ## Run the API on http://127.0.0.1:8000 (docs at /docs)
	$(BIN)/uvicorn app.main:app --app-dir backend --reload --reload-dir backend --reload-dir melodyai --port 8000

frontend: ## Run the web app on http://127.0.0.1:5173
	cd frontend && npm run dev

dev: ## Run backend and frontend together (Ctrl-C stops both)
	@trap 'kill 0' INT TERM EXIT; $(MAKE) backend & $(MAKE) frontend & wait

test: ## Run all Python tests (incl. slow audio tests) and frontend tests
	$(BIN)/python -m pytest -q
	cd frontend && npm test

test-fast: ## Python tests without the slow audio end-to-end ones
	$(BIN)/python -m pytest -q -m "not slow"

test-frontend:
	cd frontend && npm run typecheck && npm test

evaluate: ## Full reproducible benchmark (~25 min) -> experiments/runs/, configs/calibration.json, docs/RESULTS.md
	$(BIN)/python -m melodyai.cli evaluate --config configs/eval_default.yaml
	$(BIN)/python -m melodyai.cli report

evaluate-quick: ## Small benchmark (~2 min)
	$(BIN)/python -m melodyai.cli evaluate --config configs/eval_quick.yaml

evaluate-scale: ## Scale benchmark (~40 min): recordings/TikTok edits up to 2k tracks, humming up to 100k songs -> docs/SCALE_RESULTS.md
	$(BIN)/python -m melodyai.cli evaluate-scale --config configs/eval_scale.yaml
	$(BIN)/python -m melodyai.cli report-scale

evaluate-world: ## Real-song open-world (lyrics) evaluation -> data/reports/, docs/WORLD_RESULTS.md (LIMIT=N, ARGS=...)
	$(BIN)/python -m melodyai.cli evaluate-world --config configs/eval_world.yaml $(if $(LIMIT),--limit $(LIMIT)) $(ARGS)

ingest: ## Parallel, resumable bulk import of a licensed collection: make ingest DIR=/path/to/music
	$(BIN)/python -m melodyai.cli ingest "$(DIR)"

report: ## Regenerate docs/RESULTS.md from the latest benchmark run
	$(BIN)/python -m melodyai.cli report

train: ## Train the neural melody encoder (≈20 min on Apple Silicon MPS) -> models/melody_encoder.pt
	$(BIN)/python -m melodyai.cli train --config configs/train_default.yaml

train-smoke: ## Tiny training run (seconds) to check the pipeline
	$(BIN)/python -m melodyai.cli train --config configs/train_smoke.yaml

asr-model: ## Download a faster-whisper model (MODEL=large-v3-turbo|large-v3|medium|small|base|tiny; ~1.6 GB for turbo)
	$(BIN)/python -m melodyai.cli asr-model --model $(MODEL)

world-sync: ## Add popular songs' lyrics to the open-world index (PRESET=quick|standard|large, LIMIT=N; resumable)
	$(BIN)/python -m melodyai.cli world-sync --preset $(PRESET) $(if $(LIMIT),--limit $(LIMIT))

lint: ## Ruff (Python) + TypeScript type check
	$(BIN)/ruff check melodyai backend tests scripts
	cd frontend && npm run typecheck

format:
	$(BIN)/ruff check --fix melodyai backend tests scripts
	$(BIN)/ruff format melodyai backend tests scripts

build: ## Production build of the frontend (served by the API at http://127.0.0.1:8000)
	cd frontend && npm run build

clean: ## Remove caches (keeps the catalog, models and experiment results)
	rm -rf .pytest_cache .ruff_cache frontend/dist experiments/cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
