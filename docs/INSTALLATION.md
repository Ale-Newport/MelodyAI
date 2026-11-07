# Installation

## Requirements

| | Required | Notes |
|---|---|---|
| Python | 3.10–3.12 (3.11 recommended) | 3.13 is not yet supported by every audio/ML dependency |
| Node.js | 18+ (tested with 26) | for the web UI |
| ffmpeg | optional | decodes M4A/AAC/WebM uploads; WAV/MP3/FLAC/OGG work without it |
| uv | optional | faster installs; `make setup` uses it when present |
| GPU | not required | PyTorch uses Apple MPS or CUDA when available, otherwise CPU |

Nothing in the default setup needs Docker, PostgreSQL, a GPU, API keys or a cloud service.

## macOS / Linux (recommended path)

```bash
git clone <your fork> MelodyAI && cd MelodyAI
make setup        # .venv with Python 3.11 + all extras (ml, asr, pitch, dev) + frontend packages
make demo-data    # generate the 48 original demo songs and import them (≈10 s)
make dev          # API on :8000 and web app on :5173
```

Open <http://127.0.0.1:5173>. API docs: <http://127.0.0.1:8000/docs>.

Optional steps:

```bash
make asr-model MODEL=base   # download a Whisper model (~150 MB) for spoken/sung lyrics
make train                  # train the neural encoder (≈20–30 min on Apple Silicon) -> models/melody_encoder.pt
make evaluate               # reproducible benchmark -> experiments/runs/, docs/RESULTS.md
make test                   # Python + frontend tests
make build                  # production frontend build, then the API serves it on :8000
```

`make setup-min` installs only the classical engine and the API (no PyTorch, Whisper or
Basic Pitch); recognition by humming/whistling and typed lyrics still work.

On macOS with Homebrew: `brew install python@3.11 node ffmpeg uv`.
On Debian/Ubuntu: `sudo apt install python3.11 python3.11-venv nodejs npm ffmpeg libsndfile1`.

## Windows

The Makefile targets assume a POSIX shell. Use **WSL2** (then follow the Linux steps), or run
the equivalent commands in PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -e ".[ml,asr,pitch,dev]" -e backend
.venv\Scripts\python -m melodyai.cli demo-data
.venv\Scripts\uvicorn app.main:app --app-dir backend --port 8000
# second terminal
cd frontend; npm install; npm run dev
```

Windows has not been tested by the authors.

## Docker (alternative)

```bash
docker compose up --build    # http://localhost:8000 (API + built web app, CPU only)
```

The image generates the demo catalog on first start into a named volume. It was not built in the
development environment of this release.

## Configuration

Copy `.env.example` to `.env` to change the store location, database URL, limits, PyTorch device,
vector backend or CORS origins. All variables are optional and documented in the file.

## Microphone permissions

Browsers only allow microphone capture on `https://` or `http://localhost` / `127.0.0.1`. The first
recording triggers the browser permission prompt; if it was denied, re-enable it in the site
settings. Recordings are processed by your local server and are not stored unless you enable
retention *and* tick the per-recording consent box.

## What was validated in the release environment

macOS 27 on Apple Silicon (arm64), Python 3.11.15, Node 26, Chromium-based in-app browser.
The Python test suite, the frontend tests/type-check/build, the full benchmark and UI flows driven
by browser automation (file upload path) ran there. Real microphone recording, Safari, Firefox,
Windows, Linux and Docker were **not** validated and need manual checking.
