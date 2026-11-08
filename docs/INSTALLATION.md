# Installation

## Requirements

| | Required | Notes |
|---|---|---|
| Python | 3.10–3.12 (3.11 recommended) | 3.13 is not yet supported by every audio/ML dependency |
| Node.js | 18+ (tested with 26) | for the web UI |
| ffmpeg | optional | decodes M4A/AAC/WebM uploads; WAV/MP3/FLAC/OGG work without it |
| uv | optional | faster installs; `make setup` uses it when present |
| GPU | not required | PyTorch uses Apple MPS or CUDA when available, otherwise CPU |

Nothing in the default setup needs Docker, PostgreSQL, a GPU or API keys. The only online services
used are optional and free: the **online song search** (on by default, switchable under Settings) sends
typed or transcribed *words*, never audio, to LRCLIB and Deezer, and the popular-songs sync runs only
when you start it ([OPEN_WORLD.md](OPEN_WORLD.md)). Whisper models come from the Hugging Face Hub, and
only on an explicit download.

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
make asr-model MODEL=large-v3-turbo   # Whisper for spoken/sung lyrics (~1.6 GB; MODEL=base is ~150 MB but weak on singing)
make world-sync PRESET=quick          # lyrics of popular songs -> data/store/world.db (real songs by their words)
make lakh                             # optional: melodies of ~25,000 real songs for humming (Lakh MIDI, ~1.5 GB download)
make train                  # train the neural encoder (≈20–30 min on Apple Silicon) -> models/melody_encoder.pt
make evaluate               # reproducible benchmark -> experiments/runs/, docs/RESULTS.md
make evaluate-world         # real-song (open-world) evaluation -> data/reports/, docs/WORLD_RESULTS.md
make test                   # Python + frontend tests
make build                  # production frontend build, then the API serves it on :8000
```

`make setup-min` installs only the classical engine and the API (no PyTorch, Whisper or
Basic Pitch); recognition by humming/whistling and typed lyrics still work.

On macOS with Homebrew: `brew install python@3.11 node ffmpeg uv`.
On Debian/Ubuntu: `sudo apt install python3.11 python3.11-venv nodejs npm ffmpeg libsndfile1`.

### Speech recognition model

The Whisper setting defaults to `auto`: the best model already downloaded (`large-v3-turbo` >
`large-v3` > `medium` > `small` > `base` > `tiny`). `large-v3-turbo` is recommended for sung lyrics: a
~1.6 GB download into the Hugging Face cache (`~/.cache/huggingface`). Download it with
`make asr-model MODEL=large-v3-turbo`, or with the *Download* button under *Settings → Speech
recognition*, which the server offers only when started with `MELODYAI_ALLOW_MODEL_DOWNLOAD=true` (in
`.env` or the environment). Whisper runs on CPU (Apple Silicon: float32) or CUDA. Typed lyrics work
without any model.

### Popular-songs index (real songs)

```bash
make world-sync PRESET=quick       # ~1,900 songs (UI estimate, about half an hour); LIMIT=N to keep the N most important
make world-sync PRESET=standard    # measured: 6,954 songs in 2 h 08 min, world.db ≈ 45 MB
make world-sync PRESET=large       # ~24,600 songs, about 7 h 20 min from empty (measured)
```

The same presets are available under *Settings → Online song search → Popular songs*. A sync is
resumable: Ctrl-C (or *Stop*) ends it after the current song, and running it again continues. Only one
sync runs at a time per store, also across processes. Requests to LRCLIB are sequential and paced
(about 1 per second), because it is a free, shared service. Details: [OPEN_WORLD.md](OPEN_WORLD.md#5-popular-songs-sync-melodyaiworldsyncpy).

## Windows

The Makefile targets assume a POSIX shell. Use **WSL2** (then follow the Linux steps), or run
the equivalent commands in PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -e ".[ml,asr,pitch,dev]" -e backend
.venv\Scripts\python -m melodyai.cli demo-data
.venv\Scripts\python -m melodyai.cli asr-model --model large-v3-turbo   # optional, ~1.6 GB
.venv\Scripts\python -m melodyai.cli world-sync --preset quick          # optional, real songs by their words
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

The open-world and Whisper variables below are read from the process environment. Put them in `.env`
(the server copies the `MELODYAI_*` entries of `.env` into its environment at startup, without overriding
variables that are already set) or export them in the shell that starts the server. The CLI only reads the
shell environment.

| Variable | Effect |
|---|---|
| `MELODYAI_OFFLINE=1` | no open-world traffic: online song search, popular-songs sync and previews are off (world index and web cache only). The licensed provider (`MELODYAI_ACRCLOUD_*`) and Whisper model downloads (`MELODYAI_ALLOW_MODEL_DOWNLOAD`) are governed by their own settings: leave those unset for a server without network access |
| `MELODYAI_GENIUS_TOKEN` | enables the Genius lyrics-search source (off by default) |
| `MELODYAI_ANTHROPIC_API_KEY` | enables the Claude source (off by default; needs `pip install -e ".[llm]"`); `MELODYAI_CLAUDE_MODEL` picks the model |
| `MELODYAI_CLAUDE_SOURCE=1` | lets the Claude source use a generic `ANTHROPIC_API_KEY` |
| `MELODYAI_ASR_COMPUTE_TYPE` | CTranslate2 compute type of Whisper (default float32 on Apple Silicon, int8 elsewhere) |

Full list and what each source receives: [OPEN_WORLD.md](OPEN_WORLD.md#9-configuration). With online
search on, the server needs HTTPS access to `lrclib.net` and `api.deezer.com`. The browser loads
covers and 30 s previews from Deezer's CDN (`*.dzcdn.net`).

## Microphone permissions

Browsers only allow microphone capture on `https://` or `http://localhost` / `127.0.0.1`. The first
recording triggers the browser permission prompt; if it was denied, re-enable it in the site
settings. Recordings are processed by your local server and are not stored unless you enable
retention *and* tick the per-recording consent box.

## What was validated in the release environment

macOS 27 on Apple Silicon (arm64), Python 3.11.15, Node 26, Chromium-based in-app browser.
The Python test suite, the frontend tests/type-check/build, the full benchmark and UI flows driven
by browser automation (file upload path) ran there. The `standard` and `large` popular-songs syncs and the
open-world evaluation (with Whisper `large-v3-turbo`) ran there too. Real microphone
recording, Safari, Firefox, Windows, Linux and Docker were **not** validated and need manual checking.
