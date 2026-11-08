# REST API

Base URL `http://127.0.0.1:8000`. Interactive OpenAPI docs at `/docs` (Swagger) and `/redoc`;
the schema is at `/openapi.json`. All endpoints are local; there is no authentication (the server
is meant to run on your machine).

## Recognition

| Method | Path | Description |
|---|---|---|
| POST | `/api/recognize` | Recognise; form field `mode` = `auto` \| `clip` \| `hum` \| `whistle` \| `lyrics` \| `combined` |
| POST | `/api/recognize/{clip,hum,whistle,lyrics,combined}` | Same, mode fixed by the path (`clip` = recording of the actual track: fingerprinting) |
| GET | `/api/recognitions?limit=20` | History (results only) |
| GET | `/api/recognitions/{id}` | One stored result |
| DELETE | `/api/recognitions/{id}` | Delete one result (and its recording if retained) |
| DELETE | `/api/recognitions` | Delete all history and retained recordings |

Multipart form fields: `audio` (file: WAV, MP3, FLAC, OGG, M4A/WebM with ffmpeg), `text`
(typed lyrics, ≤ 2000 chars), `algorithm` (see `/api/models`), `language` (`en`/`es`), `top_k` (1–20),
`pitch_extractor`, `fusion_method` (`weighted`/`rrf`/`combmnz`), `retain_audio` (only honoured if
the retention setting is `keep_with_consent`).

```bash
curl -s -F mode=hum -F audio=@my_hum.wav http://127.0.0.1:8000/api/recognize | jq '.status, .results[0].song.title'
curl -s -F text="the lighthouse keeper counts the ships" http://127.0.0.1:8000/api/recognize/lyrics
```

Response (abridged):

```json
{
  "status": "match",                 // match | uncertain | no_match
  "message": "Match found.",         // "No sufficiently reliable match was found." when no_match
  "resolved_mode": "hum",
  "score_kind": "relative",          // "calibrated_probability" only when a Platt model exists
  "decision": {"status": "match", "confidence": "high", "top_score": 0.92, "margin": 0.54,
               "thresholds": {"accept": 0.55, "uncertain": 0.4, "margin": 0.0},
               "threshold_source": "calibrated", "calibrated_probability": 0.97},
  "results": [{"rank": 1, "song_id": "demo-12", "score": 0.923,
               "song": {"title": "Midnight Train", "artist": "...", "has_audio": true},
               "engines": {"melody": {"rank": 1, "score": 0.92, "start_time": 10.6, "end_time": 17.4}},
               "match_start": 10.6, "match_end": 17.4}],
  "engines": {"melody": {"algorithm": "ensemble", "profile": "hum", "reliability": 1.0, "candidates": []},
              "lyrics": {"source": "asr", "transcript": {"speech_detected": false}, "candidates": []}},
  "query": {"quality": {}, "pitch_track": {}, "notes": {}, "contour": []},
  "alignment": {"query_contour": [], "reference_contour": [], "path": [], "accumulated_cost": [[]]},
  "elapsed_s": 1.9
}
```

`score` is a **relative similarity** within the chosen algorithm, not a probability. In `clip` mode
it is the verified peak-overlap fraction and `results[].engines.fingerprint` adds `speed` (clip speed
relative to the original), `pitch_semitones`, `evidence` and the matched `start_time`/`end_time`.

If the user enabled the licensed provider in Settings and the server has credentials, the response also
contains `external`: a list of `{provider, kind: "music"|"humming", status: match|no_result|error,
matches: [{title, artists, album, release_date, isrc, score, score_kind, play_offset_s, links: {spotify,
youtube, deezer, musicbrainz}}], sent_seconds, elapsed_s}`. These results are separate from `results`
(different catalog, different score scale).

Whenever a recognition has query text (typed, or a Whisper transcript), `engines.world` holds the
open-world search: `{status: ok|no_candidates|no_query|disabled|error, query_text, online,
offline_reason: ""|"setting_off"|"server_offline", sources: {name: {status, n, elapsed_s, message}},
candidates: [...], decision, elapsed_s}`. When it decides the answer, `resolved_mode` is
`"lyrics_online"` and `results[]` holds entries with `song_id = "world:<id>"`, a `song` with
`reference_type: "online"`, `cover_url`, `deezer_id` and `links: {deezer, spotify, youtube}`, and
`engines.world: {score, coverage, matched_text, line_range, sources}`. See [OPEN_WORLD.md](OPEN_WORLD.md).

Errors: `400` empty file / invalid id, `413` file too large, `422` undecodable or too long/short
audio, unknown algorithm, invalid parameters.

## Open-world search

| Method | Path | Description |
|---|---|---|
| GET | `/api/world/status` | World index size and bytes, online search enabled / server online, sources (`name, label, online, requires_key, available`), last sync, presets, a sync running in another process |
| POST | `/api/world/search` | JSON `{q, language?, top_k?}`: search songs by words. The words travel in the body, never in the URL. Returns the `engines.world` shape above |
| POST | `/api/world/sync` | JSON `{preset: quick\|standard\|large, limit?}` → background job (`kind = "world_sync"`); `409` if a sync is already running (also in another process), `503` if the server is offline |
| GET | `/api/world/songs?q=&offset=&limit=` | Songs of the world index (metadata only, no lyrics), most popular first |
| GET | `/api/world/preview/{deezer_id}` | `307` to a fresh 30 s Deezer preview (stored preview URLs expire) |

`DELETE /api/recognitions` also purges the cached online searches and the songs learned from
recognitions (`cached_searches`, `learned_songs`). `/api/*` refuses cross-site requests.

## Catalog

| Method | Path | Description |
|---|---|---|
| GET | `/api/songs?q=&status=&offset=&limit=` | List / search / filter |
| POST | `/api/songs` | Add one song (multipart: `reference` file + metadata + optional `lyrics_text` or `lyrics_file`) |
| GET / PATCH / DELETE | `/api/songs/{id}` | Read / edit metadata / delete (removes it from every index) |
| PUT | `/api/songs/{id}/lyrics` | JSON `{text, language, source, license}`; empty text removes lyrics |
| POST | `/api/songs/{id}/reindex` | Recompute features of one song |
| GET | `/api/songs/{id}/audio` | Reference audio or synthesised preview of a MIDI |
| GET | `/api/songs/{id}/melody` | Stored notes and contour |
| GET | `/api/catalog/status` | Counts by status, feature version, index sizes |
| GET | `/api/catalog/duplicates` | Same title+artist or composition id groups |
| POST | `/api/catalog/import` | Bulk: `manifest` (CSV/JSON) + `files[]` it references |
| POST | `/api/catalog/index` | Background reindex job; JSON `{song_ids?, all?}` |
| POST | `/api/catalog/demo` | Background job: generate + import the demo catalog |

`POST /api/songs` fields: `title` (required), `artist`, `album`, `year`, `language`,
`reference_type` (`midi`, `monophonic_audio`, `vocal_audio`, `instrumental_audio`,
`polyphonic_audio`), `composition_id`, `source`, `license`, `listen_url` (http/https only),
`midi_track`. Duplicate files (same SHA-256) return `409` with the existing id.

Manifest columns: `title, artist, album, year, language, composition_id, source, license,
reference_type, reference_file, lyrics_file, listen_url`. File references are matched by base
name against the uploaded files; no server paths are ever read.

## Analysis, models, experiments, training

| Method | Path | Description |
|---|---|---|
| POST | `/api/audio/analyze` | `audio`, `profile` (hum/whistle/sing), `extractor`, `song_id?`, `match` → waveform, spectrogram, pitch track, notes, intervals, chroma, alignment |
| GET | `/api/models` | Pitch extractors, matching algorithms, Whisper model status, neural checkpoints, device, licensed providers (configured? recordings/humming) |
| POST | `/api/models/neural/activate` | JSON `{checkpoint}` (id from `/api/models`) |
| POST | `/api/models/asr/download` | JSON `{model}`; only if `MELODYAI_ALLOW_MODEL_DOWNLOAD=true` |
| GET | `/api/experiments` | Benchmark runs, scale runs and training runs |
| GET | `/api/experiments/{run_id}` | Full `metrics.json` |
| GET | `/api/experiments/{run_id}/queries?algorithm=&condition=&only_errors=` | Per-query rows |
| GET | `/api/experiments/{run_id}/figures/{name}.png` | Figures |
| POST | `/api/experiments/evaluate` | JSON `{config: "quick" \| "default"}` → job |
| POST | `/api/training/jobs` | JSON `{config: "default" \| "smoke", epochs?, loss?, lr?, batch_size?, resume?}` → job |
| GET | `/api/training/jobs`, `/api/training/jobs/{id}` | Job state |
| POST | `/api/training/jobs/{id}/cancel` | Cooperative cancellation |
| GET | `/api/jobs`, `/api/jobs/{id}`; POST `/api/jobs/{id}/cancel` | Any background job |

Job object: `{id, kind, status: queued|running|completed|failed|cancelled|interrupted, progress (0–1),
message, config, result, error, created_at, started_at, finished_at}`.

## Settings and health

| Method | Path | Description |
|---|---|---|
| GET / PUT | `/api/settings` | UI language, theme, default mode, algorithm, extractor, fusion, ASR model (`auto` default, or a size) / language, recording limits, top-k, history, retention, `online_search_enabled` (default true), `external_provider_enabled` (consent, default false), `provider_policy` (`fallback` \| `always`) |
| GET | `/api/health` | Version, song counts, ASR readiness, neural model, device, ffmpeg, limits |

## Limits (environment variables)

`MELODYAI_MAX_UPLOAD_MB` (20), `MELODYAI_MAX_REFERENCE_MB` (60), `MELODYAI_MAX_QUERY_SECONDS` (60),
`MELODYAI_MIN_QUERY_SECONDS` (0.5), `MELODYAI_MAX_CONCURRENT_RECOGNITIONS` (2), bulk import ≤ 200 files.
