# Security policy

## Supported versions

| Version | Supported |
|---|---|
| latest release (`0.1.x`) and `main` | ✅ |
| older | ❌ |

## Reporting a vulnerability

Please **do not open a public issue**. Report it privately through GitHub:
[**Report a vulnerability**](https://github.com/Ale-Newport/MelodyAI/security/advisories/new)
(Security tab → *Report a vulnerability*).

Include the affected version or commit, how to reproduce it and the impact you expect. You can expect an
acknowledgement within 7 days and a status update at least every 14 days until it is resolved. Fixes are released
as a patch version and credited in the advisory unless you prefer otherwise.

## Security model

MelodyAI is a **local-first, single-user** application:

* The API has **no authentication** and is meant to listen on `127.0.0.1` only (the default for `make dev`, and
  `docker compose` publishes port 8000 on `127.0.0.1`). Do not expose it to a network you do not trust.
* Requests from other web sites are refused (`Origin` / `Sec-Fetch-Site` checks), and requests whose `Host` header is
  not an allowed local name are refused, which blocks DNS-rebinding attacks (`MELODYAI_ALLOWED_HOSTS`).
* Uploads are limited in size and duration, and durations are checked before decoding. `ffmpeg` only ever receives
  uploaded bytes on stdin, never a path from the client.
* Audio is processed locally and not stored unless the user allows it. Online song search sends only words (to
  LRCLIB and Deezer) and can be switched off (`MELODYAI_OFFLINE=1`).
* Optional keys (ACRCloud, Genius, Anthropic) are read from the environment or a git-ignored `.env`; never commit them.

In scope: the API, the web app, the engine's handling of untrusted audio, MIDI, lyrics and manifests, the Docker
image and the GitHub workflows. Out of scope: deployments that deliberately expose the API publicly, and
vulnerabilities in third-party models or datasets that MelodyAI does not redistribute.

## Repository safeguards

Secret scanning with push protection, Dependabot alerts and security updates, CodeQL, dependency review and weekly
`pip-audit` / `npm audit` runs are enabled. GitHub Actions use a read-only token by default and actions pinned to
commit SHAs.
