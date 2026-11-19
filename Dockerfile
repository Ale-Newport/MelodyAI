# CPU image: classical engine + API + built frontend. Whisper models are not baked in.
# Build: docker build -t melodyai .   Run: docker compose up (binds to 127.0.0.1:8000 only)
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY melodyai ./melodyai
COPY backend ./backend
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[asr]" -e backend faiss-cpu tensorboard matplotlib
COPY configs ./configs
COPY models ./models
COPY --from=web /web/dist ./frontend/dist
# Run as an unprivileged user; /data (the catalog store) is the only writable location.
RUN useradd --create-home --uid 10001 melodyai && mkdir -p /data/store && chown -R melodyai:melodyai /data /app/models
USER melodyai
ENV MELODYAI_STORE_DIR=/data/store MELODYAI_DATA_DIR=/data MELODYAI_EXPERIMENTS_DIR=/data/experiments \
    MELODYAI_DEVICE=cpu NUMBA_CACHE_DIR=/tmp/numba-cache
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)" || exit 1
# Import the demo catalog once per volume (a marker file, since other commands create catalog.db empty).
CMD ["sh", "-c", "[ -f /data/store/.demo-imported ] || { python -m melodyai.cli demo-data && touch /data/store/.demo-imported; }; exec uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000"]
