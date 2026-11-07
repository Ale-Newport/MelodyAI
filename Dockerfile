# CPU image: classical engine + API + built frontend. Whisper models are not baked in.
FROM node:20-slim AS web
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
ENV MELODYAI_STORE_DIR=/data/store MELODYAI_DEVICE=cpu
EXPOSE 8000
CMD ["sh", "-c", "python -m melodyai.cli status >/dev/null 2>&1; [ -f /data/store/catalog.db ] || python -m melodyai.cli demo-data; uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000"]
