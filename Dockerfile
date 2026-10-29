FROM python:3.11-slim

WORKDIR /app

# System dependencies: ffmpeg for video transcode/preview, ca-certificates for TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application source (excluding .env, storage, vendor via .dockerignore)
COPY . .

# Ensure storage directories exist
RUN mkdir -p storage/uploads storage/outputs

ENV PYTHONUNBUFFERED=1
ENV APP_ENV=production
ENV STORAGE_DIR=/app/storage

# Railway injects $PORT
EXPOSE 8001

# Chỉ chạy web server (tải video MXH chạy sync trong request, không cần Celery worker cho lần đầu)
CMD sh -c "python run.py --host 0.0.0.0 --port ${PORT:-8001} --no-worker"
