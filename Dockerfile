FROM python:3.11-slim

# -B keeps the image clean; unbuffered logs stream to the platform console.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so Docker caches this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code and pre-trained artifacts.
COPY app ./app
COPY artifacts ./artifacts
COPY data ./data
COPY train.py ./

# Run as a non-root user.
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=3)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
