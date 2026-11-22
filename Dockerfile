FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CORTEX_DB_PATH=/data/cortex.sqlite \
    CORTEX_WORKSPACE=/workspace \
    CORTEX_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY cortex ./cortex

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[serve]"

COPY scripts ./scripts
COPY configs ./configs
COPY assets ./assets

RUN mkdir -p /data /workspace

EXPOSE 8000

# Mount a trained checkpoint at /models and pass --model-dir /models/<run>.
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
