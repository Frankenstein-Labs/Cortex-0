FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CORTEX_CHECKPOINT=/models/cortex-0-nano/best.safetensors \
    CORTEX_TOKENIZER=/models/tokenizer.json \
    CORTEX_MEMORY_DB=/data/cortex_memory.db \
    CORTEX_TASKS_DB=/data/cortex_tasks.db

WORKDIR /app

COPY pyproject.toml README.md ./
COPY cortex ./cortex
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[serve]"

RUN mkdir -p /models /data
VOLUME ["/models", "/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "cortex.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
