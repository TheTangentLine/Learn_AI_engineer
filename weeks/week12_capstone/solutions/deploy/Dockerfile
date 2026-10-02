# Course Copilot service: the Week 12 pipeline (retrieval, gate, answerer, guards) behind the Week 11 gateway.
# NO model weights and NO index are baked in: the Hugging Face cache (the embedder and the re-ranker) and the index directory are mounted read-only at run time.
# Build context: the directory that `build_context.py` assembles (copilot/, common/, llmapi/, requirements-copilot.txt, this file).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

RUN useradd --create-home --uid 10001 app
WORKDIR /app

# dependencies first: this layer is rebuilt only when the requirements change
# torch from the CPU-only index: the default Linux wheel may pull several gigabytes of CUDA libraries that this service never uses
COPY requirements-copilot.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-copilot.txt

COPY common ./common
COPY llmapi ./llmapi
COPY copilot ./copilot
# the embedding and re-ranking caches (SQLite) live next to common/: the one writable place
RUN mkdir -p /app/outputs && chown app:app /app/outputs

ENV COPILOT_INDEX_DIR=/data/index \
    HOST=0.0.0.0 \
    PORT=8000
# secrets (COPILOT_KEYS, COPILOT_ADMIN_TOKEN) are NOT baked in: they arrive as environment variables at run time

USER app
EXPOSE 8000

# healthy = the process answers /healthz; whether the index is loaded is /readyz's job
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "copilot.service"]
