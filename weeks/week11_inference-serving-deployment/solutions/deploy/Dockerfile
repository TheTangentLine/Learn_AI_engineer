# The API gateway: FastAPI + uvicorn, retrieval over baked-in markdown, NO model weights (those are served by llama-server in its own container).
# Build context: the directory that `build_context.py` assembles (llmapi/, docs/, requirements-api.txt, this file).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# a non-root user: a compromised API process should not own the container
RUN useradd --create-home --uid 10001 app
WORKDIR /app

# dependencies first: this layer is rebuilt only when requirements-api.txt changes
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY llmapi ./llmapi
COPY docs ./docs

ENV LLMAPI_DOCS_DIR=/app/docs \
    HOST=0.0.0.0 \
    PORT=8000
# secrets (LLMAPI_KEYS, LLMAPI_ADMIN_TOKEN) are NOT baked in: they arrive as environment variables at run time

USER app
EXPOSE 8000

# the container is healthy when the PROCESS answers; whether the model is ready is /readyz's job (the orchestrator's readiness probe)
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "llmapi"]
