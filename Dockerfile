# The graphRAG backend: one image for the API, the worker and the scheduler.
#
# The same image runs all three; docker-compose.yml picks the command. The API
# runs a single process on purpose: the rate limiter counts in memory, and a
# graph file is opened by one process at a time.
FROM python:3.14-slim

# Build tools for any dependency that has to compile from source.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app \
    PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface
WORKDIR /home/app/service

# The CPU build of torch first: left to itself, sentence-transformers pulls the
# CUDA build, about a gigabyte of GPU libraries a CPU container never uses.
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies before code, so this layer survives code changes.
COPY --chown=app pyproject.toml README.md ./
COPY --chown=app src ./src
COPY --chown=app examples ./examples
RUN pip install --no-cache-dir --user ".[backend,api,graph,embeddings,judge,sentry]"

# Fetch the embedding model at build time, so the first request does not wait
# on a download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Where graphs, built artifacts and each pod's downloaded copies live. The
# compose file mounts shared volumes over the first two.
RUN mkdir -p /home/app/data/graphs /home/app/data/artifacts /home/app/data/cache
ENV GRAPHRAG_STORE_PATH=/home/app/data/graphs/graph.lbug \
    GRAPHRAG_ARTIFACT_ROOT=/home/app/data/artifacts \
    GRAPHRAG_POD_CACHE_ROOT=/home/app/data/cache

# Listens on $PORT when the platform sets one (Cloud Run does), else 8000.
# exec replaces the shell, so uvicorn receives the platform's stop signal.
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn graphrag.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
