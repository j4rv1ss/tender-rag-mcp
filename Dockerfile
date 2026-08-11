# Tender RAG — MCP server image. Serves either transport:
#
#   stdio (default) — the container IS the server; a client launches it and
#     speaks JSON-RPC over stdin/stdout, so run it with -i and never print to
#     stdout:      docker run -i --rm --env-file .env tender-rag
#   HTTP (hosted)   — override the command; needs MCP_AUTH_TOKEN:
#     docker run -p 8000:8000 -e MCP_AUTH_TOKEN=... --env-file .env tender-rag \
#       python -m app.mcp_server --http --host 0.0.0.0
#
# Query-only by design: in-process fastembed embeddings + a cloud chat model +
# a managed Postgres. Scraping needs local binaries, so it stays on your PC.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_PROVIDER=fastembed \
    FASTEMBED_CACHE=/app/models

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so cold starts don't download it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/models')"

COPY app ./app
COPY db ./db

# No scraper binaries here: answers pre-loaded tenders only. Load the corpus
# from your PC with scripts/load_data.py.
ENV ENABLE_SCRAPING=false

# Only used in HTTP mode; hosts inject their own $PORT, which main() prefers.
ENV PORT=8000
EXPOSE 8000

# stdio by default — render.yaml overrides this with the --http command.
CMD ["python", "-m", "app.mcp_server"]
