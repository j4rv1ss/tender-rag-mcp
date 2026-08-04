# Tender RAG — app image for free hosting (Render / Fly / etc.).
# Cloud mode: in-process fastembed embeddings (no Ollama, no embedding API/quota)
# + Groq (chat) + a managed Postgres. Scraping is done on your PC, not here.
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

# Render/Fly inject $PORT; default to 8000 for a local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
