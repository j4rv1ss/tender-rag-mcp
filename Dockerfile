# Tender RAG — app image for free hosting (Render / Fly / etc.).
# Cloud mode uses Gemini (embeddings) + Groq (chat) + a managed Postgres, so this
# image stays tiny: no Ollama, no scraper binaries. Scraping is done on your PC.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY db ./db

# Render/Fly inject $PORT; default to 8000 for a local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
