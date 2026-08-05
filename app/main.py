"""FastAPI app: routers, static chat page, exception handling."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import chat, health, ingest, tenders

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
log = logging.getLogger("tender_rag")

app = FastAPI(
    title="Tender RAG POC",
    version="0.1.0",
    description="Ask questions about a scraped tender. RAG over PostgreSQL + "
                "pgvector with a local Ollama LLM. Swagger below; chat UI at /.",
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(tenders.router)
app.include_router(chat.router)

_static = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(str(_static / "index.html"))


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": f"{type(exc).__name__}: {exc}"})


@app.on_event("startup")
def _startup() -> None:
    log.info("Tender RAG POC up | db=%s | chat=%s(%s) | embed=%s(%d)",
             settings.database_url_safe, settings.active_chat_model,
             settings.chat_provider, settings.embed_model, settings.embed_dim)
    # Warm the persistent embed connection + model so the first real query is
    # fast (a cold connection's first request costs ~2.5s).
    try:
        from app.services.embeddings import embed_query
        embed_query("warmup")
        from app.services import reranker
        reranker.warmup()                                 # load reranker model (if on)
        from app.services import llm
        llm.chat("You are a warmup.", "Reply with OK.")   # prime the LLM connection
        log.info("embedding + reranker + chat warmed up")
    except Exception as e:  # noqa: BLE001 - non-fatal
        log.warning("warmup skipped: %s", e)
