"""
The browser/REST front door — a FastAPI app mounted at /api by the MCP server.

MCP (app.mcp_server) serves AI assistants; this serves browsers and scripts.
Both are thin adapters over app.services, so the two interfaces cannot answer
differently. Mounting means one process, one port, same origin — no CORS.

    python -m app.mcp_server --http     ->  /  /api/*  /api/docs  /mcp  /healthz

Domain failures are translated to status codes here, once, rather than in every
route: the MCP side maps the same exceptions to ToolError in app.mcp_server._db.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routers import chat, health, ingest, tenders
from app.services.embeddings import EmbeddingError
from app.services.llm import LLMError
from app.services.normalize import TenderNotFound
from app.services.scrape import ScrapeError

log = logging.getLogger("tender_rag.web")

api = FastAPI(
    title="Tender RAG API",
    version="0.3.0",
    description="Ask questions about scraped public-procurement tenders. RAG over "
                "PostgreSQL + pgvector. The same corpus is exposed to AI assistants "
                "over MCP at /mcp.",
    # Mounted under /api, so docs land at /api/docs.
    docs_url="/docs",
    redoc_url=None,
)

api.include_router(health.router)
api.include_router(tenders.router)
api.include_router(ingest.router)
api.include_router(chat.router)


def _detail(status: int, message: str) -> JSONResponse:
    # {"detail": ...} matches FastAPI's own HTTPException shape, which is what
    # the chat page reads on every error path.
    return JSONResponse(status_code=status, content={"detail": message})


@api.exception_handler(EmbeddingError)
async def _embedding_down(request: Request, exc: EmbeddingError) -> JSONResponse:
    return _detail(503, f"embedding backend unavailable: {exc}")


@api.exception_handler(LLMError)
async def _llm_down(request: Request, exc: LLMError) -> JSONResponse:
    return _detail(503, f"chat model unavailable: {exc}")


@api.exception_handler(ScrapeError)
async def _scrape_failed(request: Request, exc: ScrapeError) -> JSONResponse:
    return _detail(502, f"scrape failed: {exc}")


@api.exception_handler(TenderNotFound)
async def _not_found(request: Request, exc: TenderNotFound) -> JSONResponse:
    return _detail(404, str(exc))


@api.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error on %s", request.url.path)
    return _detail(500, f"{type(exc).__name__}: {exc}")
