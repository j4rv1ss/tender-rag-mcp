"""POST /chat and POST /summary — the RAG endpoints the browser page calls.

Both are plain `def`, so FastAPI runs them in its threadpool. That matters: a
question costs seconds (embed -> retrieve -> rerank -> LLM) and must never block
the event loop that is also serving the MCP stream from the same process.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Tender
from app.schemas import ChatRequest, ChatResponse, SummaryRequest
from app.services import ingest_service, rag

router = APIRouter(tags=["chat"])

DEFAULT_SOURCE = "etenders"


def _resolve(db: Session, source: str | None, tender_id: str) -> Tender:
    """Load a tender, scraping it first if this server is allowed to."""
    src = (source or "").strip() or DEFAULT_SOURCE
    tender = ingest_service.ensure_tender(
        db, src, tender_id, allow_scrape=settings.enable_scraping)
    if tender is None:
        reason = ("this server is query-only (scraping disabled)"
                  if not settings.enable_scraping else "unknown source")
        raise HTTPException(
            status_code=404,
            detail=f"tender {src}/{tender_id} is not loaded ({reason}). Load it "
                   "first, or clear the tender id to search every loaded tender.")
    return tender


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Answer a question about one tender, or across the whole corpus.

    Omit (or blank) `tender_id` to search every loaded tender.
    """
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is empty")

    tender_id = (req.tender_id or "").strip()
    if not tender_id:
        return rag.answer_across_corpus(db, question, req.top_k)
    return rag.answer_question(db, _resolve(db, req.source, tender_id),
                               question, req.top_k)


@router.post("/summary", response_model=ChatResponse)
def summary(req: SummaryRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """A grounded brief of one tender: scope, dates, fees, eligibility, documents."""
    tender_id = (req.tender_id or "").strip()
    if not tender_id:
        raise HTTPException(status_code=422, detail="tender_id is required")
    return rag.summarize(db, _resolve(db, req.source, tender_id))
