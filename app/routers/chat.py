"""
POST /chat — RAG Q&A.
  - per-tender when source+tender_id given (auto-scrapes it if not ingested);
  - cross-corpus when tender_id omitted (searches all ingested tenders).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.schemas import ChatRequest, ChatResponse, SummaryRequest
from app.services import ingest_service, rag
from app.services.embeddings import EmbeddingError
from app.services.llm import LLMError
from app.services.scrape import ScrapeError

router = APIRouter(tags=["chat"])


def _load_tender(db: Session, source: str, tender_id: str, auto_fetch: bool):
    """Fetch/ensure a tender or raise the right HTTP error."""
    try:
        tender = ingest_service.ensure_tender(db, source, tender_id,
                                              allow_scrape=auto_fetch)
    except ScrapeError as e:
        raise HTTPException(status_code=502,
                            detail=f"couldn't fetch {source}/{tender_id}: {e}") from e
    if tender is None:
        reason = ("this server is query-only (scraping disabled), so only pre-loaded "
                  "tenders can be answered" if not settings.enable_scraping else
                  "unknown source, or auto_fetch off")
        raise HTTPException(
            status_code=404,
            detail=f"tender {source}/{tender_id} is not loaded ({reason}). Load it "
                   "first, or omit tender_id to search all loaded tenders.")
    return tender


@router.post("/summary", response_model=ChatResponse)
def summary(req: SummaryRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """A grounded brief of one tender: what it is, dates, fees, eligibility, docs."""
    try:
        tender = _load_tender(db, req.source, req.tender_id, req.auto_fetch)
        return rag.summarize(db, tender)
    except (EmbeddingError, LLMError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question is empty")
    try:
        if req.tender_id:
            source = req.source or "etenders"
            # In DB? else ingest-from-disk; else scrape on demand.
            try:
                tender = ingest_service.ensure_tender(
                    db, source, req.tender_id, allow_scrape=req.auto_fetch)
            except ScrapeError as e:
                raise HTTPException(status_code=502,
                                    detail=f"couldn't fetch {source}/{req.tender_id}: {e}") from e
            if tender is None:
                reason = ("this server is query-only (scraping disabled), so only "
                          "pre-loaded tenders can be answered"
                          if not settings.enable_scraping else
                          "unknown source, or auto_fetch off")
                raise HTTPException(
                    status_code=404,
                    detail=f"tender {source}/{req.tender_id} is not loaded and could "
                           f"not be fetched ({reason}). Omit tender_id to search all "
                           "loaded tenders, or load this one first.")
            return rag.answer_question(db, tender, req.question, req.top_k)
        # cross-corpus
        return rag.answer_across_corpus(db, req.question, req.top_k)
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
