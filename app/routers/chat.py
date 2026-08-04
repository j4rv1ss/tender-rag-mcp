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
from app.schemas import ChatRequest, ChatResponse
from app.services import ingest_service, rag
from app.services.embeddings import EmbeddingError
from app.services.llm import LLMError
from app.services.scrape import ScrapeError

router = APIRouter(tags=["chat"])


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
