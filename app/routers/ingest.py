"""POST /ingest (one tender) and POST /ingest-all (every source's output)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import (FetchRequest, IngestAllResponse, IngestRequest,
                         IngestResponse)
from app.services import ingest_service, normalize
from app.services.embeddings import EmbeddingError
from app.services.scrape import ScrapeError

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, db: Session = Depends(get_db)) -> IngestResponse:
    try:
        return ingest_service.ingest_tender(db, req.source, req.tender_id)
    except normalize.TenderNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post("/fetch", response_model=IngestResponse)
def fetch(req: FetchRequest, db: Session = Depends(get_db)) -> IngestResponse:
    """Scrape a tender on demand (if not already on disk) and ingest it."""
    try:
        tender = ingest_service.ensure_tender(db, req.source, req.tender_id,
                                              allow_scrape=True)
    except ScrapeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    if tender is None:
        raise HTTPException(status_code=400,
                            detail=f"auto-scrape not supported for source {req.source!r}")
    # report the resulting document/chunk counts
    from sqlalchemy import func, select
    from app.models import Chunk, Document
    docs = db.execute(select(func.count(Document.id))
                      .where(Document.tender_pk == tender.id)).scalar()
    chunks = db.execute(select(func.count(Chunk.id))
                        .where(Chunk.tender_pk == tender.id)).scalar()
    return IngestResponse(source=tender.source, tender_id=tender.tender_id,
                          tender_pk=tender.id, documents=docs, chunks=chunks,
                          embedded=chunks, reingested=False)


@router.post("/ingest-all", response_model=IngestAllResponse)
def ingest_all(db: Session = Depends(get_db)) -> IngestAllResponse:
    """Ingest every tender_*.json across all scraper outputs. Slow (embeds all
    document chunks) — mind the CPU-only Ollama."""
    try:
        return IngestAllResponse(**ingest_service.ingest_all(db))
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
