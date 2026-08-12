"""POST /ingest — index a tender so it can be asked about.

Loads from the scraper output on disk, falling back to scraping the portal when
this server is configured for it (ENABLE_SCRAPING). Cloud hosts have no scraper
binaries, so there it is disk-only and 404s for anything not already loaded.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Chunk, Document
from app.schemas import IngestRequest, IngestResult
from app.services import ingest_service

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResult)
def ingest(req: IngestRequest, db: Session = Depends(get_db)) -> IngestResult:
    tender_id = (req.tender_id or "").strip()
    if not tender_id:
        raise HTTPException(status_code=422, detail="tender_id is required")
    source = (req.source or "").strip() or "etenders"

    tender = ingest_service.ensure_tender(
        db, source, tender_id, allow_scrape=settings.enable_scraping)
    if tender is None:
        raise HTTPException(
            status_code=404,
            detail=f"{source}/{tender_id} is not on disk and could not be fetched "
                   "(scraping disabled or source unsupported)")

    docs = db.execute(select(func.count(Document.id))
                      .where(Document.tender_pk == tender.id)).scalar()
    chunks = db.execute(select(func.count(Chunk.id))
                        .where(Chunk.tender_pk == tender.id)).scalar()
    return IngestResult(source=tender.source, tender_id=tender.tender_id,
                        documents=docs or 0, chunks=chunks or 0)
