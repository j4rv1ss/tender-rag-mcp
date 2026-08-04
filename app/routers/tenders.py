"""GET /tenders and /tenders/{tender_id}."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Document, Tender
from app.schemas import TenderOut

router = APIRouter(tags=["tenders"])


def _to_out(t: Tender, doc_count: int | None = None) -> TenderOut:
    return TenderOut(
        source=t.source, tender_id=t.tender_id, tender_number=t.tender_number,
        title=t.title, organization=t.organization, description=t.description,
        category=t.category, country=t.country, issue_date=t.issue_date,
        closing_date=t.closing_date,
        estimated_value_amount=float(t.estimated_value_amount)
        if t.estimated_value_amount is not None else None,
        estimated_value_currency=t.estimated_value_currency,
        source_website=t.source_website, tender_url=t.tender_url, status=t.status,
        document_count=doc_count)


@router.get("/tenders", response_model=list[TenderOut])
def list_tenders(db: Session = Depends(get_db)) -> list[TenderOut]:
    rows = db.execute(select(Tender).order_by(Tender.created_at.desc())).scalars().all()
    return [_to_out(t) for t in rows]


@router.get("/tenders/{tender_id}", response_model=TenderOut)
def get_tender(tender_id: str, source: str = "etenders",
               db: Session = Depends(get_db)) -> TenderOut:
    t = db.execute(select(Tender).where(Tender.source == source,
                                        Tender.tender_id == tender_id)
                   ).scalar_one_or_none()
    if t is None:
        raise HTTPException(status_code=404,
                            detail=f"tender {source}/{tender_id} not ingested")
    doc_count = db.execute(
        select(func.count(Document.id)).where(Document.tender_pk == t.id)).scalar()
    return _to_out(t, doc_count)
