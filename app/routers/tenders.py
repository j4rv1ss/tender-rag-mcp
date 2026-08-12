"""GET /tenders — what is loaded and answerable."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Tender
from app.schemas import TenderRow

router = APIRouter(tags=["tenders"])


@router.get("/tenders", response_model=list[TenderRow])
def list_tenders(db: Session = Depends(get_db)) -> list[TenderRow]:
    rows = db.execute(
        select(Tender).order_by(Tender.source, Tender.tender_id)).scalars()
    return [TenderRow(source=t.source, tender_id=t.tender_id,
                      tender_number=t.tender_number, title=t.title) for t in rows]
