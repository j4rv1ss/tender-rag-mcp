"""
Ingest scraper output into the DB + vector store:
  normalize -> upsert Tender -> insert Documents -> chunk + embed -> insert Chunks.

Idempotent per tender (re-ingest replaces its documents + chunks). Two entry
points: ingest_tender (one tender by id) and ingest_all (every tender_*.json
across all sources).
"""
from __future__ import annotations

import json
import logging

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk, Document, Tender
from app.schemas import IngestResponse
from app.services import normalize
from app.services.chunking import chunk_document
from app.services.embeddings import embed_texts

log = logging.getLogger("tender_rag.ingest")


def _ingest_normalized(db: Session, source: str, tender_data: dict,
                       doc_records: list[dict]) -> IngestResponse:
    tender = db.execute(
        select(Tender).where(Tender.source == source,
                             Tender.tender_id == tender_data["tender_id"])
    ).scalar_one_or_none()

    reingested = tender is not None
    if tender is None:
        tender = Tender(**tender_data)
        db.add(tender)
        db.flush()
    else:
        for key, value in tender_data.items():
            setattr(tender, key, value)
        db.execute(delete(Document).where(Document.tender_pk == tender.id))
        db.flush()

    n_docs = n_chunks = 0
    for rec in doc_records:
        pages = rec.get("pages", [])
        doc = Document(tender_pk=tender.id, **{k: rec.get(k) for k in (
            "file_name", "file_type", "original_path", "title", "url",
            "extracted_text", "page_count", "method")})
        db.add(doc)
        db.flush()
        n_docs += 1

        pieces = chunk_document(pages, rec.get("extracted_text"))
        if not pieces:
            continue
        vectors = embed_texts([p.chunk_text for p in pieces])
        for piece, vector in zip(pieces, vectors):
            db.add(Chunk(
                tender_pk=tender.id, document_id=doc.id,
                chunk_number=piece.chunk_number, chunk_text=piece.chunk_text,
                page_number=piece.page_number, embedding=vector,
                metadata_={"source": source, "tender_id": tender.tender_id,
                           "document": doc.file_name, "page": piece.page_number}))
            n_chunks += 1

    db.commit()
    return IngestResponse(
        source=source, tender_id=tender.tender_id, tender_pk=tender.id,
        documents=n_docs, chunks=n_chunks, embedded=n_chunks,
        reingested=reingested)


def ingest_tender(db: Session, source: str, tender_id: str) -> IngestResponse:
    tender_data, doc_records = normalize.load_and_normalize(source, tender_id)
    return _ingest_normalized(db, source, tender_data, doc_records)


def ingest_from_path(db: Session, source: str, path) -> IngestResponse:
    """Ingest directly from a specific scraper-output file (used after scraping)."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    tender_data, doc_records = normalize.normalize_raw(
        source, raw, normalize._fallback_id(path))
    if tender_data is None:
        raise normalize.TenderNotFound(f"{path.name} has no tender data to ingest")
    return _ingest_normalized(db, source, tender_data, doc_records)


def ensure_tender(db: Session, source: str, tender_id: str,
                  allow_scrape: bool = True) -> Tender | None:
    """
    Return the Tender for (source, tender_id), fetching it if needed:
      already in DB -> return it;
      scraped on disk but not ingested -> ingest then return;
      not present + allow_scrape -> run the scraper, ingest, return.
    Returns None if it can't be obtained (and scraping is off or unsupported).
    """
    from app.services import scrape

    existing = db.execute(
        select(Tender).where(Tender.source == source, Tender.tender_id == tender_id)
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    # scraped to disk already?
    path = normalize._tender_json_path(source, tender_id)
    if path.exists():
        res = ingest_tender(db, source, tender_id)
        return db.get(Tender, res.tender_pk)

    # Scraping needs local scraper binaries; disabled on cloud/query-only hosts.
    if not allow_scrape or not settings.enable_scraping or not scrape.can_scrape(source):
        return None

    scraped = scrape.scrape_tender(source, tender_id)   # may raise ScrapeError
    try:
        res = ingest_from_path(db, source, scraped)
    except normalize.TenderNotFound as e:
        raise scrape.ScrapeError(
            f"the {source} scraper ran but found no tender for '{tender_id}' - it "
            "may have closed and dropped off the portal, or the id/URL is wrong") from e
    return db.get(Tender, res.tender_pk)


def ingest_all(db: Session) -> dict:
    """Ingest every tender_*.json across all sources; skip files with no tender."""
    results: list[dict] = []
    skipped: list[str] = []
    totals = {"tenders": 0, "documents": 0, "chunks": 0}

    for source, path in normalize.iter_tender_files():
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            tender_data, doc_records = normalize.normalize_raw(
                source, raw, normalize._fallback_id(path))
        except Exception as e:  # noqa: BLE001 - a bad file shouldn't stop the batch
            skipped.append(f"{path.name}: {type(e).__name__}: {e}")
            continue
        if tender_data is None:
            skipped.append(f"{path.name}: no tender data")
            continue
        try:
            res = _ingest_normalized(db, source, tender_data, doc_records)
        except Exception as e:  # noqa: BLE001
            db.rollback()
            skipped.append(f"{path.name}: ingest error: {type(e).__name__}: {e}")
            continue
        log.info("ingested %s/%s: %d docs, %d chunks",
                 source, res.tender_id, res.documents, res.chunks)
        results.append({"source": source, "tender_id": res.tender_id,
                        "documents": res.documents, "chunks": res.chunks})
        totals["tenders"] += 1
        totals["documents"] += res.documents
        totals["chunks"] += res.chunks

    return {"totals": totals, "ingested": results, "skipped": skipped}
