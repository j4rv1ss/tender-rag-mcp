"""
Vector retrieval over pgvector.

One search() serves both modes: pass tender_pk to scope to a single tender
(per-tender chat), or omit it to search the whole corpus (cross-corpus chat).
Every result carries its tender identity so cross-corpus answers can attribute
each fact to the right tender. Cosine distance (<=>) with the HNSW index.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, Tender


@dataclass
class RetrievedChunk:
    chunk_id: int
    tender_pk: int
    source: str
    tender_id: str
    tender_number: str | None
    tender_title: str | None
    document_id: int
    document_name: str | None
    page_number: int | None
    text: str
    score: float           # cosine similarity ~[0..1]; higher = closer


class PgVectorRetriever:
    def search(self, db: Session, query_vector: list[float], k: int,
               tender_pk: int | None = None) -> list[RetrievedChunk]:
        distance = Chunk.embedding.cosine_distance(query_vector)
        stmt = (
            select(Chunk, Document.file_name, Tender.source, Tender.tender_id,
                   Tender.tender_number, Tender.title, distance.label("distance"))
            .join(Document, Document.id == Chunk.document_id)
            .join(Tender, Tender.id == Chunk.tender_pk)
        )
        if tender_pk is not None:
            stmt = stmt.where(Chunk.tender_pk == tender_pk)
        stmt = stmt.order_by(distance).limit(k)

        out: list[RetrievedChunk] = []
        for chunk, file_name, source, tid, tnum, ttitle, dist in db.execute(stmt).all():
            out.append(RetrievedChunk(
                chunk_id=chunk.id, tender_pk=chunk.tender_pk, source=source,
                tender_id=tid, tender_number=tnum, tender_title=ttitle,
                document_id=chunk.document_id, document_name=file_name,
                page_number=chunk.page_number, text=chunk.chunk_text,
                score=round(1.0 - float(dist), 4)))
        return out


retriever = PgVectorRetriever()
