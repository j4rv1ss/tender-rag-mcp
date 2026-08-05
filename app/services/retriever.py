"""
Retrieval over pgvector, with optional hybrid (vector + keyword) search.

- Vector search: cosine distance (<=>) over the HNSW index — great for meaning.
- Keyword search: Postgres full-text (tsvector/GIN) — nails exact tokens the dense
  vectors miss (tender numbers, form codes like "Form EXP-4.1", clause refs).
- Hybrid: run both, fuse their rankings with Reciprocal Rank Fusion (RRF), return
  the best k. Falls back to pure vector search if anything goes wrong (e.g. the tsv
  column isn't present yet).

Both modes take an optional tender_pk to scope to ONE tender (per-tender chat) or
search the whole corpus (cross-corpus). Every result carries its tender identity.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import settings
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


_HYBRID_SQL = text("""
WITH vec AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:qvec AS vector)) AS rnk
    FROM chunks
    WHERE (:tender_pk IS NULL OR tender_pk = :tender_pk)
    ORDER BY embedding <=> CAST(:qvec AS vector)
    LIMIT :n
),
kw AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rnk
    FROM chunks, websearch_to_tsquery('english', :qtext) AS q
    WHERE (:tender_pk IS NULL OR tender_pk = :tender_pk) AND tsv @@ q
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT :n
),
fused AS (
    SELECT COALESCE(v.id, k.id) AS id,
           COALESCE(1.0 / (:rrf + v.rnk), 0) + COALESCE(1.0 / (:rrf + k.rnk), 0) AS score
    FROM vec v FULL OUTER JOIN kw k ON v.id = k.id
)
SELECT c.id, c.tender_pk, c.document_id, c.chunk_text, c.page_number,
       d.file_name, t.source, t.tender_id, t.tender_number, t.title,
       (c.embedding <=> CAST(:qvec AS vector)) AS distance
FROM fused f
JOIN chunks c    ON c.id = f.id
JOIN documents d ON d.id = c.document_id
JOIN tenders t   ON t.id = c.tender_pk
ORDER BY f.score DESC
LIMIT :k
""")

_RRF_K = 60   # RRF damping constant (standard); larger = ranks matter less


class PgVectorRetriever:
    def retrieve(self, db: Session, query_vector: list[float], query_text: str,
                 k: int, tender_pk: int | None = None) -> list[RetrievedChunk]:
        """Hybrid (vector+keyword) when enabled, else vector-only; safe fallback."""
        if settings.hybrid_search and query_text.strip():
            try:
                return self._hybrid(db, query_vector, query_text, k, tender_pk)
            except Exception:                       # noqa: BLE001 - be resilient
                db.rollback()                       # clear the aborted read txn
        return self.search(db, query_vector, k, tender_pk)

    def _hybrid(self, db: Session, query_vector: list[float], query_text: str,
                k: int, tender_pk: int | None) -> list[RetrievedChunk]:
        qvec = "[" + ",".join(f"{float(x):.7f}" for x in query_vector) + "]"
        rows = db.execute(_HYBRID_SQL, {
            "qvec": qvec, "qtext": query_text, "tender_pk": tender_pk,
            "n": settings.retrieve_candidates, "rrf": _RRF_K, "k": k,
        }).all()
        return [RetrievedChunk(
            chunk_id=r.id, tender_pk=r.tender_pk, source=r.source,
            tender_id=r.tender_id, tender_number=r.tender_number,
            tender_title=r.title, document_id=r.document_id,
            document_name=r.file_name, page_number=r.page_number,
            text=r.chunk_text, score=round(1.0 - float(r.distance), 4))
            for r in rows]

    def search(self, db: Session, query_vector: list[float], k: int,
               tender_pk: int | None = None) -> list[RetrievedChunk]:
        """Pure vector (cosine) search."""
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
