"""
RAG orchestration, two modes:
  - per-tender: retrieve within one tender, prompt with its metadata.
  - cross-corpus: retrieve across all tenders, attribute answers to tenders.
Both: embed question -> pgvector top-K -> build grounded prompt -> LLM -> refs.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import settings
from app.models import Tender
from app.schemas import ChatResponse, Reference
from app.services import llm, prompts, reranker
from app.services.embeddings import embed_query
from app.services.retriever import RetrievedChunk, retriever


def _retrieve(db: Session, question: str, k: int,
              tender_pk: int | None) -> list[RetrievedChunk]:
    """Hybrid retrieval, then (optionally) cross-encoder rerank down to k."""
    qvec = embed_query(question)
    if settings.use_reranker:
        pool = retriever.retrieve(db, qvec, question,
                                  settings.rerank_candidates, tender_pk)
        return reranker.rerank(question, pool, k)
    return retriever.retrieve(db, qvec, question, k, tender_pk)


def _references(chunks: list[RetrievedChunk], cross: bool) -> list[Reference]:
    return [
        Reference(
            tender=(c.tender_number or c.tender_title or c.tender_id) if cross else None,
            document=c.document_name, page=c.page_number, score=c.score,
            snippet=(c.text[:240] + "…") if len(c.text) > 240 else c.text)
        for c in chunks
    ]


def answer_question(db: Session, tender: Tender, question: str,
                    top_k: int | None = None) -> ChatResponse:
    """Per-tender: answer from one tender's metadata + documents."""
    k = top_k or settings.top_k
    chunks = _retrieve(db, question, k, tender_pk=tender.id)
    if not chunks:
        return ChatResponse(
            mode="tender", tender_id=tender.tender_id, question=question,
            answer="I couldn't find indexed content for this tender to answer "
                   "from. Has it been ingested?",
            references=[], chunks_used=0)
    answer = llm.chat(prompts.SYSTEM,
                      prompts.build_user_prompt(tender, chunks, question))
    return ChatResponse(mode="tender", tender_id=tender.tender_id,
                        question=question, answer=answer,
                        references=_references(chunks, cross=False),
                        chunks_used=len(chunks))


# The brief used to retrieve on ONE query naming the title and all eight headings.
# That failed twice over: the dense vector came out smeared across eight subjects,
# AND websearch_to_tsquery ANDs its terms, so a query that long matched zero chunks
# and silenced the keyword leg entirely — the very leg that finds literal strings
# like "Tender Document Fees: 100.00". Result: headings reported "Not stated" while
# the facts sat unretrieved one page away.
#
# So: one SHORT, single-concept query per heading. Short keeps the keyword leg alive
# (verified: "tender document fee" matches 3 chunks, the old long form matched 0) and
# keeps each vector sharp. No title prefix — retrieval is already scoped by tender_pk,
# so the title only dilutes both legs.
_SUMMARY_ASPECTS = (
    "tender document fee",
    "bid security deposit",
    "estimated value budget",
    "closing date submission deadline",
    "bid opening",
    "briefing session site visit",
    "eligibility criteria",
    "documents to submit",
    "evaluation method basis of award",
    "scope of work",
    "contract period",
    "contact person enquiries",
)
_PER_ASPECT = 3          # chunks pulled per aspect before de-duplication
_SUMMARY_BUDGET = 20     # cap on the union, to keep the prompt affordable


def summarize(db: Session, tender: Tender) -> ChatResponse:
    """Grounded brief of ONE tender: what it is, dates, fees, eligibility, docs..."""
    per_aspect = [_retrieve(db, aspect, _PER_ASPECT, tender_pk=tender.id)
                  for aspect in _SUMMARY_ASPECTS]
    # Round-robin by rank, not by aspect: every aspect contributes its #1 hit before
    # any aspect contributes its #2, so no heading can be crowded out by another.
    seen: set[int] = set()
    chunks: list[RetrievedChunk] = []
    for rank in range(_PER_ASPECT):
        for hits in per_aspect:
            if rank < len(hits) and len(chunks) < _SUMMARY_BUDGET:
                c = hits[rank]
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    chunks.append(c)
    # Document order reads better than relevance order in a brief.
    chunks.sort(key=lambda c: (c.document_id, c.page_number or 0))
    if not chunks:
        return ChatResponse(
            mode="summary", tender_id=tender.tender_id, question="(summary)",
            answer="I couldn't find indexed content for this tender to summarise.",
            references=[], chunks_used=0)
    answer = prompts.strip_uncovered(
        llm.chat(prompts.SYSTEM_SUMMARY,
                 prompts.build_summary_prompt(tender, chunks)))
    return ChatResponse(mode="summary", tender_id=tender.tender_id,
                        question="(summary)", answer=answer,
                        references=_references(chunks, cross=False),
                        chunks_used=len(chunks))


def answer_across_corpus(db: Session, question: str,
                         top_k: int | None = None) -> ChatResponse:
    """Cross-corpus: search every tender, attribute the answer to tenders."""
    k = top_k or settings.top_k
    chunks = _retrieve(db, question, k, tender_pk=None)
    if not chunks:
        return ChatResponse(
            mode="corpus", tender_id=None, question=question,
            answer="I couldn't find relevant content in any ingested tender.",
            references=[], chunks_used=0)
    answer = llm.chat(prompts.SYSTEM_CROSS,
                      prompts.build_cross_prompt(chunks, question))
    return ChatResponse(mode="corpus", tender_id=None, question=question,
                        answer=answer, references=_references(chunks, cross=True),
                        chunks_used=len(chunks))
