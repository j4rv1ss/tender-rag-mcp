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
from app.services import llm, prompts
from app.services.embeddings import embed_query
from app.services.retriever import RetrievedChunk, retriever


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
    chunks = retriever.search(db, embed_query(question), k, tender_pk=tender.id)
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


def answer_across_corpus(db: Session, question: str,
                         top_k: int | None = None) -> ChatResponse:
    """Cross-corpus: search every tender, attribute the answer to tenders."""
    k = top_k or settings.top_k
    chunks = retriever.search(db, embed_query(question), k, tender_pk=None)
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
