"""
Cross-encoder reranking (in-process, via fastembed).

After hybrid retrieval pulls a wide candidate pool, a cross-encoder re-scores each
candidate against the actual question - far more precise than the bi-encoder
similarity used for retrieval - and we keep only the best top_k for the LLM.

Runs locally with real accuracy gains; disabled on the free cloud (RAM) via
USE_RERANKER=false. Any failure falls back to the retrieval order, never breaks.
"""
from __future__ import annotations

from app.config import settings
from app.services.retriever import RetrievedChunk

_MODEL = None   # lazily loaded on first use (model load ~a few seconds, once)


def _model():
    global _MODEL
    if _MODEL is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _MODEL = TextCrossEncoder(model_name=settings.reranker_model,
                                  cache_dir=settings.fastembed_cache or None)
    return _MODEL


def warmup() -> None:
    """Load the model at startup so the first real query isn't slow."""
    if settings.use_reranker:
        _model()


def rerank(query: str, chunks: list[RetrievedChunk],
           top_k: int) -> list[RetrievedChunk]:
    """Reorder chunks by cross-encoder relevance to the query; return the top_k."""
    if not settings.use_reranker or not chunks:
        return chunks[:top_k]
    try:
        scores = list(_model().rerank(query, [c.text for c in chunks]))
    except Exception:   # noqa: BLE001 - a rerank failure must not break answering
        return chunks[:top_k]
    ranked = sorted(zip(scores, chunks), key=lambda sc: sc[0], reverse=True)
    return [c for _, c in ranked[:top_k]]
