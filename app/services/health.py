"""Diagnostics: DB + pgvector, and whichever AI providers are actually in use.

Lives in services (not a transport layer) so the MCP server stays a thin adapter.
"""
from __future__ import annotations

import httpx
from sqlalchemy import text

from app.config import settings
from app.db import engine


def check() -> dict:
    """Probe every dependency the server needs; never raises."""
    report: dict = {"status": "ok", "checks": {}}

    def degrade() -> None:
        if report["status"] != "error":
            report["status"] = "degraded"

    # PostgreSQL + pgvector
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            has_vec = conn.execute(text(
                "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
            )).scalar()
        report["checks"]["postgres"] = "ok"
        report["checks"]["pgvector"] = "ok" if has_vec else "extension not created"
        if not has_vec:
            report["status"] = "degraded"
    except Exception as e:  # noqa: BLE001
        report["checks"]["postgres"] = f"error: {e}"
        report["status"] = "error"

    # Embeddings provider
    report["checks"]["embed_provider"] = settings.embed_provider
    if settings.embed_provider == "fastembed":
        # In-process model — verify it loads (and is baked/cached).
        try:
            from app.services.embeddings import embed_query
            report["checks"]["embed_model"] = (
                "ok" if len(embed_query("ping")) == settings.embed_dim else "bad dim")
        except Exception as e:  # noqa: BLE001
            report["checks"]["embed_model"] = f"error: {e}"
            report["status"] = "error"
    elif settings.embed_provider == "gemini":
        report["checks"]["google_key"] = "set" if settings.google_api_key.strip() else "MISSING"

    # Chat provider: cloud OpenAI-compatible endpoints (Llama API + Groq backup),
    # else the optional offline Ollama model.
    report["checks"]["chat_provider"] = settings.chat_provider
    if settings.chat_provider == "api":
        report["checks"]["llama_api_key"] = (
            "set" if settings.llama_api_key.strip() else "not set")
        report["checks"]["groq_backup_key"] = (
            "set" if settings.groq_api_key.strip() else "not set")
        report["checks"]["chat_endpoints"] = [
            f'{e["provider"]}:{e["model"]}' for e in settings.chat_endpoints]

    # Ollama — only relevant when it actually serves embeddings or chat.
    needs_ollama = (settings.embed_provider == "ollama"
                    or (settings.chat_provider == "ollama" and settings.chat_model.strip()))
    if needs_ollama:
        try:
            with httpx.Client(base_url=settings.ollama_url, timeout=5.0) as c:
                tags = c.get("/api/tags").json()
            names = {m.get("name", "").split(":")[0] for m in tags.get("models", [])}
            full = {m.get("name") for m in tags.get("models", [])}
            report["checks"]["ollama"] = "ok"

            def model_ok(name: str) -> bool:
                return name in full or name.split(":")[0] in names

            if settings.embed_provider == "ollama":
                report["checks"]["embed_model"] = (
                    "ok" if model_ok(settings.embed_model)
                    else f"missing — run: ollama pull {settings.embed_model}")
                if not model_ok(settings.embed_model):
                    degrade()
            if settings.chat_provider == "ollama" and settings.chat_model.strip():
                report["checks"]["chat_model"] = (
                    "ok" if model_ok(settings.chat_model)
                    else f"missing — run: ollama pull {settings.chat_model}")
                if not model_ok(settings.chat_model):
                    degrade()
        except Exception as e:  # noqa: BLE001
            report["checks"]["ollama"] = f"error: {e}"
            report["status"] = "error"

    report["config"] = {
        "database": settings.database_url_safe,
        "embed_provider": settings.embed_provider,
        "embed_model": settings.active_embed_model,
        "embed_dim": settings.embed_dim,
        "chat_provider": settings.chat_provider,
        "chat_model": settings.active_chat_model,
        "scraping": settings.enable_scraping,
        "top_k": settings.top_k,
    }
    return report


def warmup() -> None:
    """Load the embedding/reranker models and prime the LLM connection.

    A cold embed connection costs ~2.5s on its first real query; paying that
    up front keeps the first tool call fast. Non-fatal — raises nothing.
    """
    from app.services import llm, reranker
    from app.services.embeddings import embed_query

    embed_query("warmup")
    reranker.warmup()                                  # load reranker model (if on)
    llm.chat("You are a warmup.", "Reply with OK.")    # prime the LLM connection
