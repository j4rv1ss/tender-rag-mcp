"""
Load all locally-scraped tenders into the configured database, embedding with the
configured provider.

Run this FROM YOUR PC with .env pointed at the CLOUD database (Neon) and
GOOGLE_API_KEY set, to populate the cloud with Gemini (768-dim) embeddings:

    # one-time: create the tables + pgvector extension in the cloud DB
    python scripts/load_data.py --init

    # load every locally-scraped tender_*.json into the cloud DB
    python scripts/load_data.py

It reads the scraper output from SCRAPERS_ROOT (your local disk) and writes the
tenders + Gemini embeddings to whatever DATABASE the .env points at. The deployed
app then serves questions over that data (query-only, no scraping in the cloud).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# allow running as `python scripts/load_data.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text                                    # noqa: E402

from app.config import settings                                # noqa: E402
from app.db import SessionLocal, engine                        # noqa: E402
from app.services.ingest_service import ingest_all             # noqa: E402


def _statements(sql: str) -> list[str]:
    """Split init.sql into statements. Strip line comments FIRST — a ';' can appear
    inside a '-- ...' comment and must not be treated as a statement separator."""
    no_comments = "\n".join(re.sub(r"--.*$", "", ln) for ln in sql.splitlines())
    return [s.strip() for s in no_comments.split(";") if s.strip()]


def reset_schema() -> None:
    """Drop the app tables so init can recreate them (needed when the embedding
    dimension changes, e.g. switching providers)."""
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS chunks, documents, tenders CASCADE"))
    print("dropped existing tables")


def init_schema() -> None:
    path = Path(__file__).resolve().parent.parent / "db" / "init.sql"
    sql = path.read_text(encoding="utf-8")
    # The VECTOR(n) column must match the active provider's embed_dim.
    sql = re.sub(r"VECTOR\(\d+\)", f"VECTOR({settings.embed_dim})", sql)
    with engine.begin() as conn:
        for stmt in _statements(sql):
            conn.execute(text(stmt))
    print(f"schema ready (tables + pgvector, embedding VECTOR({settings.embed_dim}))")


def main() -> None:
    print(f"DB    : {settings.database_url_safe}")
    print(f"Embed : {settings.embed_provider} ({settings.active_embed_model}, "
          f"{settings.embed_dim}-dim)")

    if "--reset" in sys.argv:
        reset_schema()
    if "--init" in sys.argv or "--reset" in sys.argv:
        init_schema()

    db = SessionLocal()
    try:
        result = ingest_all(db)
    finally:
        db.close()

    t = result["totals"]
    print(f"\nLoaded: {t['tenders']} tenders, {t['documents']} documents, "
          f"{t['chunks']} chunks")
    for s in result["skipped"]:
        print("  skipped:", s)


if __name__ == "__main__":
    main()
