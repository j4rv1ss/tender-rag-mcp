"""Seed the POC: ingest one tender from the scraper output (default etenders/162660)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal          # noqa: E402
from app.services import ingest_service  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest one tender into the RAG store.")
    ap.add_argument("--source", default="etenders")
    ap.add_argument("--tender-id", default="162660")
    args = ap.parse_args()

    with SessionLocal() as db:
        res = ingest_service.ingest_tender(db, args.source, args.tender_id)
    print(f"ingested {res.source}/{res.tender_id}: pk={res.tender_pk} "
          f"documents={res.documents} chunks={res.chunks} embedded={res.embedded} "
          f"(reingested={res.reingested})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
