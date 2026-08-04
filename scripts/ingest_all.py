"""Ingest every tender_*.json across all scraper outputs into the RAG store."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal          # noqa: E402
from app.services import ingest_service  # noqa: E402


def main() -> int:
    with SessionLocal() as db:
        res = ingest_service.ingest_all(db)
    t = res["totals"]
    print(f"\n=== ingested {t['tenders']} tenders, {t['documents']} documents, "
          f"{t['chunks']} chunks ===")
    for r in res["ingested"]:
        print(f"  {r['source']:10} {r['tender_id']:28} docs={r['documents']:2} "
              f"chunks={r['chunks']}")
    if res["skipped"]:
        print(f"\nskipped {len(res['skipped'])}:")
        for s in res["skipped"]:
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
