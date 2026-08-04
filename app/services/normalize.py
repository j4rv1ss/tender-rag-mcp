"""
Per-source mappers: each scraper writes a different JSON shape; an adapter maps
it to the one canonical tender + document shape the DB stores. The full original
JSON is kept in tenders.raw_json, so nothing is lost.

Document containers differ by scraper — "documents" (most), "attachments"
(transnet), or a single "document" (ppadb, randwater) — handled by _documents().
No PDF parsing: the scraper already extracted text per page, and we reuse
extraction.pages[] for page-aware chunking.

To add a portal: write an adapter (raw, fallback_id) -> (tender|None, docs) and
register it in ADAPTERS.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from app.config import settings


class TenderNotFound(Exception):
    pass


# ---------------------------------------------------------------- helpers

CANON = ["tender_number", "title", "organization", "description", "category",
         "issue_date", "closing_date", "estimated_value_amount",
         "estimated_value_currency", "tender_url", "status"]


def canonical(source: str, tender_id: str, website: str, country: str | None,
              raw: dict, **kw: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"source": source, "tender_id": str(tender_id),
                         "source_website": website, "country": country,
                         "raw_json": raw}
    for k in CANON:
        d[k] = kw.get(k)
    return d


def _documents(raw: dict) -> list[dict[str, Any]]:
    """
    Normalise the scraper's document list regardless of container key, keeping
    only documents that carry extracted text (extraction.pages for chunking).
    """
    items: list | None = None
    for key in ("documents", "attachments"):
        if isinstance(raw.get(key), list):
            items = raw[key]
            break
    if items is None and isinstance(raw.get("document"), dict):
        items = [raw["document"]]
    if not items:
        return []

    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        ext = it.get("extraction")
        if not ext:                       # skipped / failed / gated docs
            continue
        fpath = it.get("file") or ""
        name = (Path(fpath).name if fpath else None) \
            or it.get("filename") or it.get("title") or "document"
        out.append({
            "file_name": name,
            "file_type": it.get("format") or it.get("file_type"),
            "original_path": fpath or None,
            "title": it.get("title"),
            "url": it.get("url"),
            "extracted_text": ext.get("text"),
            "page_count": ext.get("page_count"),
            "method": ext.get("method"),
            "pages": ext.get("pages") or [],
        })
    return out


def _join(*parts: str | None) -> str | None:
    vals = [p for p in parts if p]
    return " / ".join(vals) or None


# ---------------------------------------------------------------- adapters

def etenders(raw: dict, fid: str):
    t = raw.get("tender") or {}
    amt = t.get("value_amount")
    return canonical("etenders", t.get("tender_id") or fid,
                     "etenders.gov.za", "South Africa", raw,
                     tender_number=t.get("tender_number"),
                     title=t.get("tender_number") or t.get("description"),
                     organization=t.get("buyer"),
                     description=t.get("description"),
                     category=t.get("category"),
                     issue_date=t.get("publication_date"),
                     closing_date=t.get("closing_date"),
                     estimated_value_amount=amt if isinstance(amt, (int, float)) else None,
                     estimated_value_currency=t.get("value_currency"),
                     status=t.get("status")), _documents(raw)


def transnet(raw: dict, fid: str):
    t = raw.get("tender") or {}
    if not t:
        return None, []
    return canonical("transnet", t.get("tender_id") or fid,
                     "transnetetenders.azurewebsites.net", "South Africa", raw,
                     tender_number=t.get("tender_reference_number"),
                     title=t.get("name_of_tender") or t.get("description"),
                     organization=t.get("name_of_institution"),
                     description=t.get("description"),
                     category=t.get("tender_category") or t.get("tender_type"),
                     issue_date=t.get("date_published"),
                     closing_date=t.get("closing_date"),
                     status=t.get("tender_status")), _documents(raw)


def sadc(raw: dict, fid: str):
    if not (raw.get("title") or raw.get("node_id")):
        return None, []
    inl = raw.get("inline_fields") or {}
    return canonical("sadc", raw.get("node_id") or raw.get("slug") or fid,
                     "sadc.int", "SADC region", raw,
                     tender_number=inl.get("reference_number"),
                     title=raw.get("title"),
                     organization=inl.get("procurement_entity"),
                     description=raw.get("body_text"),
                     closing_date=raw.get("closing_date"),
                     status=raw.get("state")), _documents(raw)


def zppa(raw: dict, fid: str):
    t = raw.get("tender") or {}
    if not t:
        return None, []
    return canonical("zppa", t.get("resource_id") or fid,
                     "eprocure.zppa.org.zm", "Zambia", raw,
                     tender_number=t.get("tender_unique_id") or t.get("app_reference_number"),
                     title=t.get("title"),
                     organization=t.get("procuring_entity"),
                     description=t.get("description"),
                     category=t.get("procurement_type"),
                     issue_date=t.get("publication_date"),
                     closing_date=t.get("closing_date")), _documents(raw)


def ppadb(raw: dict, fid: str):
    t = raw.get("tender") or {}
    if not t:
        return None, []
    return canonical("ppadb", t.get("tender_no") or fid,
                     "ipms.ppadb.co.bw", "Botswana", raw,
                     tender_number=t.get("tender_no"),
                     title=t.get("description"),
                     organization=t.get("department"),
                     description=t.get("description"),
                     category=raw.get("document_type"),
                     issue_date=t.get("publishing_date"),
                     status=t.get("status")), _documents(raw)


def randwater(raw: dict, fid: str):
    t = raw.get("tender") or {}
    if not t:
        return None, []
    return canonical("randwater", t.get("reference") or fid,
                     "randwater.co.za", "South Africa", raw,
                     tender_number=t.get("reference"),
                     title=t.get("description") or t.get("title"),
                     organization="Rand Water",
                     description=t.get("description"),
                     closing_date=t.get("closing_date")), _documents(raw)


def capetown(raw: dict, fid: str):
    t = raw.get("tender") or {}
    if not t:
        return None, []
    f = t.get("fields") or {}
    return canonical("capetown", t.get("tender_number") or fid,
                     "capetown.gov.za", "South Africa", raw,
                     tender_number=t.get("tender_number"),
                     title=f.get("description") or t.get("description"),
                     organization=_join(t.get("directorate"), t.get("department")),
                     description=f.get("description") or t.get("description"),
                     issue_date=t.get("posted_datetime") or t.get("posted_date"),
                     closing_date=t.get("closing_datetime") or t.get("closing_date"),
                     status=f.get("tender_status")), _documents(raw)


def nra(raw: dict, fid: str):
    t = raw.get("tender") or {}
    if not t:
        return None, []
    stats = t.get("stats") or {}
    return canonical("nra", raw.get("slug") or t.get("tender_no") or fid,
                     "nra.co.za", "South Africa", raw,
                     tender_number=t.get("tender_no"),
                     title=t.get("title"),
                     organization="SANRAL",
                     description=t.get("notice_heading") or t.get("title"),
                     issue_date=stats.get("last_updated") or stats.get("create_date")
                     ), _documents(raw)


def cpbn(raw: dict, fid: str):
    t = raw.get("tender") or {}
    if not t:                              # listing / dry-run files have tender=null
        return None, []
    return canonical("cpbn", t.get("reference_number") or fid,
                     "cpbn.com.na", "Namibia", raw,
                     tender_number=t.get("reference_number"),
                     title=t.get("title"),
                     organization=t.get("institution"),
                     description=t.get("description"),
                     category=t.get("category"),
                     issue_date=t.get("date_of_issue"),
                     closing_date=t.get("closing_date")), _documents(raw)


ADAPTERS: dict[str, Callable[[dict, str], tuple[dict | None, list[dict]]]] = {
    "etenders": etenders, "transnet": transnet, "sadc": sadc, "zppa": zppa,
    "ppadb": ppadb, "randwater": randwater, "capetown": capetown, "nra": nra,
    "cpbn": cpbn,
}


# ---------------------------------------------------------------- loading

def _fallback_id(path: Path) -> str:
    return re.sub(r"^tender_", "", path.stem)


def _tender_json_path(source: str, tender_id: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", str(tender_id))
    return settings.scrapers_root / source / "output" / f"tender_{stem}.json"


def normalize_raw(source: str, raw: dict, fallback_id: str
                  ) -> tuple[dict | None, list[dict]]:
    adapter = ADAPTERS.get(source)
    if adapter is None:
        raise TenderNotFound(f"no mapper for source {source!r}; "
                             f"known: {', '.join(ADAPTERS)}")
    return adapter(raw, fallback_id)


def load_and_normalize(source: str, tender_id: str) -> tuple[dict, list[dict]]:
    path = _tender_json_path(source, tender_id)
    if not path.exists():
        raise TenderNotFound(f"scraper output not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    tender, docs = normalize_raw(source, raw, _fallback_id(path))
    if tender is None:
        raise TenderNotFound(f"{source}/{tender_id} has no tender data to ingest")
    return tender, docs


def iter_tender_files() -> list[tuple[str, Path]]:
    """Every tender_*.json across all known sources, in a stable order."""
    out: list[tuple[str, Path]] = []
    for source in ADAPTERS:
        base = settings.scrapers_root / source / "output"
        if base.is_dir():
            for p in sorted(base.glob("tender_*.json")):
                out.append((source, p))
    return out
