"""Prompt construction for grounded, cited tender Q&A (per-tender + cross-corpus)."""
from __future__ import annotations

import re

from app.models import Tender
from app.services.retriever import RetrievedChunk

# Rule shared by both modes: structured fields are authoritative; never swap one
# date/field for another, and say "not available" when a field is missing.
_METADATA_RULE = (
    "Dates, value, status, contacts and other facts are authoritative in the TENDER "
    "METADATA section (which includes an 'Other details' list of this tender's own "
    "fields). Match the user's wording to the right field by MEANING, not exact "
    "words - people use many synonyms for the same field:\n"
    "- Closing date = ending date, end date, deadline, submission/bid deadline, due "
    "date, last date to submit, bid closing, when bids are due.\n"
    "- Published / issue date = date published, date posted, advertised date, "
    "publication/release date, date of issue, when it was advertised.\n"
    "- Bid opening date = opening date, opening of bids, tender/bid opening, when "
    "bids are opened (this is NOT the published date and NOT the closing date).\n"
    "- Estimated value = budget, contract value, tender amount, estimated cost.\n"
    "- Briefing = site meeting, pre-bid or clarification meeting, compulsory "
    "briefing session.\n"
    "- Contact = contact person, who to contact, contact email/phone.\n"
    "These are SEPARATE fields: give the one the user means and never swap one for "
    "another. If the matching field is not present in the metadata, say it is not "
    "available for this tender - do not substitute a different field or guess."
)

SYSTEM = (
    "You are a procurement/tender assistant. Answer the user's question using ONLY "
    "the tender metadata and the numbered document excerpts provided. "
    + _METADATA_RULE + " "
    "Cite the source for each fact as [<document> p.<page>] using the tags shown on "
    "the excerpts. If the answer is not in the provided context, say you could not "
    "find it in this tender's documents - do not use outside knowledge. Be concise "
    "and specific; prefer lists for requirements, dates, or required documents."
)

SYSTEM_CROSS = (
    "You are a procurement/tender assistant answering across a corpus of MANY "
    "tenders from different portals. Use ONLY the numbered excerpts provided. Each "
    "excerpt is tagged with the tender it belongs to and its document/page. "
    + _METADATA_RULE + " "
    "In your answer, make clear WHICH tender each fact comes from, and cite sources "
    "as [<tender> | <document> p.<page>]. If the excerpts do not answer the "
    "question, say so - do not guess. Be concise; use lists when comparing tenders."
)


# Keys already shown in the canonical block, or too noisy to repeat.
_SKIP_KEYS = {
    "title", "name_of_tender", "tender_number", "tender_reference_number",
    "reference", "reference_number", "description", "description_of_the_bid",
    "status", "tender_status", "category", "tender_category", "closing_date",
    "closing_date_and_time", "deadline_for_bid_submission", "name_of_institution",
    "procuring_entity", "buyer", "organization", "tender_unique_id",
    "app_reference_number", "resource_id", "webpage", "attachment_base",
    "tender_id", "ocid", "offer_no",
}
# Flat keys some scrapers keep directly on the tender dict (not in a sub-dict).
_FLAT_KEYS = (
    "bid_opening_date", "opening_of_bids", "clarification_deadline",
    "end_of_clarification_period", "briefing_date", "briefing_venue",
    "briefing_details", "contact_name", "contact_person", "contact_email",
    "contact_phone", "contact", "province", "delivery_location",
    "procurement_method", "procedure", "evaluation_mechanism", "payment_amount",
    "payment_terms", "document_price", "bid_security_type", "award_criteria",
    "date_of_issue", "publication_date", "date_published", "special_conditions",
    "bid_opening", "opening_date",
)


def _norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_")


def _extra_fields(raw) -> list[tuple[str, str]]:
    """
    The tender's OWN scraped fields (bid opening date, briefing date, contacts,
    clarification deadline, payment, ...) pulled from raw_json - so the assistant
    can answer about them even though they aren't canonical columns.
    """
    if not isinstance(raw, dict):
        return []
    t = raw.get("tender") if isinstance(raw.get("tender"), dict) else raw
    collected: dict[str, str] = {}
    for container in (t.get("all_fields"), t.get("fields"), raw.get("inline_fields")):
        if isinstance(container, dict):
            for k, v in container.items():
                if isinstance(v, str) and v.strip():
                    collected.setdefault(_norm_key(k), v.strip())
    for k in _FLAT_KEYS:
        v = t.get(k)
        if isinstance(v, str) and v.strip():
            collected.setdefault(_norm_key(k), v.strip())

    out: list[tuple[str, str]] = []
    seen_vals: set[str] = set()
    for k, v in collected.items():
        if k in _SKIP_KEYS:
            continue
        val = v[:180]
        if val in seen_vals:                       # drop exact-duplicate values
            continue
        seen_vals.add(val)
        out.append((k.replace("_", " ").capitalize(), val))
    return out[:18]


def _metadata_block(t: Tender) -> str:
    fields = [
        ("Tender number", t.tender_number),
        ("Title", t.title),
        ("Organization", t.organization),
        ("Category", t.category),
        ("Country", t.country),
        ("Status", t.status),
        ("Date published / issued", t.issue_date),
        ("Closing date", t.closing_date),
        ("Estimated value",
         f"{t.estimated_value_amount} {t.estimated_value_currency}"
         if t.estimated_value_amount else None),
        ("Source website", t.source_website),
    ]
    present = {k for k, v in fields if v}
    lines = [f"- {k}: {v}" for k, v in fields if v]
    # Be explicit about missing date fields so the model won't substitute.
    if "Date published / issued" not in present:
        lines.append("- Date published / issued: not available for this tender")
    if "Closing date" not in present:
        lines.append("- Closing date: not available for this tender")

    # The tender's own scraped fields (opening date, briefing, contacts, ...).
    extra = _extra_fields(t.raw_json)
    if extra:
        lines.append("Other details from this tender's page:")
        lines.extend(f"- {label}: {val}" for label, val in extra)

    desc = (t.description or "").strip()
    if desc:
        lines.append(f"- Description: {desc[:1200]}")
    return "\n".join(lines) or "(no structured metadata)"


def build_user_prompt(tender: Tender, chunks: list[RetrievedChunk],
                      question: str) -> str:
    ctx = []
    for i, c in enumerate(chunks, 1):
        tag = f"{c.document_name or 'document'} p.{c.page_number}" \
            if c.page_number else (c.document_name or "document")
        ctx.append(f"[{i}] [{tag}]\n{c.text.strip()}")
    context = "\n\n".join(ctx) or "(no relevant excerpts were found)"
    return (f"TENDER METADATA\n{_metadata_block(tender)}\n\n"
            f"DOCUMENT EXCERPTS\n{context}\n\n"
            f"QUESTION\n{question.strip()}\n\n"
            f"Answer using only the above. Cite sources as [<document> p.<page>].")


def build_cross_prompt(chunks: list[RetrievedChunk], question: str) -> str:
    # Which tenders are represented, so the model can name them.
    tenders: dict[str, str] = {}
    for c in chunks:
        label = c.tender_number or c.tender_title or c.tender_id
        tenders.setdefault(c.tender_id, f"{label} [{c.source}]")
    involved = "\n".join(f"- {v}" for v in tenders.values()) or "(none)"

    ctx = []
    for i, c in enumerate(chunks, 1):
        tname = c.tender_number or c.tender_title or c.tender_id
        page = f" p.{c.page_number}" if c.page_number else ""
        ctx.append(f"[{i}] [{tname} | {c.document_name or 'document'}{page}]\n"
                   f"{c.text.strip()}")
    context = "\n\n".join(ctx) or "(no relevant excerpts were found)"
    return (f"TENDERS IN CONTEXT\n{involved}\n\n"
            f"DOCUMENT EXCERPTS (across tenders)\n{context}\n\n"
            f"QUESTION\n{question.strip()}\n\n"
            f"Answer using only the above. Attribute each fact to its tender and "
            f"cite as [<tender> | <document> p.<page>].")
