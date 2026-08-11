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
    "another. Also handle opposite or negated questions: if the user asks whether "
    "something is allowed, permitted, or required and the documents say it is NOT "
    "(or the reverse), reply with a clear Yes or No that matches the documents. If "
    "the matching field is not present in the metadata, say it is not available for "
    "this tender - do not substitute a different field or guess."
)

# How answers should be written: friendly formatting, real sentences, no raw dumps.
_STYLE_RULE = (
    "Write the answer as clear, complete sentences - never just paste a raw field "
    "value. Reformat dates and times into a readable form: render '28/08/2026 "
    "10:00:00' as '28 August 2026 at 10:00' (slash dates are day/month/year; drop "
    "the seconds). Show money with its currency (e.g. 'ZMW 1,000'). Use a short "
    "bulleted list only when the answer is naturally several items (e.g. required "
    "documents); otherwise reply in prose. Format the wording only - never change "
    "the underlying facts."
)

# Strict grounding: only the provided context, exact values, no guessing.
_GROUNDING_RULE = (
    "GROUNDING (strict - never break these):\n"
    "- Answer ONLY from the TENDER METADATA and the numbered DOCUMENT EXCERPTS in "
    "the user's message. Never use outside knowledge; never guess, never infer "
    "beyond what is written, never fill a gap with an assumption.\n"
    "- Use EXACT values from the source - dates, times, amounts, percentages, names "
    "and reference numbers copied verbatim. You may only reformat a date/amount for "
    "readability while keeping the same value.\n"
    "- Use the label the DOCUMENT itself uses for a field. If the user's term is not "
    "exactly what the source calls it, still answer but name the real field - e.g. "
    "'The document does not state a \"procurement method\"; the Evaluation Method is "
    "X.' Do not relabel a field as a term the source never uses. (Everyday date "
    "wordings - deadline = closing date, etc. - may be treated as the same field.)\n"
    "- If the answer is not present in the provided context, reply exactly: 'Not "
    "stated in the provided documents for this tender.' Never substitute a related "
    "or nearby fact to seem helpful."
)

# The labelled structure every answer must follow.
_FORMAT_RULE = (
    "OUTPUT FORMAT - reply using these labelled lines; omit any line that does not "
    "apply:\n"
    "Answer: <the direct answer with the exact value(s). For a single fact, keep it "
    "to 1-2 sentences. But when the question is about a heading/section or asks for "
    "a LIST - the contents, sections, sub-headings, clauses, requirements, "
    "documents, or criteria, or 'what does X include/contain' - enumerate EVERY "
    "relevant item present in the excerpts as a bullet list, listing the actual "
    "heading/sub-heading TITLES verbatim (e.g. under a section, list each of its "
    "sub-clause headings that appear) and nesting sub-items under their parent. Do "
    "NOT replace them with a prose summary, do NOT stop at the top level, and do "
    "NOT give only a few examples when more are present.>\n"
    "Details: <extra supporting specifics from the documents, only if they add value "
    "beyond the Answer>\n"
    "Sources: <document> p.<page>[; <document> p.<page> ...]  (write 'tender "
    "metadata' when the fact came from the metadata section)\n"
    "Confidence: High | Medium | Low  (High = stated verbatim in metadata or one "
    "excerpt; Medium = assembled from several excerpts; Low = only weakly implied)"
)

SYSTEM = (
    "You are a precise procurement/tender analyst answering questions about ONE "
    "tender.\n\n"
    + _GROUNDING_RULE + "\n\n" + _METADATA_RULE + "\n\n" + _STYLE_RULE + "\n\n"
    + _FORMAT_RULE
)

SYSTEM_CROSS = (
    "You are a precise procurement/tender analyst answering across a corpus of MANY "
    "tenders from different portals. Each excerpt is tagged with the tender it "
    "belongs to, so always make clear WHICH tender each fact comes from.\n\n"
    + _GROUNDING_RULE + "\n\n" + _METADATA_RULE + "\n\n" + _STYLE_RULE + "\n\n"
    + _FORMAT_RULE +
    "\nIn the Sources line for cross-tender answers, name the tender too: "
    "<tender> | <document> p.<page>."
)

SYSTEM_SUMMARY = (
    "You are a precise procurement analyst. Produce a factual BRIEF of ONE tender "
    "using ONLY the TENDER METADATA and the numbered DOCUMENT EXCERPTS provided.\n\n"
    + _GROUNDING_RULE + "\n\n"
    "Write the brief under the headings below. Give each fact in the source's own "
    "wording (reformat dates readably, show money with its currency) and add a page "
    "cite like [p.5] when it comes from an excerpt.\n"
    "OMIT what the sources do not cover: drop that item, and drop the whole heading "
    "when none of its items has a fact. Never pad with 'Not stated', 'N/A', "
    "'Not available' or similar - a heading's ABSENCE is how the brief says the "
    "documents do not cover it. A short brief of only known facts is the goal.\n"
    "Keep each heading to 1-2 lines; use short bullets for eligibility and "
    "documents.\n"
    "**<tender title>**\n"
    "- What it is for:\n"
    "- Buyer / organization:\n"
    "- Key dates: published; closing; bid opening; briefing / site visit\n"
    "- Fees & value: document fee; estimated value; bid security\n"
    "- Evaluation method & basis of award:\n"
    "- Contract period / delivery:\n"
    "- Who can bid (eligibility):\n"
    "- Documents to submit:\n"
    "- Contact:\n"
    "Invent nothing - every line must be traceable to the metadata or an excerpt."
)


# Models comply with "omit what you don't have" only most of the time, so the brief
# gets a deterministic second pass. The tell is the citation: every real fact carries
# one ([p.7] or [TENDER METADATA]), a padded line carries none. That distinction is
# what keeps a document's own "Bid security: Not Applicable [p.2]" while dropping an
# invented "Estimated value: Not stated in the provided documents".
_NO_INFO = re.compile(
    r"(?i)\b(not stated|not available|not specified|not provided|not mentioned|"
    r"not indicated|not disclosed|not given|not found|no information|unspecified|"
    r"unknown)\b")
# Cites come in several shapes - [p.5], [p.2, p.3], [1, p.1], [8, Excerpt 11],
# [TENDER METADATA] - so match the marker anywhere inside the brackets. Anchoring
# this to the start of the bracket would misread "[3, p.7]" as uncited and delete a
# real "Not Applicable [3, p.7]".
_CITED = re.compile(r"(?i)\[[^\]]*(?:p\.\s*\d+|tender metadata|excerpt)[^\]]*\]")
# "- Key dates:" — a label whose value is empty, i.e. it only had sub-bullets.
_EMPTY_LABEL = re.compile(r"^\s*[-*]\s*[^:]{1,60}:\s*$")


def strip_uncovered(brief: str) -> str:
    """Remove uncited 'Not stated'-style lines, then headings left with nothing."""
    kept = [ln for ln in brief.splitlines()
            if not (_NO_INFO.search(ln) and not _CITED.search(ln))]
    out: list[str] = []
    for i, line in enumerate(kept):
        if _EMPTY_LABEL.match(line):
            indent = len(line) - len(line.lstrip())
            nxt = next((k for k in kept[i + 1:] if k.strip()), "")
            if not nxt or (len(nxt) - len(nxt.lstrip())) <= indent:
                continue          # every sub-bullet was dropped; drop the heading
        out.append(line)
    return "\n".join(out).strip()


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
            f"Answer using ONLY the above, in the required "
            f"Answer / Details / Sources / Confidence format.")


def build_summary_prompt(tender: Tender, chunks: list[RetrievedChunk]) -> str:
    ctx = []
    for i, c in enumerate(chunks, 1):
        tag = f"{c.document_name or 'document'} p.{c.page_number}" \
            if c.page_number else (c.document_name or "document")
        ctx.append(f"[{i}] [{tag}]\n{c.text.strip()}")
    context = "\n\n".join(ctx) or "(no excerpts were found)"
    return (f"TENDER METADATA\n{_metadata_block(tender)}\n\n"
            f"DOCUMENT EXCERPTS\n{context}\n\n"
            f"Write the grounded tender brief for the above, using ONLY the metadata "
            f"and excerpts.")


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
            f"Answer using ONLY the above, in the required "
            f"Answer / Details / Sources / Confidence format, naming the tender for "
            f"each fact.")
