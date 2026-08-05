"""
Structure-aware, page-aware chunking for tender documents.

The scraper splits each document into pages, so we chunk within each page and keep
the page_number (accurate citations). Within a page we split on the document's own
structure - section/clause headings (Section III, ITB 7.2, GCC 40.1, Form EXP-4.1,
ALL-CAPS headings) - so a chunk is a coherent clause or requirement, not a blind
500-token cut. Tables and eligibility/requirement blocks stay together: an oversized
but coherent block is kept whole (up to ~1.5x target) rather than split mid-table.
Only genuinely huge blocks are token-split (with overlap). Sizes use tiktoken.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from app.config import settings

_enc = tiktoken.get_encoding("cl100k_base")   # model-agnostic token sizing

# Structural keyword / clause-code prefixes common across tender documents.
_KEYWORD = re.compile(
    r"^(SECTION|PART|FORM|ANNEX|SCHEDULE|APPENDIX|CLAUSE|SUB-?CLAUSE|"
    r"ITB|GCC|SCC|BDS|PCC|GC|PC)\b", re.IGNORECASE)
# A numbered clause heading: "22.1 The Employer...", "1. Scope", "3.1 It is..."
_NUMBERED = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Za-z(\"']")


@dataclass
class Chunk:
    chunk_number: int
    chunk_text: str
    page_number: int | None


def _is_heading(line: str) -> bool:
    """True if this line starts a new structural section/clause."""
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if _KEYWORD.match(s) or _NUMBERED.match(s):
        return True
    # ALL-CAPS heading line (e.g. "ELIGIBILITY", "TECHNICAL REQUIREMENTS")
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4 and len(s) <= 80 \
            and sum(c.isupper() for c in letters) / len(letters) > 0.85:
        return True
    return False


def _looks_tabular(block: str) -> bool:
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if len(lines) < 3:
        return False
    cols = sum(1 for ln in lines if "\t" in ln or re.search(r"\S {2,}\S", ln))
    return cols / len(lines) > 0.4


def _blocks(text: str) -> list[str]:
    """Break page text into structural blocks (heading + its body)."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)   # drop control chars
    blocks: list[str] = []
    cur: list[str] = []
    for ln in text.split("\n"):
        if _is_heading(ln) and any(x.strip() for x in cur):
            blocks.append("\n".join(cur).strip())
            cur = [ln]
        else:
            cur.append(ln)
    if any(x.strip() for x in cur):
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b.strip()]


def _token_split(text: str, size: int, overlap: int) -> list[str]:
    """Split an oversized block into overlapping ~size-token pieces."""
    tokens = _enc.encode(text)
    pieces, step = [], max(1, size - overlap)
    for start in range(0, len(tokens), step):
        piece = _enc.decode(tokens[start:start + size]).strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(tokens):
            break
    if len(pieces) >= 2 and len(_enc.encode(pieces[-1])) < overlap:
        pieces[-2] = pieces[-2] + "\n" + pieces[-1]
        pieces.pop()
    return pieces


def _pack(blocks: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack structural blocks into ~size chunks, breaking at boundaries."""
    hard_max = int(size * 1.5)          # keep coherent blocks/tables whole up to this
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    def flush() -> None:
        nonlocal cur, cur_tok
        if cur:
            chunks.append("\n".join(cur).strip())
            cur, cur_tok = [], 0

    for b in blocks:
        bt = len(_enc.encode(b))
        if bt > hard_max and not _looks_tabular(b):
            flush()
            chunks.extend(_token_split(b, size, overlap))     # genuinely huge -> split
        elif bt > size:
            flush()
            chunks.append(b)                                  # keep whole (clause/table)
        else:
            if cur_tok + bt > size and cur:
                flush()
            cur.append(b)
            cur_tok += bt
    flush()
    return chunks


def _split_page(text: str, size: int, overlap: int) -> list[str]:
    text = re.sub(r"[ \t]{4,}", "   ", text)          # trim runaway spacing, keep hints
    if not text.strip():
        return []
    pieces = _pack(_blocks(text), size, overlap)
    # Drop pieces with no real words (e.g. a lone page number) - nothing to embed.
    return [p for p in pieces if re.search(r"[A-Za-z0-9]{2,}", p)]


def chunk_document(pages: list[dict], full_text: str | None = None) -> list[Chunk]:
    """
    pages: [{page, text}] from the scraper. Falls back to full_text as one
    page-less block if pages are missing.
    """
    size, overlap = settings.chunk_tokens, settings.chunk_overlap
    chunks: list[Chunk] = []
    n = 0

    if pages:
        for p in pages:
            page_no = p.get("page")
            for piece in _split_page(p.get("text") or "", size, overlap):
                n += 1
                chunks.append(Chunk(n, piece, page_no))
    elif full_text:
        for piece in _split_page(full_text, size, overlap):
            n += 1
            chunks.append(Chunk(n, piece, None))
    return chunks
