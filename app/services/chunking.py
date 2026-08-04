"""
Page-aware chunking. The scraper already split each document into pages
(extraction.pages = [{page, text}]), so we chunk within each page and keep the
page_number — that makes citations accurate (answer -> document + page).

Chunks target ~chunk_tokens with ~chunk_overlap overlap, measured with tiktoken.
A tiny trailing chunk is merged back into the previous one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from app.config import settings

_enc = tiktoken.get_encoding("cl100k_base")   # model-agnostic token sizing


@dataclass
class Chunk:
    chunk_number: int
    chunk_text: str
    page_number: int | None


def _split_page(text: str, size: int, overlap: int) -> list[str]:
    """Split one page into overlapping ~size-token pieces on paragraph/word bounds."""
    # Drop control chars (form-feeds etc.) that would make degenerate chunks.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    tokens = _enc.encode(text)
    if len(tokens) <= size:
        pieces = [text]
    else:
        pieces = []
        step = max(1, size - overlap)
        for start in range(0, len(tokens), step):
            piece = _enc.decode(tokens[start:start + size]).strip()
            if piece:
                pieces.append(piece)
            if start + size >= len(tokens):
                break
        # Merge a tiny tail into its predecessor.
        if len(pieces) >= 2 and len(_enc.encode(pieces[-1])) < overlap:
            pieces[-2] = pieces[-2] + "\n" + pieces[-1]
            pieces.pop()
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
