"""Pydantic request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source: str = Field(default="etenders", examples=["etenders"])
    tender_id: str = Field(examples=["162660"])


class IngestResponse(BaseModel):
    source: str
    tender_id: str
    tender_pk: int
    documents: int
    chunks: int
    embedded: int
    reingested: bool


class TenderOut(BaseModel):
    source: str
    tender_id: str
    tender_number: str | None = None
    title: str | None = None
    organization: str | None = None
    description: str | None = None
    category: str | None = None
    country: str | None = None
    issue_date: str | None = None
    closing_date: str | None = None
    estimated_value_amount: float | None = None
    estimated_value_currency: str | None = None
    source_website: str | None = None
    tender_url: str | None = None
    status: str | None = None
    document_count: int | None = None


class ChatRequest(BaseModel):
    # Omit source+tender_id to ask across ALL ingested tenders (cross-corpus).
    source: str | None = Field(default=None, examples=["etenders"])
    tender_id: str | None = Field(default=None, examples=["162660"])
    question: str = Field(examples=["What documents are mandatory for this tender?"])
    top_k: int | None = None
    # If the tender isn't ingested, scrape it on demand first (per-tender only).
    auto_fetch: bool = True


class FetchRequest(BaseModel):
    source: str = Field(examples=["randwater"])
    tender_id: str = Field(examples=["RW10414443-26"])


class Reference(BaseModel):
    tender: str | None = None          # set in cross-corpus mode
    document: str | None = None
    page: int | None = None
    score: float
    snippet: str


class ChatResponse(BaseModel):
    mode: str                          # "tender" | "corpus"
    tender_id: str | None = None
    question: str
    answer: str
    references: list[Reference]
    chunks_used: int


class IngestAllResponse(BaseModel):
    totals: dict
    ingested: list[dict]
    skipped: list[str]
