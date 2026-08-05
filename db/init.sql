-- Tender RAG POC — schema (authoritative). Run once against the tender_rag DB.
-- Requires superuser for CREATE EXTENSION.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- tenders
CREATE TABLE IF NOT EXISTS tenders (
    id                       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source                   TEXT        NOT NULL,          -- 'etenders', 'sadc', ...
    tender_id                TEXT        NOT NULL,          -- business id from the portal
    tender_number            TEXT,
    title                    TEXT,
    organization             TEXT,
    description              TEXT,
    category                 TEXT,
    country                  TEXT,
    issue_date               TEXT,               -- source date strings vary; kept verbatim
    closing_date             TEXT,
    estimated_value_amount   NUMERIC,
    estimated_value_currency TEXT,
    source_website           TEXT,
    tender_url               TEXT,
    status                   TEXT,
    raw_json                 JSONB       NOT NULL,          -- full scraped tender JSON
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, tender_id)
);

-- ---------------------------------------------------------------- documents
CREATE TABLE IF NOT EXISTS documents (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tender_pk      BIGINT      NOT NULL REFERENCES tenders(id) ON DELETE CASCADE,
    file_name      TEXT,
    file_type      TEXT,
    original_path  TEXT,
    title          TEXT,
    url            TEXT,
    extracted_text TEXT,
    page_count     INTEGER,
    method         TEXT,                                    -- text_layer | ocr | hybrid
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_tender_pk_idx ON documents(tender_pk);

-- ---------------------------------------------------------------- chunks (pgvector)
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tender_pk    BIGINT      NOT NULL REFERENCES tenders(id) ON DELETE CASCADE,
    document_id  BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_number INTEGER     NOT NULL,
    chunk_text   TEXT        NOT NULL,
    page_number  INTEGER,
    embedding    VECTOR(768),                              -- set by the embed model
    metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    -- full-text vector for keyword (hybrid) search; auto-computed from chunk_text
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- filter-by-tender then vector search
CREATE INDEX IF NOT EXISTS chunks_tender_pk_idx ON chunks(tender_pk);
-- approximate nearest neighbour, cosine distance
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
-- keyword search (hybrid retrieval)
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
