# Tender RAG — How it's built (with pictures)

This explains **how the app works inside**, in simple words, with diagrams.
If you just want to use it, read the main [README](../README.md) first.

Quick reminder of the idea: you ask a question about a tender, and the app
answers using that tender's real documents. It does this by (1) turning text
into "meaning numbers", (2) finding the closest ones to your question, and
(3) letting an AI write the answer from them. That method is called **RAG**
(Retrieve, then Generate).

> **How to read the diagrams:** each box is a part of the system; each arrow
> means "sends data to". A plain-English explanation comes right after every
> diagram.

---

## 1. The big picture

```mermaid
flowchart LR
  subgraph SC[Scrapers - separate programs]
    OUT[Tender files on disk<br/>tender_*.json + document text]
  end
  subgraph APP[Tender RAG app - FastAPI]
    API[Web API + Chat page]
    SCR[Scrape service<br/>fetch a missing tender]
    ING[Loader<br/>map, split, embed]
    RAGS[Answer engine]
  end
  DB[(PostgreSQL<br/>+ pgvector)]
  OLL[Ollama - on your PC<br/>makes embeddings]
  GROQ[Groq - cloud<br/>writes the answer]

  OUT --> ING
  API --> SCR --> ING
  ING --> DB
  ING --> OLL
  API --> RAGS
  RAGS --> DB
  RAGS --> OLL
  RAGS --> GROQ
```

**In words:**
- The **scrapers** (separate programs) download tenders from 9 websites and save
  them as files.
- The **Loader** reads those files, splits the documents into small pieces, asks
  **Ollama** (on your PC) to turn each piece into meaning-numbers, and stores
  everything in **PostgreSQL + pgvector**.
- When you ask a question, the **Answer engine** finds the closest pieces in the
  database and asks **Groq** (fast cloud AI) to write the answer.
- If you ask about a tender that isn't loaded, the **Scrape service** runs the
  right website's scraper first, then loads it.

Two AIs, two jobs: **Ollama** makes the numbers (free, on your PC); **Groq**
writes the answers (fast, in the cloud).

---

## 2. What's stored in the database (3 tables)

```mermaid
erDiagram
  TENDERS ||--o{ DOCUMENTS : has
  DOCUMENTS ||--o{ CHUNKS : has
  TENDERS ||--o{ CHUNKS : has
  TENDERS {
    bigint id PK
    text source
    text tender_id
    text title
    text organization
    text country
    text issue_date
    text closing_date
    jsonb raw_json
  }
  DOCUMENTS {
    bigint id PK
    bigint tender_pk FK
    text file_name
    text extracted_text
    int page_count
  }
  CHUNKS {
    bigint id PK
    bigint tender_pk FK
    bigint document_id FK
    text chunk_text
    int page_number
    vector embedding "768 numbers"
  }
```
*(Some columns are left out above to keep it readable — see `db/init.sql` for all.)*

**In words — three tables, linked like folders inside folders:**
- **`tenders`** — one row per tender. The basic facts: title, organization,
  country, published date, closing date, etc. It also keeps the **whole original
  file** in `raw_json`, so nothing is ever lost.
- **`documents`** — one row per file attached to a tender, with its full text.
- **`chunks`** — the important one. Each document is cut into small pieces
  ("chunks"). Each chunk stores its text, which page it came from, and its
  **embedding** (the list of 768 numbers that represents its meaning).

The `chunks.embedding` column is the special **pgvector** type. A **HNSW index**
on it makes "find the closest chunks" very fast, even with thousands of chunks.

**The dates are stored as plain text on purpose** — the 9 websites write dates in
6 different formats (e.g. `2026-08-05`, `5th August 2026`, `13 Aug 2026`), so we
keep them exactly as written instead of forcing one format.

---

## 3. Loading a tender (step by step)

```mermaid
sequenceDiagram
  actor U as Caller
  participant API as /ingest
  participant N as Mapper
  participant DB as PostgreSQL
  participant CH as Splitter
  participant E as Ollama (embed)
  U->>API: source + tender_id
  API->>N: read the tender file
  N-->>API: standard tender + its documents
  API->>DB: save the tender + documents
  loop for each document
    API->>CH: cut into ~500-word chunks (remember the page)
    CH-->>API: list of chunks
    API->>E: turn each chunk into numbers
    E-->>API: the embeddings
    API->>DB: save the chunks + embeddings
  end
  API-->>U: loaded: X documents, Y chunks
```

**In words:** read the tender file → convert it to a standard shape (the
"Mapper") → save the tender and its documents → cut each document into chunks →
turn each chunk into numbers with Ollama → save the chunks. After this, the
tender is ready to answer questions.

This is **safe to repeat** — loading the same tender again just replaces its old
data, so you never get duplicates.

---

## 4. Answering a question (step by step)

```mermaid
sequenceDiagram
  actor U as You
  participant API as /chat
  participant F as "ensure it's loaded"
  participant E as Ollama (embed)
  participant PG as pgvector
  participant G as Groq (answer)
  U->>API: source + tender_id + question
  API->>F: is this tender loaded?
  alt not loaded yet
    F->>F: scrape it, then load it
  end
  API->>E: turn the question into numbers
  E-->>API: question vector
  API->>PG: find the closest chunks
  PG-->>API: top chunks (+ document + page)
  API->>G: tender facts + chunks + question
  G-->>API: written answer
  API-->>U: answer + references (document, page)
```

**In words:**
1. Check the tender is loaded (if not, fetch it first — see section 5).
2. Turn your question into numbers (Ollama).
3. Ask pgvector for the few chunks closest in meaning.
4. Send the tender's basic facts + those chunks + your question to Groq.
5. Groq writes a short answer that only uses what it was given, and cites the
   document + page. You get it back.

---

## 5. Asking about a tender that isn't loaded (auto-fetch)

You don't have to load a tender before asking. The app figures it out:

```mermaid
flowchart TD
  Q[You ask about a tender] --> A{In the database already?}
  A -- yes --> ANS[Answer it]
  A -- no --> B{Already downloaded to disk?}
  B -- yes --> L[Load it] --> ANS
  B -- no --> S[Run that website's scraper] --> L
```

**In words:** first it checks the database; if not there, it checks whether the
tender was already downloaded; if not, it runs the right website's scraper to
download it, then loads it, then answers. The first time is slower (it has to
download); after that it's instant. Works for 8 of the 9 sites automatically —
**nra** needs a human-verification check, so that one is loaded by hand.

---

## 6. The 9 "Mappers" (why they're needed)

Each of the 9 websites saves tenders in a **different layout** — different field
names, and different ways of listing documents (some call them `documents`, one
calls them `attachments`, two have a single `document`). A RAG system needs one
consistent shape, so `services/normalize.py` has **one small mapper per website**
that translates its layout into the standard `tenders` + `documents` shape.

Adding a 10th website later = write one more mapper. Everything else (splitting,
embedding, searching, answering) stays the same.

---

## 7. Two ways to ask

- **About one tender** — give a `source` + `tender_id`. It only looks inside that
  tender. (e.g. "What is the closing date of tender 162660?")
- **About all tenders at once** — leave out the tender id. It searches **every**
  tender and tells you which tender each fact came from. (e.g. "Which tenders are
  for construction?") Sources are cited as `[tender | document p.page]`.

---

## 8. How documents are split (chunking)

The scrapers already give us each document **page by page**. We cut each page
into chunks of about **500 words** (with a little overlap so sentences aren't cut
awkwardly). Because we split **inside each page**, every chunk remembers its real
**page number** — that's how the answers can cite the exact page.

Tiny leftover pieces (like a lone page number) are dropped, since there's nothing
useful to search in them.

---

## 9. How the AI is told to answer (the prompt)

Before Groq writes an answer, it's given strict instructions:
- Use **only** the tender facts and chunks provided — no outside knowledge.
- **Cite** the document and page for each fact.
- If the answer isn't in the provided text, **say "not available"** — don't guess.
- The published date and the closing date are **different** — never swap them.

This is what keeps answers trustworthy and stops the AI from making things up.

---

## 10. Finding the closest chunks (retrieval)

pgvector compares your question's numbers to every chunk's numbers using **cosine
distance** (a standard "how similar in meaning" measure) and returns the closest
few (default 4). The **HNSW index** makes this fast. This search code is kept
separate, so the storage could be swapped later without touching the rest.

---

## 11. Why it's fast (~0.5 seconds)

Most of the speed came from one fix: **reusing the connection to Ollama**.
Opening a brand-new connection for every question made Ollama's first request
take ~2.5 seconds each time. Now the app keeps **one open connection** (to both
Ollama and Groq) and holds it for 5 minutes, so questions after the first take
about **0.1s to make the numbers + ~0.4s for Groq ≈ half a second total**. The
app also "warms up" both connections when it starts.

---

## 12. When something goes wrong (error handling)

The app fails **safely and clearly**, never with a made-up answer:
- Tender not found and can't be fetched → a clear message telling you what to do.
- The tender exists nowhere / has closed → "it may have closed", not a guess.
- AI service down → a clear error (including how to fix it), not a wrong answer.
- No relevant chunks found → "I couldn't find relevant content", not invention.
- `/health` shows the status of the database, pgvector, Ollama, and Groq at a
  glance.

---

## 13. Taking it to production (later)

This is a proof-of-concept running on one PC. To run it for many users:

```mermaid
flowchart TB
  C[Many users] --> LB[Gateway<br/>HTTPS + rate limits]
  LB --> APP[Several app copies]
  APP --> PGM[(Managed PostgreSQL<br/>+ pgvector)]
  APP --> RQ[[Background loaders<br/>do the heavy embedding]]
  APP --> CACHE[(Redis<br/>remember recent answers)]
  APP --> LLMP[AI: Groq or a GPU box]
```

**Main upgrades for production:**
- **Load tenders in the background** (a queue + workers) so the website stays snappy.
- **Cache** common questions and answers (Redis) to avoid repeating work.
- **Managed database** with pgvector, and index tuning for large data.
- **Scale the AI**: keep Groq, or run bigger models on a GPU machine.
- **Operations**: proper logging, monitoring, rate limits, and keeping secrets in
  a secure vault instead of a file.

Because the AI and embeddings live behind small, separate files
(`services/llm.py`, `services/embeddings.py`), swapping providers is a settings
change — not a rewrite.

---

## 14. Setting it up from scratch

1. **Install PostgreSQL 18.**
2. **Add pgvector:** `scripts/install_pgvector.ps1` (builds the add-on and copies
   it into PostgreSQL — needs admin once).
3. **Install Ollama** and download the models:
   `ollama pull nomic-embed-text` and `ollama pull llama3.2:3b`.
4. **Python:** `py -3.12 -m venv .venv` then `pip install -r requirements.txt`.
5. **Settings:** copy `.env.example` to `.env`; set `POSTGRES_PASSWORD` and a free
   `GROQ_API_KEY` (from https://console.groq.com).
6. **Create the tables:** `scripts/setup_db.ps1`.
7. **Load tenders:** `scripts/ingest_all.py`, then run
   `uvicorn app.main:app` and open http://localhost:8000/.
