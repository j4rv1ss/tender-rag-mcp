# Tender RAG — a chatbot that answers questions about tenders

This is a small AI app. You ask a question about a government **tender** (a public
contract offer), and it gives you an answer **based on that tender's real
documents** — with references to the exact document and page it came from.

It is built on top of the tender **scrapers** in `c:\anshul\MVP` (separate
programs that download tenders and their files from 9 government websites).

Example:

> **You:** What is the closing date for tender 162660?
> **App:** The closing date is 2026-08-05 11:00. *(from the tender's advert, page 1)*

---

## 1. What problem does it solve?

A tender comes with long PDF/Word documents (sometimes 100+ pages). Reading them
to find one fact — the closing date, who to contact, what documents you must
submit — is slow. This app reads the documents for you and answers questions in
about **half a second**, and it **shows its sources** so you can trust the answer.

Very important: it only answers from the **actual documents**. If the answer is
not in the documents, it says *"not available"* instead of making something up.

---

## 2. How it works (in simple words)

Think of it like a very fast librarian:

1. **You give it tenders.** It reads each tender's documents and remembers them.
2. **You ask a question.** It finds the few paragraphs most related to your
   question, reads them, and writes a short answer with references.

To do this it uses three ideas. Here they are in plain English:

- **Embedding** = turning a piece of text into a list of numbers that captures
  its *meaning*. Two texts about the same thing get similar numbers. (Think of it
  as giving every paragraph a "GPS coordinate" based on meaning.)
- **Vector database (pgvector)** = a database that can store those number-lists
  and quickly find the ones **closest in meaning** to your question. (Like
  finding the nearest coffee shops to your location — but for meaning.)
- **LLM (Large Language Model)** = the "brain" that reads the found paragraphs
  and writes a human answer. Here we use **Groq** (a fast cloud AI service).

Putting it together — this pattern is called **RAG** (Retrieval-Augmented
Generation): *Retrieve* the relevant text, then *Generate* an answer from it.

```
Your question
   │
   ▼
turn question into numbers (embedding)        ← local, free (Ollama)
   │
   ▼
find the closest paragraphs (pgvector search) ← PostgreSQL
   │
   ▼
give those paragraphs + your question to the AI (Groq)
   │
   ▼
Answer + references (which document, which page)
```

---

## 3. The parts of the system (and why)

| Part | What it is | What it does here |
|---|---|---|
| **PostgreSQL** | a normal database | stores the tender info and the document text |
| **pgvector** | an add-on for PostgreSQL | stores the "meaning numbers" and finds similar ones |
| **Ollama** | runs AI models **on your computer** (free, offline) | makes the "meaning numbers" (embeddings) |
| **Groq** | a **cloud** AI service (free tier) | writes the actual answers, very fast |
| **FastAPI** | a web-server toolkit | provides the chat page and the API |
| **SQLAlchemy** | talks to the database from Python | reads/writes tenders, documents, chunks |

**Why two AIs (Ollama *and* Groq)?**
- **Ollama** (on your PC) does the embeddings — free, private, and the documents
  never leave your machine.
- **Groq** (cloud) writes the answers — it's much faster and smarter than a small
  local model. It needs a free key. If you remove the key, the app automatically
  falls back to a local model (slower, but fully offline).

**Why not just ask ChatGPT/an LLM directly?** Because a plain LLM doesn't know
your specific tender's documents, and it can make things up. RAG forces the
answer to come from the real documents and cite them.

---

## 4. Key words cheat-sheet

- **Tender** — a public contract a government advertises. Also called a "bid".
- **Document** — a file attached to a tender (PDF, Word, Excel).
- **Chunk** — a small piece of a document (about a paragraph). We split documents
  into chunks so we can find the *exact* relevant part, not the whole 100-page file.
- **Embedding / vector** — the list of numbers representing a chunk's meaning.
- **Ingest** — the process of reading a tender + its documents into the database
  (and making the embeddings). "Ingested" = already loaded and ready to answer.
- **RAG** — Retrieval-Augmented Generation (the retrieve-then-answer method above).

---

## 5. What you need installed (already done on this PC)

- **PostgreSQL 18** running, with the **pgvector** add-on installed.
- **Ollama** installed, with two models downloaded: `nomic-embed-text` (for
  embeddings) and `llama3.2:3b` (a local backup chat model).
- **Python 3.12**, with this project's packages installed in `tender_rag/.venv`.
- A free **Groq API key** (in the `.env` file) for fast answers.

If you ever set this up from zero, see **Section 10 (Full setup)** below.

---

## 6. How the data flows

### 6a. Ingesting a tender (loading it in)
1. A scraper has already produced a file like
   `etenders/output/tender_162660.json` (tender info + the documents' text,
   page by page).
2. A **mapper** converts that file into one standard format (each of the 9
   websites uses a different layout, so there's one mapper per website).
3. The tender info goes into the **`tenders`** table; each document's text goes
   into **`documents`**.
4. Each document is split into **chunks**; each chunk is turned into an
   **embedding** (by Ollama) and saved in **`chunks`** (pgvector).

### 6b. Answering a question
1. Your question is turned into an embedding.
2. pgvector finds the **top few most-similar chunks**.
3. Those chunks + the tender's basic info + your question are sent to **Groq**.
4. Groq writes a short answer and cites the document + page. You get it back.

---

## 7. How to use it (the fun part)

The app runs at **http://localhost:8000**.

### Easiest: the chat page
1. Open **http://localhost:8000/** in your browser.
2. **Ask about ONE tender:** fill **Source** and **Tender ID**, type a question,
   click **Ask**.
   - Source: `cpbn` · Tender ID: `W/ONB/CPBN-08/2026`
   - Question: *"What is this tender for and what is the closing date?"*
3. **Ask about ALL tenders at once:** clear the **Tender ID** box, type a
   question, click **Ask**.
   - *"Which tenders involve construction work?"*

Answers come back in about **half a second**.

### The tenders currently loaded (15)
`etenders 162660` · `transnet 114472` · `zppa 28231539` · `ppadb PR/PPADB/055` ·
`randwater RW10414743/25 R` and `RW10408567/26` · `capetown 289S/2025/26` ·
`nra x-002-227-2025-1f-non-toll-x-002-228-2025-1f-toll` ·
`cpbn W/ONB/CPBN-08/2026` · `sadc 6335 / 6562 / 6582 / 6583 / 6594 / 6615`

### Asking about a tender that isn't loaded yet
You don't need to load it first — **just ask** with its source + ID. If the app
doesn't have it, it will **automatically run the scraper, load it, then answer**
(this first time is slower because it has to download it). After that it's instant.

> Works for 8 of the 9 sites automatically. **nra** can't be auto-loaded (its
> site uses a "prove you're human" check), so you'd load that one by hand.

### Good questions to try
- *"What is the published date?"* vs *"What is the closing date?"* (two different dates)
- *"Who is the contact person?"*
- *"What documents must I submit?"*
- *"What are the eligibility requirements?"*
- *"List all the security-services tenders."* (ask across all tenders)

---

## 8. The API (for developers / Swagger)

Interactive docs: **http://localhost:8000/docs** (try any endpoint in the browser).

| Method | Address | What it does |
|---|---|---|
| GET  | `/health` | check everything is working (database, pgvector, Ollama, Groq) |
| POST | `/chat` | ask a question (one tender, or all tenders) |
| POST | `/fetch` | load a specific tender now (scrape + ingest) |
| POST | `/ingest` | load a tender that's already scraped to disk |
| POST | `/ingest-all` | load every tender that's been scraped |
| GET  | `/tenders` | list all loaded tenders |
| GET  | `/tenders/{id}?source=` | details of one tender |
| GET  | `/` | the chat page |

**Ask one tender:**
```json
POST /chat
{ "source": "cpbn", "tender_id": "W/ONB/CPBN-08/2026",
  "question": "What is the closing date?" }
```
**Ask all tenders** (leave out source/tender_id):
```json
POST /chat
{ "question": "Which tenders involve construction?" }
```
**Answer looks like:**
```json
{ "mode": "tender",
  "answer": "The closing date is 5th August 2026 11:00 ...",
  "references": [ { "document": "ACC_ Bidding Document...pdf", "page": 178, "score": 0.71 } ],
  "chunks_used": 4 }
```

---

## 9. Folder structure (what's where)

```
tender_rag/
  app/
    main.py            the web app (starts everything)
    config.py          settings (read from the .env file)
    db.py              connects to PostgreSQL
    models.py          the 3 tables: Tender, Document, Chunk
    schemas.py         shapes of the requests/replies
    routers/           the API endpoints (health, chat, ingest, tenders)
    services/
      normalize.py     the 9 mappers (one per website)
      ingest_service.py loads tenders; also auto-fetches missing ones
      scrape.py        runs a website's scraper on demand
      chunking.py      splits documents into chunks
      embeddings.py    makes embeddings via Ollama
      llm.py           gets answers from Groq (or local Ollama)
      retriever.py     the pgvector "find similar chunks" search
      prompts.py       the instructions given to the AI
      rag.py           ties retrieve + answer together
    static/index.html  the chat web page
  db/init.sql          creates the database tables
  scripts/             setup + loading helper scripts
  docs/architecture.md pictures/diagrams of the system
  .env                 your settings + secret keys (keep private!)
  requirements.txt     the Python packages needed
```

---

## 10. Full setup (from a blank machine)

Only needed if setting up somewhere new — this PC is already done.

1. **Install PostgreSQL 18.**
2. **Add pgvector** (the vector add-on):
   `powershell -ExecutionPolicy Bypass -File scripts\install_pgvector.ps1`
3. **Install Ollama** and download the models:
   ```
   winget install Ollama.Ollama
   ollama pull nomic-embed-text
   ollama pull llama3.2:3b
   ```
4. **Create the Python environment:**
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
5. **Settings:** copy `.env.example` to `.env`. Fill in:
   - `POSTGRES_PASSWORD` — your database password.
   - `GROQ_API_KEY` — a free key from https://console.groq.com (for fast answers).
6. **Create the database + tables:**
   `powershell -ExecutionPolicy Bypass -File scripts\setup_db.ps1`
7. **Load some tenders:** `.venv\Scripts\python scripts\ingest_all.py`

---

## 11. Start / stop the app

**Start:**
```
cd c:\anshul\MVP\tender_rag
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Then open http://localhost:8000/. Leave that window open while you use it.

**Stop:** press `Ctrl + C` in that window.

If chat is ever unreachable, also make sure Ollama is running (`ollama serve`;
it usually starts on its own).

---

## 12. Common questions

- **Do I need internet?** Yes, for Groq (the answer-writer). Embeddings and the
  database are local. Remove `GROQ_API_KEY` to run fully offline on the local
  model (slower answers).
- **Is it free?** Yes. Ollama and PostgreSQL are free; Groq has a free tier.
- **Does it use my ChatGPT/Claude account?** No. It uses Groq for answers and
  Ollama on your PC — nothing else.
- **The first question after starting is slow (~2–3s).** That's a one-time warm
  up. After that, questions take about **0.5 seconds**.
- **Asking about a brand-new tender is slow.** The first time, it has to download
  it (scrape). After that it's instant. Very large sites (etenders) can take a
  few minutes the first time.
- **It said "not available for this tender".** That's correct behaviour — the
  fact isn't in that tender's documents, so it won't guess.
- **Where are my secrets?** In `tender_rag/.env` (database password + Groq key).
  Keep this file private — don't put it in a shared repository.

---

## 13. Limits (honest)

- It can only answer about a tender it has **downloaded**. It won't invent facts
  about tenders it has never seen.
- A tender must still **exist on the website**. If a tender has closed and been
  removed, the scraper can't fetch it — you'll get a clear message, not a guess.
- **nra** can't be auto-downloaded (human-verification check); load it manually.
- Answer quality depends on the documents. If a document is a scanned image with
  poor text, the extracted text (and answers) may be weaker.

---

For the technical diagrams (system, database, and step-by-step flows), see
[docs/architecture.md](docs/architecture.md).
