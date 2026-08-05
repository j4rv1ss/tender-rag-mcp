# Tender RAG — a chatbot that answers questions about tenders

This is a small AI app. You ask a question about a government **tender** (a public
contract offer), and it gives you an answer **based on that tender's real
documents** — with references to the exact document and page it came from.

It is built on top of the tender **scrapers** in `c:\anshul\MVP` (separate
programs that download tenders and their files from 9 government websites).

Example:

> **You:** What is the closing date for tender RW10408567/26?
> **App:** The closing date of this tender is **28 August 2026 at 12:00 pm**.
> *(from the tender's cover page, page 1)*

---

## 1. What problem does it solve?

A tender comes with long PDF/Word documents (sometimes 100+ pages). Reading them
to find one fact — the closing date, who to contact, what documents you must
submit — is slow. This app reads the documents for you and answers questions in
about **half a second**, and it **shows its sources** so you can trust the answer.

Very important: it only answers from the **actual documents**. If the answer is
not in the documents, it says *"not available"* instead of making something up.

It also understands you even when you use **different words** — "ending date",
"deadline" and "closing date" all mean the same thing — and it writes the answer
as a **clean sentence with nicely formatted dates and amounts**, not raw data.

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
  and writes a human answer. Here we use **Groq** (a fast, free cloud AI service).

Putting it together — this pattern is called **RAG** (Retrieval-Augmented
Generation): *Retrieve* the relevant text, then *Generate* an answer from it.

```
Your question
   │
   ▼
turn question into numbers (embedding)        ← in-process, free (fastembed)
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
| **fastembed** | a small AI model that runs **inside the app** (free, offline) | makes the "meaning numbers" (embeddings) — no separate service, no API, no limits |
| **Groq** | a **cloud** AI service (free tier) | writes the actual answers, very fast |
| **FastAPI** | a web-server toolkit | provides the chat page and the API |
| **SQLAlchemy** | talks to the database from Python | reads/writes tenders, documents, chunks |

**Why is the "meaning numbers" part built into the app?**
Making embeddings in-process (with a small model called `bge-small`, 384 numbers
each) means there is **no extra service to run, no API key, and no daily limit** —
it just works, on your PC and in the cloud. The documents never leave the app.

**Why Groq for the answers?** Groq is fast and free. If Groq is ever busy or its
free limit is reached, the app automatically tries a **smaller Groq model**, and
as a last resort a **local model** (Ollama, if you have it) — so you always get an
answer. (Ollama is optional — only needed if you want a fully offline fallback.)

**Why not just ask ChatGPT/an LLM directly?** Because a plain LLM doesn't know
your specific tender's documents, and it can make things up. RAG forces the
answer to come from the real documents and cite them.

---

## 4. Key words cheat-sheet

- **Tender** — a public contract a government advertises. Also called a "bid".
- **Document** — a file attached to a tender (PDF, Word, Excel).
- **Chunk** — a small piece of a document (about a paragraph). We split documents
  into chunks so we can find the *exact* relevant part, not the whole 100-page file.
- **Embedding / vector** — the list of 384 numbers representing a chunk's meaning.
- **Ingest** — the process of reading a tender + its documents into the database
  (and making the embeddings). "Ingested" = already loaded and ready to answer.
- **RAG** — Retrieval-Augmented Generation (the retrieve-then-answer method above).

---

## 5. What you need installed (already done on this PC)

- **PostgreSQL 18** running, with the **pgvector** add-on installed.
- **Python 3.12**, with this project's packages installed in `tender_rag/.venv`
  (the embedding model, **fastembed**, is one of those packages — nothing extra to
  install or run).
- A free **Groq API key** (in the `.env` file) for fast answers.
- **Ollama is optional** — only if you want an offline chat fallback when there's
  no internet. Embeddings do **not** need it.

If you ever set this up from zero, see **Section 10 (Full setup)** below.

---

## 6. How the data flows

### 6a. Ingesting a tender (loading it in)
1. A scraper has already produced a file like
   `randwater/output/tender_RW10408567_26.json` (tender info + the documents'
   text, page by page).
2. A **mapper** converts that file into one standard format (each of the 9
   websites uses a different layout, so there's one mapper per website).
3. The tender info goes into the **`tenders`** table; each document's text goes
   into **`documents`**.
4. Each document is split into **chunks**; each chunk is turned into an
   **embedding** (by the built-in fastembed model) and saved in **`chunks`**
   (pgvector).

### 6b. Answering a question
1. Your question is turned into an embedding (in-process).
2. pgvector finds the **top 8 most-similar chunks**.
3. Those chunks + the tender's basic info + your question are sent to **Groq**.
4. Groq writes a short, well-formatted answer and cites the document + page.

---

## 7. How to use it (the fun part)

The app runs at **http://localhost:8000**.

### Easiest: the chat page
1. Open **http://localhost:8000/** in your browser.
2. **Ask about ONE tender:** fill **Source** and **Tender ID**, type a question,
   click **Ask**.
   - Source: `ppadb` · Tender ID: `PR/PPADB/055`
   - Question: *"What is this tender for and what is the closing date?"*
   - ⚠️ Copy the Tender ID **exactly**, including any `/` slashes and spaces.
3. **Ask about ALL loaded tenders at once:** clear the **Tender ID** box, type a
   question, click **Ask**.
   - *"Which tenders involve construction work?"*

Answers come back in about **half a second**.

### Which tenders are loaded?
Open **http://localhost:8000/tenders** (or the API) to see the current list — it
changes as you ingest or clear tenders. The scrapers cover 9 portals: `etenders`,
`transnet`, `sadc`, `zppa`, `ppadb`, `randwater`, `capetown`, `cpbn`, `nra`.

### Asking about a tender that isn't loaded yet (local only)
On your PC, you don't need to load it first — **just ask** with its source + ID and
the app will **run the scraper, load it, then answer** (the first time is slower
because it has to download it). Works for 8 of the 9 sites; **nra** needs a human
check, so load that one by hand.
*(In the cloud this auto-scrape is turned off — the cloud app only answers tenders
that were already loaded. See DEPLOY.md.)*

### Good questions to try
- *"What is the published date?"* vs *"What is the closing date?"* (different dates)
- *"What is the deadline?"* (a synonym — it still finds the closing date)
- *"Are alternative bids allowed?"* (an opposite/yes-no question)
- *"Who is the contact person?"* · *"What documents must I submit?"*
- *"List all the construction tenders."* (ask across all loaded tenders)

---

## 8. The API (for developers / Swagger)

Interactive docs: **http://localhost:8000/docs** (try any endpoint in the browser).

| Method | Address | What it does |
|---|---|---|
| GET  | `/health` | check everything is working (database, pgvector, embeddings, Groq) |
| POST | `/chat` | ask a question (one tender, or all tenders) |
| POST | `/fetch` | load a specific tender now (scrape + ingest) — local only |
| POST | `/ingest` | load a tender that's already scraped to disk |
| POST | `/ingest-all` | load every tender that's been scraped |
| GET  | `/tenders` | list all loaded tenders |
| GET  | `/tenders/{id}?source=` | details of one tender |
| GET  | `/` | the chat page |

**Ask one tender:**
```json
POST /chat
{ "source": "ppadb", "tender_id": "PR/PPADB/055",
  "question": "What is the closing date?" }
```
**Ask all loaded tenders** (leave out source/tender_id):
```json
POST /chat
{ "question": "Which tenders involve construction?" }
```
**Answer looks like:**
```json
{ "mode": "tender",
  "answer": "The closing date is 28 August 2026 at 12:00 pm ...",
  "references": [ { "document": "cover page.pdf", "page": 1, "score": 0.71 } ],
  "chunks_used": 8 }
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
      ingest_service.py loads tenders; also auto-fetches missing ones (local)
      scrape.py        runs a website's scraper on demand (local)
      chunking.py      splits documents into chunks
      embeddings.py    makes embeddings in-process (fastembed); Ollama/Gemini optional
      llm.py           gets answers from Groq (3-tier: 70b -> 8b -> local Ollama)
      retriever.py     the pgvector "find similar chunks" search
      prompts.py       the instructions given to the AI (synonyms, formatting, citing)
      rag.py           ties retrieve + answer together
    static/index.html  the chat web page
  db/init.sql          creates the database tables (embedding VECTOR(384))
  scripts/             setup + loading helper scripts (load_data.py for the cloud)
  docs/architecture.md pictures/diagrams of the system
  Dockerfile           builds the app image for free cloud hosting
  render.yaml          one-click deploy config for Render
  DEPLOY.md            step-by-step free deploy (Render + Neon)
  .env                 your settings + secret keys (keep private!)
  requirements.txt     the Python packages needed
```

---

## 10. Full setup (from a blank machine)

Only needed if setting up somewhere new — this PC is already done.

1. **Install PostgreSQL 18.**
2. **Add pgvector** (the vector add-on):
   `powershell -ExecutionPolicy Bypass -File scripts\install_pgvector.ps1`
3. **Create the Python environment** (this also installs the embedding model):
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
4. **Settings:** copy `.env.example` to `.env`. Fill in:
   - `POSTGRES_PASSWORD` — your database password.
   - `GROQ_API_KEY` — a free key from https://console.groq.com (for fast answers).
   - `EMBED_PROVIDER=fastembed` (the default in-process embeddings).
5. **Create the database + tables:**
   `powershell -ExecutionPolicy Bypass -File scripts\setup_db.ps1`
6. **Load some tenders:** `.venv\Scripts\python scripts\ingest_all.py`

*(Optional: install Ollama only if you want an offline chat fallback —
`winget install Ollama.Ollama` then `ollama pull llama3.2:3b`.)*

### Put it online for FREE
To host the chatbot on the internet at no cost (Render + Neon + Groq), follow the
step-by-step guide in **[DEPLOY.md](DEPLOY.md)**.

---

## 11. Start / stop the app

**Start:**
```
cd c:\anshul\MVP\tender_rag
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Then open http://localhost:8000/. Leave that window open while you use it.

**Stop:** press `Ctrl + C` in that window.

The first question after starting takes a few seconds (the embedding model loads
once), then every question is fast.

---

## 12. Common questions

- **Do I need internet?** Only for Groq (the answer-writer). The embeddings and the
  database run on your machine. Remove `GROQ_API_KEY` (and install Ollama) to run
  fully offline with a local answer model (slower).
- **Is it free?** Yes. PostgreSQL and the embeddings are free; Groq has a free tier;
  and the whole thing can be hosted free (see DEPLOY.md).
- **Does it use my ChatGPT/Claude account?** No. It uses Groq for answers and a small
  built-in model for embeddings — nothing else.
- **The first question after starting is slow (~a few seconds).** That's the embedding
  model loading once. After that, questions take about **0.5 seconds**.
- **Asking about a brand-new tender is slow (local).** The first time, it downloads
  it (scrape). After that it's instant. Large sites can take a few minutes.
- **It said "not available for this tender".** That's correct behaviour — the fact
  isn't in that tender's documents, so it won't guess.
- **A Tender ID didn't work.** Copy it **exactly** — many IDs contain `/` slashes
  and spaces (e.g. `RW10414743/25 R`). An underscore instead of a slash won't match.
- **Where are my secrets?** In `tender_rag/.env` (database password + Groq key).
  Keep this file private — don't put it in a shared repository.

---

## 13. Limits (honest)

- It can only answer about a tender it has **loaded**. It won't invent facts about
  tenders it has never seen.
- **Answer quality depends on the scraped text.** If a tender's scraper only captured
  a cover page, deep questions will correctly say "not available" — that's missing
  data, not the AI failing.
- A tender must still **exist on the website** to be scraped fresh; a closed/removed
  one can't be fetched — you'll get a clear message, not a guess.
- **nra** can't be auto-downloaded (human-verification check); load it manually.
- The **cloud version is query-only** — it answers loaded tenders but doesn't scrape.

---

For the technical diagrams (system, database, and step-by-step flows), see
[docs/architecture.md](docs/architecture.md).
