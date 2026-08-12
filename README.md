# Tender RAG — an MCP server that answers questions about tenders

This is a small AI app, exposed as an **MCP server** so any AI assistant (Claude
Desktop, Claude Code, …) can use it as a tool. You ask a question about a
government **tender** (a public contract offer), and it gives you an answer
**based on that tender's real documents** — with references to the exact document
and page it came from.

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
a **few seconds**, and it **shows its sources** so you can trust the answer.

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
  and writes a human answer. Here we use **Llama 4** (Meta's latest open model),
  reached over the cloud through **LangChain**.

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
give those paragraphs + your question to the AI (Llama 4, via LangChain)
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
| **Llama 4** | Meta's latest LLM, reached over the cloud (OpenRouter, free key) | writes the actual answers |
| **LangChain** | a library for talking to LLMs | drives the Llama 4 call — and any OpenAI-compatible model — with automatic retries + provider fallback |
| **MCP** | Model Context Protocol — the standard way an AI assistant plugs into an external tool | exposes the whole system as **tools** an assistant (Claude Desktop, Claude Code, …) can call directly |
| **SQLAlchemy** | talks to the database from Python | reads/writes tenders, documents, chunks |

**Why MCP instead of a website?** Previously this ran as a web app you opened in a
browser. Now it is primarily an **MCP server**: instead of you typing into a chat
page, your AI assistant calls the tender tools itself. That means you can ask *"is
the Rand Water cathodic-protection tender worth bidding for?"* in a normal
conversation, and the assistant will look it up in this system, read the real
documents, and answer — mixing it freely with everything else it can do.

When hosted over HTTP the server also serves a **plain chat page at `/`**, for
demos and for anyone without an MCP client. It calls the same RAG pipeline through
`POST /api/chat`; the assistant is the richer interface, the page is the zero-install
one.

That REST side is a **FastAPI app mounted at `/api`**, so it comes with request
validation and interactive docs at **`/api/docs`**:

| Method | Path | What it does |
|---|---|---|
| POST | `/api/chat` | ask about one tender, or all of them if `tender_id` is blank |
| POST | `/api/summary` | grounded brief of one tender |
| POST | `/api/ingest` | index a tender (scrapes first if enabled) |
| GET | `/api/tenders` | what is loaded |
| GET | `/api/health` | same probe as the `health_check` tool |

One process serves all of it — page at `/`, REST at `/api/*`, MCP at `/mcp` —
over the same [app/services/](app/services/), so the two front doors cannot
answer differently.

**Why is the "meaning numbers" part built into the app?**
Making embeddings in-process (with a small model called `bge-small`, 384 numbers
each) means there is **no extra service to run, no API key, and no daily limit** —
it just works, on your PC and in the cloud. The documents never leave the app.

**Why Llama 4, and why via LangChain?** Llama 4 is a strong, current open model,
and LangChain lets the app talk to it — or any OpenAI-compatible model — through
one tidy interface. The app tries models **in order**: **Llama 4 Maverick → Llama 4
Scout** (both on OpenRouter), then, if those are rate-limited, a **backup model on
Groq** (OpenAI `gpt-oss`), and only as a last resort a **local model** (Ollama, if
you configure one) — so you always get an answer. Switching provider is just a new
URL + key + model name in `.env`; no code changes.

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
- A free **OpenRouter API key** (in the `.env` file) so the app can reach **Llama 4**.
  An optional **Groq** key adds a backup model.
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
3. Those chunks + the tender's basic info + your question are sent to **Llama 4**
   (through LangChain).
4. Llama 4 writes a short, well-formatted answer and cites the document + page.

---

## 7. How to use it (the fun part)

You don't start a website. You **connect the server to an AI assistant once**, and
from then on you just talk to the assistant normally.

### Connect it (one time)

**Claude Code** — from this folder:
```
claude mcp add tender-rag -- "c:\anshul\MVP - Copy\tender_rag\.venv\Scripts\python.exe" -m app.mcp_server
```

**Claude Desktop** — edit
`%APPDATA%\Claude\claude_desktop_config.json` and add:
```json
{
  "mcpServers": {
    "tender-rag": {
      "command": "c:\\anshul\\MVP - Copy\\tender_rag\\.venv\\Scripts\\python.exe",
      "args": ["-m", "app.mcp_server"],
      "cwd": "c:\\anshul\\MVP - Copy\\tender_rag"
    }
  }
}
```
Then restart Claude Desktop. A tools icon appears — the tender tools are in it.

> The `cwd` matters: the server reads `.env` from the project folder.

### Then just ask
- *"What is this tender for and when does it close?"* → the assistant calls
  `ask_tender` (give it the source + tender ID, e.g. `ppadb` / `PR/PPADB/055`).
- *"Which tenders involve construction work?"* → `ask_all_tenders`, across everything loaded.
- *"Give me a full brief on the Rand Water cathodic protection tender."* → `summarize_tender`.
- *"What tenders do we have?"* → `list_tenders`.

⚠️ Copy a Tender ID **exactly**, including any `/` slashes and spaces.

Answers come back in a **few seconds**, and always cite the document + page.

### Which tenders are loaded?
Ask the assistant *"list the loaded tenders"* (the `list_tenders` tool). It takes an
optional search filter, e.g. *"list tenders from Rand Water"*. The scrapers cover 9
portals: `etenders`, `transnet`, `sadc`, `zppa`, `ppadb`, `randwater`, `capetown`,
`cpbn`, `nra`.

### Asking about a tender that isn't loaded yet (local only)
On your PC, you don't need to load it first — **just ask** with its source + ID and
the app will **run the scraper, load it, then answer** (the first time is slower
because it has to download it). Works for 8 of the 9 sites; **nra** needs a human
check, so load that one by hand.
*(Set `ENABLE_SCRAPING=false` to turn auto-scrape off — e.g. when the server runs
somewhere without the scraper programs. It then answers only tenders already
loaded. See DEPLOY.md.)*

### Good questions to try
- *"What is the published date?"* vs *"What is the closing date?"* (different dates)
- *"What is the deadline?"* (a synonym — it still finds the closing date)
- *"Are alternative bids allowed?"* (an opposite/yes-no question)
- *"Who is the contact person?"* · *"What documents must I submit?"*
- *"List all the construction tenders."* (ask across all loaded tenders)

---

## 8. The MCP tools (for developers)

The assistant picks these automatically from their descriptions — you rarely name
them yourself. Every tool is a thin wrapper over `app/services`, so the RAG
pipeline is identical to the old HTTP version.

| Tool | What it does | Was |
|---|---|---|
| `ask_tender` | ask about ONE tender, grounded in its documents | `POST /chat` (with id) |
| `ask_all_tenders` | ask across EVERY loaded tender, attributing each finding | `POST /chat` (no id) |
| `summarize_tender` | full brief: scope, dates, fees, eligibility, documents | `POST /summary` |
| `list_tenders` | list loaded tenders (optional filter + limit) | `GET /tenders` |
| `get_tender` | full stored metadata for one tender | `GET /tenders/{id}` |
| `ingest_tender` | index a tender already scraped to disk | `POST /ingest` |
| `fetch_tender` | scrape a tender from its portal now, then index it — local only | `POST /fetch` |
| `ingest_all_tenders` | index every scraped tender found on disk | `POST /ingest-all` |
| `health_check` | database, pgvector, embeddings and chat provider status | `GET /health` |

It also exposes:
- **Resources** — `tender://catalogue` (everything loaded) and
  `tender://{source}/{tender_id}` (one tender's record), for clients that browse
  context rather than call tools.
- **A prompt** — `bid_assessment`, a ready-made "should we bid on this?" workflow.

**Reading the answer.** The question-answering tools return markdown: the answer,
then a **Sources** section listing document, page and relevance score. The
catalogue/ingest tools return structured JSON as well, so scripts can consume them.

**Inspect it by hand** with the MCP Inspector (a browser UI for calling tools
directly — needs Node.js installed, since it runs via `npx`):
```
.venv\Scripts\mcp dev app\mcp_server.py
```

---

## 9. Folder structure (what's where)

```
tender_rag/
  app/
    mcp_server.py      the MCP server — every tool, resource and prompt
    config.py          settings (read from the .env file)
    db.py              connects to PostgreSQL
    models.py          the 3 tables: Tender, Document, Chunk
    schemas.py         shapes of the tool inputs/outputs
    services/
      normalize.py     the 9 mappers (one per website)
      ingest_service.py loads tenders; also auto-fetches missing ones (local)
      scrape.py        runs a website's scraper on demand (local)
      chunking.py      splits documents into chunks
      embeddings.py    makes embeddings in-process (fastembed); Ollama/Gemini optional
      llm.py           answers via LangChain over a chat chain
                       (Llama 4 Maverick -> Scout on OpenRouter -> Groq gpt-oss backup -> optional local)
      retriever.py     the pgvector "find similar chunks" search
      prompts.py       the instructions given to the AI (synonyms, formatting, citing)
      rag.py           ties retrieve + answer together
      health.py        checks database / pgvector / providers; warms the models
  db/init.sql          creates the database tables (embedding VECTOR(384))
  scripts/             setup + loading helper scripts (load_data.py for a cloud DB)
  docs/architecture.md pictures/diagrams of the system
  Dockerfile           builds a container that serves the MCP server over stdio
  DEPLOY.md            connecting clients + using a hosted Postgres
  .env                 your settings + secret keys (keep private!)
  requirements.txt     the Python packages needed
```

*(The old `main.py`, `routers/` and `static/index.html` are gone — they were the
FastAPI HTTP layer that `mcp_server.py` replaces. `services/` is untouched.)*

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
   - `LLAMA_API_KEY` — a free key from https://openrouter.ai (chat = Llama 4).
   - *(optional)* `GROQ_API_KEY` — a free key from https://console.groq.com (backup model).
   - `EMBED_PROVIDER=fastembed` (the default in-process embeddings).
5. **Create the database + tables:**
   `powershell -ExecutionPolicy Bypass -File scripts\setup_db.ps1`
6. **Load some tenders:** `.venv\Scripts\python scripts\ingest_all.py`

*(Optional: install Ollama only if you want an offline chat fallback —
`winget install Ollama.Ollama`, pull a small chat model, and set `CHAT_MODEL` in
`.env`. Not used for embeddings.)*

7. **Connect it to your assistant:** see Section 7.

### Putting it online
By default the server runs on **your machine** over stdio, launched by your
assistant. To reach it from anywhere, serve the HTTP transport instead:

```
.venv\Scripts\python -m app.mcp_server --http
```

That exposes `/mcp`. It is **open by default** — set `MCP_AUTH_TOKEN` to require
`Authorization: Bearer <token>` instead. `render.yaml` deploys it free on Render;
see **[DEPLOY.md](DEPLOY.md)** for the walkthrough. Hosted means query-only (no
scraper binaries) and needs a hosted Postgres.

---

## 11. Start / stop the server

**You don't start it yourself.** Once connected (Section 7), your AI assistant
launches the server when it needs it and shuts it down when it closes — there is no
window to leave open and no port to remember.

To restart it after changing code or `.env`, restart the assistant (in Claude
Desktop: quit and reopen).

**Run it by hand** — only useful for debugging; it will just sit waiting for
JSON-RPC on stdin:
```
cd "c:\anshul\MVP - Copy\tender_rag"
.venv\Scripts\python -m app.mcp_server
```
Press `Ctrl + C` to stop. To poke at the tools interactively, use the Inspector
(Section 8) instead.

The first question after a start takes a few seconds (the embedding model loads
once — the server pre-warms it in the background); after that, most of each
answer's time is the Llama 4 cloud call (~1–3s).

---

## 12. Common questions

- **Do I need internet?** Only for the Llama 4 call (the answer-writer). The
  embeddings and the database run on your machine. Set a local `CHAT_MODEL` (via
  Ollama) and drop the cloud keys to run fully offline with a local model (slower).
- **Is it free?** Yes. PostgreSQL and the embeddings are free; Llama 4 runs on an
  OpenRouter free key; and the whole thing can be hosted free (see DEPLOY.md).
- **Does it use my ChatGPT/Claude account?** The *tender answer* is written by
  **Llama 4** (via OpenRouter, through LangChain), grounded in the documents, with
  embeddings from a small built-in model. Your assistant (e.g. Claude) only decides
  *which tool to call* and relays the result — it does not read the documents itself
  or invent tender facts.
- **The first question after starting is slow (~a few seconds).** That's the embedding
  model loading once. After that, each answer is mostly the Llama 4 cloud call (~1–3s).
- **Asking about a brand-new tender is slow (local).** The first time, it downloads
  it (scrape). After that it's instant. Large sites can take a few minutes.
- **It said "not available for this tender".** That's correct behaviour — the fact
  isn't in that tender's documents, so it won't guess.
- **A Tender ID didn't work.** Copy it **exactly** — many IDs contain `/` slashes
  and spaces (e.g. `RW10414743/25 R`). An underscore instead of a slash won't match.
- **Where are my secrets?** In `tender_rag/.env` (database password + the Llama 4
  OpenRouter key, plus the optional Groq backup key). Keep this file private — don't
  put it in a shared repository.

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
- With `ENABLE_SCRAPING=false` the server is **query-only** — it answers loaded
  tenders but doesn't scrape new ones.
- **One client at a time.** A stdio MCP server is launched by (and belongs to) the
  assistant that started it. Two assistants each get their own copy of the process;
  they share the database, not the server.

---

For the technical diagrams (system, database, and step-by-step flows), see
[docs/architecture.md](docs/architecture.md).
