# Deploy the Tender RAG app for FREE (beginner guide)

This puts your chatbot online at a public URL, using only free services. No credit
card, no servers to manage.

## The free stack

| Piece | Free service | What it does |
|---|---|---|
| **App** | **Render** (free web service) | runs the FastAPI app + chat page |
| **Database** | **Neon** (free Postgres + pgvector) | stores tenders + the "meaning numbers" |
| **Embeddings** | **built into the app** (fastembed) | turns text into vectors — in-process, no API |
| **Answers** | **Groq** (free API) | writes the chat answers |

**Why embeddings are built-in:** running them as a cloud API hit free-tier daily
caps, and running Ollama needs too much memory for a free host. So the app now makes
its own embeddings **inside the process** using a small ONNX model (`bge-small`,
384-dim). That means **no embedding API, no key, no quotas** — it just works.

```mermaid
flowchart LR
  U[Anyone with the link] --> R[Render<br/>FastAPI app + built-in embeddings]
  R --> N[(Neon<br/>Postgres + pgvector)]
  R --> Q[Groq API<br/>answers]
```

**One important limit:** the cloud app is **query-only**. It answers questions about
the tenders you load into it, but does **not** scrape new ones (scraping needs heavy
programs that don't fit a free host). To add tenders later, scrape them on your PC
and re-run the load step.

---

## Before you start

Free accounts (all free, no card):
- **Neon** — https://neon.tech (the database)
- **Groq** — https://console.groq.com (you already have this key)
- **Render** — https://render.com (the host), plus a **GitHub** account for the code.

There is **no Google/Gemini key needed** anymore.

---

## Step 1 — Create the database (Neon)

1. Sign up at https://neon.tech → **New Project** (any name/region).
2. Open **Dashboard → Connection Details** and copy the **connection string**:
   ```
   postgresql://neondb_owner:PASSWORD@ep-xxxx.REGION.aws.neon.tech/neondb?sslmode=require
   ```
   It contains the host, database, user, and password you'll need below.

---

## Step 2 — Load your tenders into Neon (from your PC)

This reads the tenders scraped on your PC and writes them — with the built-in
embeddings — into Neon. Nothing is re-scraped. Run it in `c:\anshul\MVP\tender_rag`.

Open **PowerShell** and set the values just for this session (does not touch your
local `.env`):

```powershell
$env:POSTGRES_HOST     = "ep-xxxx.REGION.aws.neon.tech"   # from your Neon string
$env:POSTGRES_DB       = "neondb"
$env:POSTGRES_USER     = "neondb_owner"
$env:POSTGRES_PASSWORD = "YOUR_NEON_PASSWORD"
$env:POSTGRES_SSLMODE  = "require"
$env:EMBED_PROVIDER    = "fastembed"

# create the tables (at the right vector size) and load all tenders
.venv\Scripts\python scripts\load_data.py --reset
```

`--reset` drops any old tables and recreates them, so it's safe to re-run. You should
see `Loaded: 15 tenders, ... chunks`. (The first run downloads the ~30 MB model once.)

> Close the PowerShell window afterwards so the values disappear. Your local `.env`
> is untouched.

---

## Step 3 — Put the code on GitHub

Render deploys from a Git repo. Push the `tender_rag` folder to a **new, empty**
GitHub repo (so the `Dockerfile` is at the repo root):

```powershell
cd c:\anshul\MVP\tender_rag
git add -A
git commit -m "deploy: in-process embeddings"
# create an EMPTY repo at https://github.com/new (no README), then:
git remote add origin https://github.com/<you>/tender-rag.git   # skip if already added
git push -u origin main
```

The first push pops up a **GitHub sign-in** — complete it. Your `.env` is git-ignored,
so no secrets are uploaded.

---

## Step 4 — Deploy on Render

1. https://render.com → **New → Blueprint** → connect GitHub → pick your repo.
   Render reads `render.yaml` and creates a free web service.
2. Fill the **secret** values it asks for:

   | Key | Value |
   |---|---|
   | `POSTGRES_HOST` | your Neon host (`ep-...neon.tech`) |
   | `POSTGRES_DB` | `neondb` |
   | `POSTGRES_USER` | your Neon user |
   | `POSTGRES_PASSWORD` | your Neon password |
   | `GROQ_API_KEY` | your Groq key (`gsk_...`) |

   The rest (`POSTGRES_SSLMODE=require`, `EMBED_PROVIDER=fastembed`,
   `ENABLE_SCRAPING=false`, `GROQ_MODEL`, `TOP_K`) are already set by `render.yaml`.
3. **Apply / Create.** First build ~4–6 min (it bakes the embedding model in).

**Manual alternative:** New → **Web Service** → connect repo → it detects the
`Dockerfile` → set the same env vars by hand → Create.

---

## Step 5 — Try it

1. Render gives a URL like `https://tender-rag.onrender.com`.
2. `https://<app>/health` → expect `"status": "ok"`, `embed_provider: fastembed`,
   `chat_provider: groq`, `pgvector: ok`.
3. `https://<app>/` → the chat page. Ask e.g. Source `zppa`, Tender ID `28231539`,
   *"What is the closing date?"*

Share the link — it's live.

---

## Good to know (free-tier behaviour)

- **First request is slow (~30–60s), then fast.** Render's free service sleeps after
  15 min idle and takes ~a minute to wake.
- **No scraping in the cloud** (by design). To add tenders: scrape on your PC, then
  re-run **Step 2** (safe to repeat).
- **Free limits:** Neon free = 0.5 GB (your data is tiny). Groq free tier is generous
  for one-at-a-time use; the app already retries / falls back on Groq rate limits.
  Embeddings are in-process, so they have **no limit at all**.
- **Secrets** live only in Render's env settings + your local shell — never in the code.

## If something's wrong

- `/health` → `postgres: error` → check the Neon values and `POSTGRES_SSLMODE=require`.
- `/health` → `groq_key: MISSING` → set `GROQ_API_KEY` on Render and redeploy.
- `embed_model: error` → the model didn't load; usually a transient first-boot issue —
  redeploy. (The image bakes the model in, so this is rare.)
- A question says *"not loaded on this server"* → that tender isn't in Neon; run
  Step 2, or omit the Tender ID to search all loaded tenders.
