# Running & deploying the Tender RAG MCP server

The server speaks two transports. Pick by who needs to reach it.

| | **stdio** (default) | **streamable HTTP** (`--http`) |
|---|---|---|
| Runs | on your PC, launched by your assistant | on a host, always on |
| Address | none — a child process | `https://<service>.onrender.com/mcp` |
| Auth | none needed (the client owns the process) | open by default; set `MCP_AUTH_TOKEN` for a bearer token |
| Scraping | works (`fetch_tender`) | off — no scraper binaries on the host |
| Best for | daily local use | testing online, sharing with others |

Start with **Part 1** if you just want it working. **Part 2** puts it online.

---

## Part 1 — Local (stdio)

### Claude Code
```
claude mcp add tender-rag -- "c:\anshul\MVP - Copy\tender_rag\.venv\Scripts\python.exe" -m app.mcp_server
```
Check it with `claude mcp list`.

### Claude Desktop
Edit `%APPDATA%\Claude\claude_desktop_config.json`:
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
Restart Claude Desktop. The tender tools appear under the tools icon.

**Both need the full path to `.venv\Scripts\python.exe`** — the system Python does
not have this project's packages. And `cwd` must be the project folder, because
that is where `.env` is read from.

> Prefer not to rely on `cwd`? Put the settings in the client config's `env` block
> instead (`POSTGRES_PASSWORD`, `LLAMA_API_KEY`, …); real environment variables win
> over `.env`.

**Verify:** `.venv\Scripts\python scripts\test_mcp.py`

---

## Part 2 — Online, on Render (HTTP)

### Step 1 — Create the database (Neon)
Render's free tier has no database, so the corpus lives in a free Neon Postgres.

1. Sign up at https://neon.tech → **New Project** (any name/region).
2. **Dashboard → Connection Details** → copy the connection string:
   ```
   postgresql://neondb_owner:PASSWORD@ep-xxxx.REGION.aws.neon.tech/neondb?sslmode=require
   ```

### Step 2 — Load your tenders into Neon (from your PC)
This reads the tenders scraped on your PC and writes them — with the built-in
embeddings — into Neon. Nothing is re-scraped.

```powershell
$env:POSTGRES_HOST     = "ep-xxxx.REGION.aws.neon.tech"   # from your Neon string
$env:POSTGRES_DB       = "neondb"
$env:POSTGRES_USER     = "neondb_owner"
$env:POSTGRES_PASSWORD = "YOUR_NEON_PASSWORD"
$env:POSTGRES_SSLMODE  = "require"
$env:EMBED_PROVIDER    = "fastembed"

.venv\Scripts\python scripts\load_data.py --reset
```

`--reset` drops any old tables and recreates them, so it's safe to re-run. Expect
`Loaded: N tenders, ... chunks`. (The first run downloads the ~30 MB model once.)

> ⚠️ The cloud uses `fastembed` (384-dim) while your local `.env` may use a
> 768-dim model. The vector column size must match the model, which is exactly
> what `--reset` rebuilds. Don't skip it when switching providers.

> Close that PowerShell window afterwards so the values disappear — your `.env` is
> untouched.

### Step 3 — Push to GitHub
Render deploys from a Git repo. Push this folder to a **new, empty** repo so the
`Dockerfile` sits at the repo root:

```powershell
cd "c:\anshul\MVP - Copy\tender_rag"
git add -A
git commit -m "MCP server over HTTP"
git remote add origin https://github.com/<you>/tender-rag.git   # skip if set
git push -u origin main
```
Your `.env` is git-ignored, so no secrets are uploaded.

### Step 4 — Deploy
1. https://render.com → **New → Blueprint** → connect GitHub → pick the repo.
   Render reads `render.yaml` and creates a free web service.
2. Fill the **secret** values it asks for:

   | Key | Value |
   |---|---|
   | `POSTGRES_HOST` | your Neon host (`ep-...neon.tech`) |
   | `POSTGRES_DB` | `neondb` |
   | `POSTGRES_USER` | your Neon user |
   | `POSTGRES_PASSWORD` | your Neon password |
   | `LLAMA_API_KEY` | your OpenRouter key |
   | `GROQ_API_KEY` | *(optional backup)* |

   `MCP_AUTH_TOKEN` is left **blank** by `render.yaml`, which serves the endpoint
   **open** — see [Security notes](#security-notes-for-the-http-transport).
   Everything else is already set.
3. **Apply.** First build ~4–6 min (it bakes the embedding model into the image).

### Step 5 — Test
1. Liveness:
   ```
   curl https://<service>.onrender.com/healthz
   → {"status":"alive","server":"tender-rag"}
   ```
2. Point a client at it — no header needed while auth is off:
   ```json
   {
     "mcpServers": {
       "tender-rag": {
         "type": "http",
         "url": "https://<service>.onrender.com/mcp"
       }
     }
   }
   ```
   Or with the CLI:
   ```
   claude mcp add --transport http tender-rag-cloud https://<service>.onrender.com/mcp
   ```

> A browser can't test `/mcp`: it is POST-only and there is no `/` route, so the
> address bar gives you a 404 (or a 401 while auth is on). Use a client.

### Turning auth back on
Set `MCP_AUTH_TOKEN` in Render → **Environment** (the **Generate** button makes a
strong value) and redeploy. Every request then needs the header, and clients add:
```json
"headers": { "Authorization": "Bearer <MCP_AUTH_TOKEN>" }
```
```
claude mcp add --transport http tender-rag-cloud https://<service>.onrender.com/mcp \
  --header "Authorization: Bearer <MCP_AUTH_TOKEN>"
```

### What the free tier costs you
- **Sleeps after 15 min idle**, ~1 min to wake. The first call after a nap may time
  out in the client — call it again.
- **512 MB RAM**, so `USE_RERANKER=false` (already set). Hybrid search still runs;
  answers are slightly less precisely ranked.
- **Query-only.** `fetch_tender` can't scrape there. To add tenders: scrape on your
  PC, re-run Step 2.

---

## Part 3 — Docker (either transport)

```
docker build -t tender-rag .

# stdio — note -i, the container IS the server
docker run -i --rm --env-file .env tender-rag

# HTTP
docker run -p 8000:8000 --env-file .env tender-rag \
  python -m app.mcp_server --http --host 0.0.0.0
# add -e MCP_AUTH_TOKEN=<token> to require a bearer token
```

---

## Security notes for the HTTP transport

- **Auth is currently OFF** (`MCP_AUTH_TOKEN` blank), so the URL *is* the only
  secret. Anyone who finds it can call every tool — read your whole corpus, and
  run `ingest_all_tenders` / `fetch_tender`, which burn CPU and your LLM credits.
  `health_check` also reports your configuration. Set the token to close this.
- **The `.onrender.com` hostname is guessable and gets crawled.** Treat an open
  deployment as public data, and keep anything confidential out of the corpus.
- **`/healthz` is always unauthenticated** (hosts must probe it) and returns
  nothing but a liveness flag.
- **DNS-rebinding protection** rejects unknown `Host` headers with `421`. Render's
  hostname is trusted automatically via `RENDER_EXTERNAL_HOSTNAME`; on any other
  host set `MCP_ALLOWED_HOSTS=your.domain.com`.
- **Sessions are stateless by default**, so a free-tier restart can't orphan a
  client with a dead session id. Pass `--stateful` to opt out.

## If something's wrong

- **Server missing from the client (stdio).** Check the command path points at
  `.venv\Scripts\python.exe`, not a system Python. Claude Desktop logs:
  `%APPDATA%\Claude\logs\`.
- **Everything returns 401.** A token is set — either clear `MCP_AUTH_TOKEN` in
  Render, or send exactly `Authorization: Bearer <token>` matching its value.
  (Browsing `/` gives this too: only `/healthz` is exempt.)
- **`421 Misdirected Request`.** The `Host` header isn't in the allowlist — set
  `MCP_ALLOWED_HOSTS`.
- **`postgres: error`** → check the values and, for Neon, `POSTGRES_SSLMODE=require`.
- **`llama_api_key: not set`** → set it and redeploy / restart the client.
- **Dimension mismatch after switching embedding providers** → re-run
  `scripts/load_data.py --reset`.
- **"tender … is not loaded"** → it isn't in that database. Load it, or ask across
  everything with `ask_all_tenders`.
- **Protocol errors on stdio.** Something wrote to **stdout**, corrupting JSON-RPC.
  All logging goes to stderr — keep it that way, and never add a bare `print()`.
