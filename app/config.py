"""Settings loaded from environment / .env (pydantic-settings)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8-sig",   # tolerate a BOM from Notepad/PowerShell
        extra="ignore",
    )

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "tender_rag"
    postgres_user: str = "postgres"
    postgres_password: str = ""
    # Cloud Postgres (Neon/Supabase) requires SSL — set to "require" there.
    postgres_sslmode: str = ""

    # Which embedding backend to use: "fastembed" | "ollama" | "gemini".
    #   fastembed — in-process ONNX model (bge-small, 384-dim). Free, no API, no
    #               quotas; the default for cloud/free hosts.
    #   ollama    — local nomic-embed-text (768-dim); the local-dev default.
    #   gemini    — Google's cloud API (768-dim) — kept as an option (has a hard
    #               free-tier daily cap, so not used for bulk corpora).
    # The provider fixes embed_dim (below), which must match the VECTOR(n) column.
    embed_provider: str = "ollama"

    # fastembed (in-process, free, offline). Default bge-small (384) fits the free
    # cloud's RAM; local/production can override to bge-base (768) for better recall.
    # fastembed_dim MUST match the model (bge-small=384, bge-base=768, bge-large=1024).
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_dim: int = 384
    fastembed_cache: str = ""      # model cache dir (Docker bakes the model here)

    # Ollama (local, free) — embeddings, and an OPTIONAL offline chat fallback.
    # chat_model is EMPTY by default (llama3.2 removed): the cloud Llama 4 API is
    # the chat model. Set CHAT_MODEL only to keep a local offline fallback.
    ollama_url: str = "http://localhost:11434"
    chat_model: str = ""
    embed_model: str = "nomic-embed-text"

    # Google Gemini (cloud) — embeddings (768-dim, Matryoshka-truncated). Optional.
    google_api_key: str = ""
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # --- Cloud chat: OpenAI-compatible providers, tried in order --------------
    # PRIMARY: Llama 4 via OpenRouter (OpenAI-compatible), so the same chat code
    # works; only base URL + key + model differ. Free key at https://openrouter.ai .
    # (Swap URL + model for any other OpenAI-compatible Llama 4 host if you prefer.)
    llama_api_key: str = ""
    llama_api_url: str = "https://openrouter.ai/api/v1"
    llama_model: str = "meta-llama/llama-4-maverick"
    # Fallback Llama models (comma-list), tried when the primary is rate-limited.
    # Scout may be the better primary — over 3 runs on real prompts it held
    # 2.6-3.9s while Maverick ranged 3.2s to 64s — but 3 runs is too thin to
    # justify the swap, and free-tier queueing moves far more than model choice.
    # Swap LLAMA_MODEL/LLAMA_FALLBACK_MODELS if you want to try it.
    llama_fallback_models: str = "meta-llama/llama-4-scout"

    # SECONDARY (optional): Groq — a cross-provider safety net using NON-Llama
    # models (OpenAI gpt-oss), each with its own free daily budget, so chat still
    # answers if the Llama API is rate-limited. Leave GROQ_API_KEY blank to skip.
    groq_api_key: str = ""
    groq_url: str = "https://api.groq.com/openai/v1"
    # 20b before 120b: on real prompts the small model stayed ~1.3s while 120b
    # ranged 6-36s, so the larger model is the worse backup as well as the slower one.
    groq_fallback_models: str = "openai/gpt-oss-20b,openai/gpt-oss-120b"

    # --- MCP transport ------------------------------------------------------
    # stdio (default) needs none of this: the client owns the process, so it is
    # already as trusted as the user. The HTTP transport is a public endpoint;
    # set a shared bearer token here to require `Authorization: Bearer <token>`.
    # Blank (the default) serves it OPEN — anyone with the URL can call any tool.
    mcp_auth_token: str = ""
    mcp_host: str = "127.0.0.1"          # 0.0.0.0 when hosted
    mcp_port: int = 8000                 # hosts inject $PORT; see main()
    mcp_path: str = "/mcp"
    # Host headers to accept (DNS-rebinding protection). Comma-separated. Render
    # injects RENDER_EXTERNAL_HOSTNAME, which is added automatically below.
    mcp_allowed_hosts: str = ""
    render_external_hostname: str = ""   # set by Render; unused elsewhere

    @property
    def allowed_hosts(self) -> list[str]:
        """Host headers the HTTP transport will accept.

        The SDK rejects every Host unless it is listed, so a hosted deployment
        MUST include its public hostname or every request 421s.
        """
        hosts = [h.strip() for h in self.mcp_allowed_hosts.split(",") if h.strip()]
        if self.render_external_hostname:
            hosts.append(self.render_external_hostname)
        if not hosts:
            hosts = ["localhost", "127.0.0.1",
                     f"localhost:{self.mcp_port}", f"127.0.0.1:{self.mcp_port}"]
        return hosts

    # RAG params
    chunk_tokens: int = 500
    chunk_overlap: int = 80
    top_k: int = 8      # chunks sent to the LLM (after fusion)
    # Hybrid retrieval: fuse keyword (full-text) + vector rankings via RRF. Keyword
    # search catches exact tokens (tender numbers, form codes) that vectors miss.
    hybrid_search: bool = True
    retrieve_candidates: int = 40   # per-ranker pool pulled before fusing to top_k
    # Cross-encoder reranker (in-process, fastembed): re-scores the fused candidates
    # so the most relevant chunks reach the LLM. ON by default (local); the free
    # cloud sets USE_RERANKER=false to stay within its small RAM.
    use_reranker: bool = True
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Reranking costs ~170ms PER CANDIDATE and scales linearly (measured), so this
    # number is the single biggest latency knob. 12 keeps a 1.5x pool over top_k.
    rerank_candidates: int = 12
    # The summary runs 12 aspect queries, so its pool is multiplied by 12 and gets
    # its own knob. Measured recall vs cost (3 tenders x 12 aspects, per summary):
    #   no rerank  47% covered   0.9s
    #   pool  6    58% covered  14.1s
    #   pool 12    61% covered  33.2s   <- chosen
    #   pool 24    67% covered  64.5s   (the original)
    # Reranking clearly earns its keep (+11pts over none), and the pool matters,
    # so this is a real accuracy/latency dial rather than free savings. 12 halves
    # the wait while giving back most of what a pool of 6 dropped. Raise it toward
    # 24 if a brief starts reporting "Not stated" for facts that are in the docs.
    summary_rerank_candidates: int = 12

    # Order the chat providers are tried in ("llama,groq" or "groq,llama").
    #
    # If you retune this, MEASURE WITH A REAL PROMPT. On a one-line prompt Groq
    # looks 2-5x faster than Llama; on the ~4k tokens a grounded answer actually
    # sends, that reverses. Measured here at 15KB of prompt, 3 runs each:
    #   llama-4-scout    2.6 / 2.8 / 3.9 s
    #   llama-4-maverick 3.2 / 3.7 / 64.3 s
    #   groq gpt-oss-20b 1.3 s, then rate-limited
    #   groq gpt-oss-120b 6.2 / 15.0 / 36.5 s
    # Free-tier queueing swings the same call from 3s to 66s, so treat any small
    # difference between endpoints as noise unless you have many samples.
    chat_provider_order: str = "llama,groq"

    # How long to wait on one endpoint before abandoning it for the next.
    #
    # Measured on an identical prompt, 6 runs per endpoint: every endpoint has a
    # median of 1.3-1.8s, but Groq stalled past 18s on 2 of 6 calls. Those stalls
    # are NOT correlated across providers, so a stalled call is worth abandoning
    # early — the next endpoint answers in ~1.5s. 12s is ~3x the slowest healthy
    # response seen (4.0s), so it cuts the tail without killing valid slow calls.
    #
    # Retries are 0 deliberately: retrying the endpoint that just stalled doubles
    # the wait before failover, and there are three more endpoints behind it.
    llm_timeout: float = 12.0
    llm_max_retries: int = 0

    # Answer length budget. Groq's gpt-oss models are REASONING models: they spend
    # this budget thinking before writing. At 600 a summary came back with
    # finish_reason='length' and output_tokens=600 of which 598 were reasoning —
    # i.e. an EMPTY answer that looked like success. They need room for both.
    # Careful: Groq's free TPM budget of 8000 counts input + max_tokens TOGETHER,
    # so raising this is not free — at 2500 a 6.2k-token summary was rejected 413
    # ("Requested 8686"). 1500 leaves reasoning room while staying inside the cap
    # for the summary prompt (~5.3k tokens at _SUMMARY_BUDGET=12).
    chat_max_tokens: int = 600      # Llama 4 — plain completion, 600 is plenty
    groq_max_tokens: int = 1500     # gpt-oss — reasoning tokens come out of this

    @property
    def chat_endpoints(self) -> list[dict]:
        """OpenAI-compatible chat endpoints to try in order. Each entry carries its
        own base URL + key + model, so one chat() loop can span both providers.
        Order follows chat_provider_order; within a provider, its own model order."""
        eps: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def add(provider: str, url: str, key: str, model: str,
                max_tokens: int) -> None:
            model = model.strip()
            if model and (provider, model) not in seen:
                seen.add((provider, model))
                eps.append({"provider": provider, "url": url, "key": key,
                            "model": model, "max_tokens": max_tokens})

        def llama() -> None:
            if self.llama_api_key.strip():
                for m in [self.llama_model, *self.llama_fallback_models.split(",")]:
                    add("llama", self.llama_api_url, self.llama_api_key, m,
                        self.chat_max_tokens)

        def groq() -> None:
            if self.groq_api_key.strip():
                for m in self.groq_fallback_models.split(","):
                    add("groq", self.groq_url, self.groq_api_key, m,
                        self.groq_max_tokens)

        builders = {"llama": llama, "groq": groq}
        for name in self.chat_provider_order.split(","):
            build = builders.get(name.strip().lower())
            if build:
                build()
        # Anything omitted from the order still belongs at the end: a configured
        # key should never be silently unusable because of a typo in the order.
        for build in builders.values():
            build()
        return eps

    @property
    def chat_provider(self) -> str:
        # "api" = at least one cloud endpoint (Llama API and/or Groq backup).
        return "api" if self.chat_endpoints else "ollama"

    @property
    def active_chat_model(self) -> str:
        eps = self.chat_endpoints
        return eps[0]["model"] if eps else (self.chat_model or "(none)")

    @property
    def embed_dim(self) -> int:
        # Must match the chunks.embedding VECTOR(n) column for the active provider.
        return self.fastembed_dim if self.embed_provider == "fastembed" else 768

    @property
    def active_embed_model(self) -> str:
        return {"fastembed": self.fastembed_model,
                "gemini": self.gemini_embed_model}.get(self.embed_provider,
                                                       self.embed_model)

    # On-demand scraping shells out to local scraper subprocesses that need
    # Playwright/Tesseract/LibreOffice + system Python 3.14. Cloud/free hosts don't
    # have those, so set ENABLE_SCRAPING=false there -> query-only over loaded data.
    enable_scraping: bool = True
    # Where the scrapers write their output (each source has <source>/output/).
    scrapers_root: Path = Path("c:/anshul/MVP")
    # The scrapers run on system Python 3.14 (fitz/pytesseract/playwright live there),
    # NOT this app's 3.12 venv. Used for on-demand scraping of un-ingested tenders.
    scraper_python: str = r"C:\Python314\python.exe"
    scrape_timeout: int = 300          # seconds; some portals (etenders feed) are slow

    @property
    def _db_query(self) -> str:
        return f"?sslmode={self.postgres_sslmode}" if self.postgres_sslmode else ""

    @property
    def database_url(self) -> str:
        return (f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
                f"{self._db_query}")

    @property
    def database_url_safe(self) -> str:
        return (f"postgresql+psycopg://{self.postgres_user}:***"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
                f"{self._db_query}")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
