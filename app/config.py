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
    llama_fallback_models: str = "meta-llama/llama-4-scout"

    # SECONDARY (optional): Groq — a cross-provider safety net using NON-Llama
    # models (OpenAI gpt-oss), each with its own free daily budget, so chat still
    # answers if the Llama API is rate-limited. Leave GROQ_API_KEY blank to skip.
    groq_api_key: str = ""
    groq_url: str = "https://api.groq.com/openai/v1"
    groq_fallback_models: str = "openai/gpt-oss-120b,openai/gpt-oss-20b"

    # --- MCP transport ------------------------------------------------------
    # stdio (default) needs none of this: the client owns the process, so it is
    # already as trusted as the user. The HTTP transport is a public endpoint,
    # so it requires a shared bearer token — the server refuses to start in HTTP
    # mode without one rather than silently exposing the corpus.
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
    rerank_candidates: int = 24     # fused hits pulled, then reranked down to top_k

    @property
    def chat_endpoints(self) -> list[dict]:
        """OpenAI-compatible chat endpoints to try in order: the Llama API first
        (Llama 4), then Groq's non-Llama backup models. Each entry carries its own
        base URL + key + model, so one chat() loop can span both providers."""
        eps: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def add(provider: str, url: str, key: str, model: str) -> None:
            model = model.strip()
            if model and (provider, model) not in seen:
                seen.add((provider, model))
                eps.append({"provider": provider, "url": url, "key": key,
                            "model": model})

        if self.llama_api_key.strip():
            for m in [self.llama_model, *self.llama_fallback_models.split(",")]:
                add("llama", self.llama_api_url, self.llama_api_key, m)
        if self.groq_api_key.strip():
            for m in self.groq_fallback_models.split(","):
                add("groq", self.groq_url, self.groq_api_key, m)
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
