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

    # Ollama (local, free) — embeddings + chat fallback when no cloud keys are set
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768

    # Google Gemini (cloud, free tier) — embeddings. If a key is set, embeddings use
    # Gemini instead of local Ollama (needed on free hosts that can't run Ollama).
    # gemini-embedding-001 is the current model; it natively emits 3072 dims but
    # supports Matryoshka truncation, so we request outputDimensionality = embed_dim
    # (768) to match the VECTOR(768) schema — no migration. Cosine search is
    # scale-invariant, so the truncated vectors need no manual normalisation.
    google_api_key: str = ""
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Groq (cloud, free tier) — chat only. If a key is set, chat uses Groq.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Smaller/faster Groq model, tried when the main one is rate-limited. It has a
    # SEPARATE free-tier budget, so answers stay fast (~1s) even after the big
    # model's daily token cap is used up — before falling back to the slow local
    # model. Set empty to disable this middle tier.
    groq_fallback_model: str = "llama-3.1-8b-instant"
    groq_url: str = "https://api.groq.com/openai/v1"

    # RAG params
    chunk_tokens: int = 500
    chunk_overlap: int = 80
    top_k: int = 4

    @property
    def chat_provider(self) -> str:
        return "groq" if self.groq_api_key.strip() else "ollama"

    @property
    def active_chat_model(self) -> str:
        return self.groq_model if self.chat_provider == "groq" else self.chat_model

    @property
    def embed_provider(self) -> str:
        return "gemini" if self.google_api_key.strip() else "ollama"

    @property
    def active_embed_model(self) -> str:
        return (self.gemini_embed_model if self.embed_provider == "gemini"
                else self.embed_model)

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
