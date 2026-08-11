"""
Embeddings. Two providers, chosen by config:
  - "gemini"  — Google's text-embedding-004 (768-dim), a free cloud API. Used when
    GOOGLE_API_KEY is set. Needed on free hosts that can't run Ollama.
  - "ollama"  — local nomic-embed-text (768-dim). Free, offline, no key (default).

Both are 768-dim, so the VECTOR(768) schema is unchanged either way. The SAME
provider must embed the stored chunks and the query, so a deployment picks one and
re-embeds the corpus with it (see scripts/load_data.py).

Gemini supports asymmetric retrieval embeddings: documents are embedded with
taskType RETRIEVAL_DOCUMENT and questions with RETRIEVAL_QUERY, which improves
match quality. Ollama here embeds both the same way.
"""
from __future__ import annotations

import time

import httpx

from app.config import settings


class EmbeddingError(RuntimeError):
    pass


# One PERSISTENT client per provider, reused across calls. A fresh connection makes
# the first request on it costly (Ollama ~2.5s cold; TLS handshake for Gemini);
# reusing it keeps each embed fast. httpx.Client is thread-safe, which is what the
# MCP server needs: sync tools run in anyio's worker threadpool.
_OLLAMA: httpx.Client | None = None
_GEMINI: httpx.Client | None = None

_MAX_CHARS = 8000     # safety cap; a ~500-token chunk is well under this


def _ollama_client() -> httpx.Client:
    global _OLLAMA
    if _OLLAMA is None or _OLLAMA.is_closed:
        _OLLAMA = httpx.Client(
            base_url=settings.ollama_url, timeout=600.0,
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0))
    return _OLLAMA


def _gemini_client() -> httpx.Client:
    global _GEMINI
    if _GEMINI is None or _GEMINI.is_closed:
        _GEMINI = httpx.Client(
            base_url=settings.gemini_url, timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0))
    return _GEMINI


def embed_texts(texts: list[str],
                task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed texts, preserving order. task_type ('query' vs document) matters for
    fastembed (bge query prefix) and Gemini; Ollama ignores it."""
    if not texts:
        return []
    if settings.embed_provider == "fastembed":
        return _embed_fastembed(texts, task_type)
    if settings.embed_provider == "gemini":
        return _embed_gemini(texts, task_type)
    return _embed_ollama(texts)


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


# --- fastembed (in-process ONNX, free, offline) -----------------------------

_FASTEMBED = None   # loaded lazily on first use (model load costs ~1-20s)


def _fastembed_model():
    global _FASTEMBED
    if _FASTEMBED is None:
        try:
            from fastembed import TextEmbedding
        except ImportError as e:  # pragma: no cover
            raise EmbeddingError(
                "fastembed is not installed — `pip install fastembed`") from e
        _FASTEMBED = TextEmbedding(model_name=settings.fastembed_model,
                                   cache_dir=settings.fastembed_cache or None)
    return _FASTEMBED


def _embed_fastembed(texts: list[str], task_type: str) -> list[list[float]]:
    model = _fastembed_model()
    try:
        # bge retrieval is asymmetric: queries get a search prefix (query_embed),
        # passages are embedded as-is (embed).
        clean = [(t or " ")[:_MAX_CHARS] for t in texts]
        gen = model.query_embed(clean) if task_type == "RETRIEVAL_QUERY" \
            else model.embed(clean)
        return [v.tolist() for v in gen]
    except Exception as e:  # noqa: BLE001
        raise EmbeddingError(
            f"fastembed failed ({settings.fastembed_model}): {e}") from e


# --- Gemini (cloud) ---------------------------------------------------------

# 50 chunks/request keeps a single batch under the free-tier per-minute token
# budget; the retry loop below paces the rest when the minute budget runs out.
_GEMINI_BATCH = 50


def _embed_gemini(texts: list[str], task_type: str) -> list[list[float]]:
    model = settings.gemini_embed_model
    full = f"models/{model}"
    c = _gemini_client()
    vectors: list[list[float]] = []
    try:
        for i in range(0, len(texts), _GEMINI_BATCH):
            batch = texts[i:i + _GEMINI_BATCH]
            reqs = [{"model": full,
                     "content": {"parts": [{"text": (t or " ")[:_MAX_CHARS]}]},
                     "taskType": task_type,
                     "outputDimensionality": settings.embed_dim} for t in batch]
            embs = _gemini_batch(c, model, reqs)
            if len(embs) != len(batch):
                raise EmbeddingError(
                    f"Gemini returned {len(embs)} embeddings for {len(batch)} inputs")
            vectors.extend(e.get("values", []) for e in embs)
        return vectors
    except EmbeddingError:
        raise
    except httpx.HTTPError as e:
        raise EmbeddingError(f"Gemini embeddings failed ({model}): {e}") from e


def _gemini_batch(c: httpx.Client, model: str, reqs: list[dict],
                  max_tries: int = 8) -> list[dict]:
    """POST one batch, waiting out free-tier rate limits (HTTP 429) and retrying."""
    for attempt in range(max_tries):
        r = c.post(f"/models/{model}:batchEmbedContents",
                   params={"key": settings.google_api_key},
                   json={"requests": reqs})
        if r.status_code in (401, 403):
            raise EmbeddingError(
                "Gemini rejected the API key (get a free one at "
                "https://aistudio.google.com/apikey and set GOOGLE_API_KEY).")
        if r.status_code == 429 and attempt < max_tries - 1:
            time.sleep(_gemini_retry_delay(r, attempt))
            continue
        r.raise_for_status()
        return r.json().get("embeddings") or []
    raise EmbeddingError(
        "Gemini rate limit (429) persisted after retries — the free-tier daily "
        "quota may be exhausted. Wait a while and re-run the load.")


def _gemini_retry_delay(r: httpx.Response, attempt: int) -> float:
    """Prefer Google's suggested retryDelay; else exponential backoff (cap 60s)."""
    try:
        for d in r.json().get("error", {}).get("details", []):
            rd = d.get("retryDelay")
            if rd:
                return min(float(str(rd).rstrip("s")) + 1.0, 65.0)
    except Exception:  # noqa: BLE001
        pass
    return min(2.0 ** attempt, 60.0)


# --- Ollama (local) ---------------------------------------------------------

_BATCH = 64   # keep each request modest (some docs produce hundreds of chunks)


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    try:
        c = _ollama_client()              # reused, NOT closed here
        for i in range(0, len(texts), _BATCH):
            batch = texts[i:i + _BATCH]
            r = c.post("/api/embed", json={"model": settings.embed_model,
                                           "input": batch})
            if r.status_code == 404:                 # older Ollama
                vectors.extend(_embed_one_safe(c, t) for t in batch)
                continue
            r.raise_for_status()
            vecs = r.json().get("embeddings")
            if vecs and len(vecs) == len(batch):
                vectors.extend(vecs)
            else:
                # Ollama drops inputs that tokenize to empty (a page-number-only
                # or control-char chunk), so a batch can come back short. Re-embed
                # that batch one at a time so every chunk still gets a vector;
                # a degenerate one falls back to a zero vector (harmless — it just
                # won't surface in cosine search) rather than skipping the tender.
                vectors.extend(_embed_one_safe(c, t) for t in batch)
        return vectors
    except httpx.HTTPError as e:
        raise EmbeddingError(f"Ollama embeddings failed ({settings.embed_model}): "
                             f"{e}. Is 'ollama serve' up and the model pulled? "
                             f"`ollama pull {settings.embed_model}`") from e


def _embed_one_safe(c: httpx.Client, text: str) -> list[float]:
    """Embed a single text; a degenerate input that yields nothing -> zero vector."""
    try:
        r = c.post("/api/embed", json={"model": settings.embed_model, "input": text})
        if r.status_code == 404:
            r = c.post("/api/embeddings", json={"model": settings.embed_model,
                                                "prompt": text})
            r.raise_for_status()
            vec = r.json().get("embedding")
        else:
            r.raise_for_status()
            got = r.json().get("embeddings") or []
            vec = got[0] if got else None
    except httpx.HTTPError:
        vec = None
    return vec if vec else [0.0] * settings.embed_dim
