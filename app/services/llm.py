"""
Chat completion. Uses Groq (cloud, free tier, fast, strong model) when a
GROQ_API_KEY is set, else falls back to local Ollama (offline). Same interface
either way. Embeddings are separate and always local (see embeddings.py).

Resilience: Groq's free tier is rate-limited (~12k tokens/min for the 70b model),
so a quick burst of questions can hit HTTP 429. Rather than surface that as an
error, we (1) briefly auto-retry when Groq says the limit clears soon, and
(2) fall back to the local Ollama model so the user always gets a grounded answer.
"""
from __future__ import annotations

import re
import time

import httpx

from app.config import settings


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Transient failure (rate limit / service error) - safe to fall back to local."""


# Persistent clients: reusing the connection avoids a per-call TLS/handshake cost
# (Groq is HTTPS) and keeps latency low + consistent.
_GROQ: httpx.Client | None = None


def _groq_client() -> httpx.Client:
    global _GROQ
    if _GROQ is None or _GROQ.is_closed:
        _GROQ = httpx.Client(
            base_url=settings.groq_url, timeout=60.0,
            headers={"Authorization": f"Bearer {settings.groq_api_key}",
                     "Content-Type": "application/json"},
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0))
    return _GROQ


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    # temperature 0 = deterministic: the same question gives the same answer.
    #
    # 3-tier fallback keeps answers fast AND always available:
    #   1. Groq main model (70b)   - best quality, ~1-2s
    #   2. Groq fallback model (8b) - separate daily budget, still fast (~1s)
    #   3. local Ollama            - offline last resort, slow (~1-2 min) but works
    if settings.chat_provider == "groq":
        try:
            return _chat_groq(system, user, temperature, settings.groq_model)
        except LLMUnavailable as e:
            fb = settings.groq_fallback_model.strip()
            if fb and fb != settings.groq_model:
                try:
                    return _chat_groq(system, user, temperature, fb)
                except LLMUnavailable:
                    pass          # both Groq models exhausted -> go local
            try:
                return _chat_ollama(system, user, temperature)
            except LLMError:
                raise LLMError(
                    f"{e} The local fallback ({settings.chat_model}) is also "
                    f"unavailable - wait ~30s and retry, or run `ollama serve`."
                ) from e
    return _chat_ollama(system, user, temperature)


# --- Groq -------------------------------------------------------------------

# If Groq says the rate limit clears within this many seconds, wait it out here
# (Groq is still faster + stronger than the local model). Longer than this, we
# give up and let chat() fall back to Ollama.
_MAX_INLINE_WAIT = 15.0


def _parse_duration(s: str) -> float:
    """Groq durations like '2m52.8s', '205ms', '15s' -> seconds."""
    total = 0.0
    for num, unit in re.findall(r"([0-9.]+)\s*(ms|m|s)", s or ""):
        v = float(num)
        total += v / 1000 if unit == "ms" else v * 60 if unit == "m" else v
    return total


def _retry_after_seconds(r: httpx.Response) -> float:
    """How long Groq asks us to wait, from headers (retry-after or reset-tokens)."""
    ra = r.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    return (_parse_duration(r.headers.get("x-ratelimit-reset-tokens", ""))
            or _parse_duration(r.headers.get("x-ratelimit-reset-requests", "")))


def _chat_groq(system: str, user: str, temperature: float, model: str) -> str:
    """Groq's OpenAI-compatible /chat/completions (httpx - no extra dependency)."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": 800,
    }
    delay = 0.0
    for attempt in range(3):
        if delay:
            time.sleep(delay)
            delay = 0.0
        try:
            r = _groq_client().post("/chat/completions", json=payload)
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"Groq unreachable: {e}") from e

        if r.status_code == 401:
            raise LLMError("Groq rejected the API key (401). Check GROQ_API_KEY "
                           "in .env (free key at console.groq.com).")
        if r.status_code == 429:
            body = r.text.lower()
            # A DAILY cap (TPD) won't clear in a few seconds, so don't waste an
            # inline retry - fail fast and let chat() drop to the fallback model.
            # Only a brief per-minute (TPM) blip is worth waiting out here.
            daily = "per day" in body or "tpd" in body
            wait = _retry_after_seconds(r)
            if not daily and attempt < 2 and 0 < wait <= _MAX_INLINE_WAIT:
                delay = wait + 0.5      # brief per-minute blip: wait, retry Groq
                continue
            raise LLMUnavailable(
                f"Groq rate limit (429) on {model}"
                + (" (daily budget used up)" if daily else f"; clears in ~{wait:.0f}s"))
        if r.status_code >= 500:
            raise LLMUnavailable(f"Groq service error ({r.status_code}).")
        try:
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"Groq chat failed ({model}): {e}") from e

    raise LLMUnavailable(f"Groq rate limit persisted after retries ({model}).")


# --- Ollama (local fallback) ------------------------------------------------

def _chat_ollama(system: str, user: str, temperature: float) -> str:
    payload = {
        "model": settings.chat_model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "keep_alive": "30m",          # keep the model resident, avoid reloads
        "options": {"temperature": temperature},
    }
    last: Exception | None = None
    for _ in (1, 2):
        try:
            with httpx.Client(base_url=settings.ollama_url, timeout=300.0) as c:
                r = c.post("/api/chat", json=payload)
                r.raise_for_status()
                return (r.json().get("message") or {}).get("content", "").strip()
        except httpx.HTTPError as e:
            last = e
    raise LLMError(
        f"Ollama chat failed ({settings.chat_model}) after 2 tries: {last}. "
        f"Is 'ollama serve' up and the model pulled? `ollama pull {settings.chat_model}`")
