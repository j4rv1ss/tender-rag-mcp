"""
Chat completion via LangChain.

Primary provider is Meta's official Llama API (Llama 4). Because that endpoint is
OpenAI-compatible, we drive it with LangChain's ChatOpenAI (base_url + key + model)
— the same class also drives Groq's OpenAI-compatible endpoint, so one code path
spans both providers. An optional local Ollama model is the offline last resort.

Resilience: each cloud endpoint has its own rate limit / daily budget. The openai
client under ChatOpenAI already retries a 429 with backoff (max_retries); when an
endpoint is still unavailable after that, chat() moves on to the next endpoint in
settings.chat_endpoints (Llama 4 Maverick -> Scout -> Groq gpt-oss backup), so the
user always gets a grounded answer. Embeddings are separate and local (embeddings.py).
"""
from __future__ import annotations

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """Transient failure (rate limit / service error) - safe to try the next endpoint."""


# One ChatOpenAI per (provider, model, temperature). Reusing it keeps the underlying
# httpx connection warm (lower, more consistent latency) across questions.
_MODELS: dict[str, ChatOpenAI] = {}

# Fail over fast rather than wait out a stall — see llm_timeout in config for the
# measurements behind these. Overridable via LLM_TIMEOUT / LLM_MAX_RETRIES.
_MAX_RETRIES = settings.llm_max_retries
_TIMEOUT = settings.llm_timeout


def _model_for(endpoint: dict, temperature: float) -> ChatOpenAI:
    key = f'{endpoint["provider"]}:{endpoint["model"]}:{temperature}'
    llm = _MODELS.get(key)
    if llm is None:
        llm = ChatOpenAI(
            model=endpoint["model"],
            base_url=endpoint["url"],
            api_key=endpoint["key"],
            temperature=temperature,   # 0 = deterministic, best for factual QA
            # Per-endpoint: reasoning models burn this budget before they write.
            max_tokens=endpoint.get("max_tokens", 600),
            timeout=_TIMEOUT,
            max_retries=_MAX_RETRIES,
        )
        _MODELS[key] = llm
    return llm


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    # temperature 0 = deterministic: the same question gives the same answer.
    #
    # Try each cloud endpoint in turn (Llama 4 first, then the Groq gpt-oss backup);
    # each has its OWN budget, so the app keeps answering when one is capped. If no
    # cloud key is configured, fall back to the optional local Ollama model.
    endpoints = settings.chat_endpoints
    if endpoints:
        last: Exception | None = None
        for ep in endpoints:
            try:
                return _chat_endpoint(system, user, temperature, ep)
            except LLMUnavailable as e:
                last = e            # this endpoint is rate-limited/down -> try next
        if settings.chat_model.strip():
            try:
                return _chat_ollama(system, user, temperature)
            except LLMError:
                pass
        raise LLMError(
            "Every chat endpoint is unavailable (rate limit / bad key / down) and "
            "no local fallback is configured. Check LLAMA_API_KEY in .env and try "
            f"again shortly. (last: {last})") from last
    return _chat_ollama(system, user, temperature)


def chat_tools(messages: list, tools: list[dict], temperature: float = 0.0):
    """Like chat(), but the model may answer with tool calls instead of prose.

    Walks the same endpoint chain with the same failover rules, and returns the
    raw AIMessage so the caller can inspect .tool_calls. Note the empty-content
    guard from _chat_endpoint deliberately does NOT apply here: a reply that is
    only tool calls has empty content and is perfectly valid.
    """
    endpoints = settings.chat_endpoints
    if not endpoints:
        raise LLMError("tool calling needs a cloud endpoint; no API key is configured")
    last: Exception | None = None
    for ep in endpoints:
        try:
            model = _model_for(ep, temperature).bind_tools(tools)
            reply = model.invoke(messages)
        except Exception as e:  # noqa: BLE001 - classify, then try the next endpoint
            last = e
            continue
        if reply.tool_calls or (reply.content or "").strip():
            return reply
        # Neither prose nor a tool call: same empty-answer failure mode chat() guards.
        last = LLMUnavailable(f"{ep['provider']} returned an empty reply for {ep['model']}")
    raise LLMError(f"every chat endpoint failed during tool calling (last: {last})")


def _chat_endpoint(system: str, user: str, temperature: float, endpoint: dict) -> str:
    """Invoke one OpenAI-compatible endpoint (Llama API or Groq) via LangChain."""
    llm = _model_for(endpoint, temperature)
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    try:
        reply = llm.invoke(messages)
    except Exception as e:  # noqa: BLE001 - classify below, then re-raise as ours
        name = type(e).__name__
        model = endpoint["model"]
        # Auth failures don't clear on retry, but with more than one endpoint we
        # still want to try the others - so surface as "unavailable", not fatal.
        if name in ("AuthenticationError", "PermissionDeniedError"):
            raise LLMUnavailable(f"{endpoint['provider']} rejected the API key "
                                 f"(check the key in .env) for {model}: {e}") from e
        if name in ("RateLimitError",):
            raise LLMUnavailable(f"{endpoint['provider']} rate limit on {model}") from e
        raise LLMUnavailable(f"{endpoint['provider']} chat failed ({model}): {e}") from e
    text = reply.content
    if isinstance(text, list):   # some providers return content parts
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                       for part in text)
    text = (text or "").strip()
    if not text:
        # An empty answer is a failure, not an answer. A reasoning model that
        # spends its whole token budget thinking returns "" with a 200 OK, and
        # without this the caller would render a blank brief instead of failing
        # over to an endpoint that can actually answer.
        reason = (reply.response_metadata or {}).get("finish_reason", "unknown")
        raise LLMUnavailable(
            f"{endpoint['provider']} returned an empty answer for "
            f"{endpoint['model']} (finish_reason={reason}); "
            "raise max_tokens for this endpoint if it is a reasoning model")
    return text


# --- Ollama (optional local, offline fallback) ------------------------------

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
        f"Ollama chat failed ({settings.chat_model!r}) after 2 tries: {last}. "
        f"Is 'ollama serve' up and the model pulled? `ollama pull {settings.chat_model}`")
