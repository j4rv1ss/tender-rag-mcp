"""
The AI/agent layer that sits between FastAPI and the MCP server.

Flow:  user question -> model picks an MCP tool -> tool runs -> model writes
the reply from what the tool returned.

Why route the web path through MCP instead of calling rag.py directly:

  * one canonical interface. The tools, their schemas and their descriptions
    are defined once and consumed identically by Claude and by this agent, so
    the browser cannot drift from what an assistant sees.
  * the model chooses. A hardcoded route must be told whether a question is
    about one tender or the whole corpus; the agent reads the question and
    picks ask_tender vs ask_all_tenders vs summarize_tender itself, and can
    chain calls for a follow-up.

What it costs, honestly: one extra model round-trip per question (choose, then
answer), so latency and token spend roughly double against the direct path.
AGENT_MODE=false restores the direct call for comparison.

Safety: only tools the server marks readOnlyHint are offered. A web visitor can
ask anything and trigger nothing expensive — no scraping, no re-indexing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import anyio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.config import settings
from app.services import llm
from app.services.mcp_client import mcp_client

log = logging.getLogger("tender_rag.agent")

SYSTEM = (
    "You are a procurement assistant with tools over a corpus of scraped public "
    "tenders. Answer the user by CALLING A TOOL — never from your own knowledge, "
    "which does not include these tenders.\n"
    "Choosing:\n"
    "- The user names or implies ONE tender -> ask_tender (needs tender_id, and "
    "source when known).\n"
    "- No specific tender -> ask_all_tenders.\n"
    "- They want an overview of one tender rather than a single fact -> "
    "summarize_tender.\n"
    "- They ask what exists, or you need a tender_id -> list_tenders first.\n"
    "Then reply with what the tool returned, preserving its Sources line verbatim "
    "so the user can verify every claim. If a tool reports an error, explain it "
    "plainly; do not invent an answer instead."
)


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    turns: int = 0


def _args_of(call) -> dict:
    """LangChain gives dict args; some providers hand back a JSON string."""
    raw = call["args"] if isinstance(call, dict) else getattr(call, "args", {})
    if isinstance(raw, str):
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
    return raw or {}


async def answer(question: str, source: str | None = None,
                 tender_id: str | None = None) -> AgentResult:
    """Run the tool-calling loop and return the final prose answer."""
    tools = await mcp_client.tools(read_only=not settings.agent_allow_writes)
    if not tools:
        raise RuntimeError("no tools available to the agent")

    # Hand the model any scoping the caller already knows, rather than making it
    # guess or ask — the web form has explicit source/tender fields.
    hint = ""
    if tender_id:
        hint = (f"\n\n(The user is asking about tender_id={tender_id!r}"
                f"{f', source={source!r}' if source else ''}.)")
    messages: list = [SystemMessage(content=SYSTEM),
                      HumanMessage(content=question.strip() + hint)]

    used: list[dict] = []
    for turn in range(1, settings.agent_max_turns + 1):
        reply: AIMessage = await anyio.to_thread.run_sync(
            lambda: llm.chat_tools(messages, tools))
        messages.append(reply)

        if not reply.tool_calls:
            text = (reply.content or "").strip()
            if not text:
                raise RuntimeError("the model returned neither an answer nor a tool call")
            return AgentResult(answer=text, tool_calls=used, turns=turn)

        for call in reply.tool_calls:
            name = call["name"] if isinstance(call, dict) else call.name
            args = _args_of(call)
            call_id = (call.get("id") if isinstance(call, dict) else call.id) or name
            log.info("agent turn %d -> %s(%s)", turn, name, ", ".join(args))
            text, is_error = await mcp_client.call(name, args)
            used.append({"tool": name, "arguments": args, "error": is_error})
            messages.append(ToolMessage(content=text, tool_call_id=call_id,
                                        name=name))

    # Out of turns: return the last tool output rather than nothing, since it is
    # the grounded material the user actually wanted.
    last = next((m.content for m in reversed(messages)
                 if isinstance(m, ToolMessage)), "")
    return AgentResult(
        answer=last or "The assistant could not settle on an answer.",
        tool_calls=used, turns=settings.agent_max_turns)
