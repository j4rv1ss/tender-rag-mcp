"""
An MCP *client* that talks to this project's own MCP server, in-process.

This is the hop that makes the agent layer real: the web path does not import
rag.py and call it, it discovers and invokes MCP tools exactly as Claude Desktop
would. Same protocol, same handshake, same schemas, same tool descriptions —
only the transport differs, and it differs for a good reason:

    stdio       client spawns a process, talks over pipes
    HTTP        client posts JSON-RPC over the network
    in-memory   both ends share an anyio stream pair  <- this module

Spawning a subprocess per request, or looping back over localhost HTTP, would
add process and socket overhead for no benefit: the server object already lives
in this process. create_client_server_memory_streams() keeps the whole JSON-RPC
protocol and skips only the socket.

A session is opened per call rather than held open. That sounds wasteful and
isn't: there is no process to spawn and no connection to negotiate, so a
handshake over memory streams costs microseconds against a multi-second answer.
It also avoids tying an anyio task group's lifetime to a request scope, which is
where long-lived in-process sessions usually go wrong.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import anyio
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

log = logging.getLogger("tender_rag.mcp_client")


@asynccontextmanager
async def session():
    """One initialized MCP session against the in-process server."""
    from app.mcp_server import mcp

    low = mcp._lowlevel_server
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        c_read, c_write = client_streams
        s_read, s_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(low.run, s_read, s_write,
                          low.create_initialization_options())
            async with ClientSession(c_read, c_write) as s:
                await s.initialize()
                yield s
            # The server task runs until cancelled; the session is done with it.
            tg.cancel_scope.cancel()


class InProcessMCP:
    """Discovers the server's tools once, then invokes them per call."""

    def __init__(self) -> None:
        self._tools: list[dict[str, Any]] | None = None
        self._readonly: set[str] = set()

    @property
    def ready(self) -> bool:
        return self._tools is not None

    async def discover(self) -> None:
        """Handshake once and cache the tool schemas + their safety hints."""
        if self._tools is not None:
            return
        async with session() as s:
            found = (await s.list_tools()).tools
        self._tools = [self._to_function_schema(t) for t in found]
        self._readonly = {t.name for t in found
                          if t.annotations and t.annotations.read_only_hint}
        log.info("MCP tools discovered in-process: %d (%d read-only)",
                 len(self._tools), len(self._readonly))

    async def tools(self, read_only: bool = True) -> list[dict[str, Any]]:
        """Tool schemas in OpenAI function shape, as discovered over the protocol.

        read_only filters on the server's own readOnlyHint annotation. That hint
        exists precisely so a caller can decide what is safe to invoke without a
        human in the loop — a web visitor should be able to ask questions, not
        trigger a portal scrape or a full re-index.
        """
        await self.discover()
        if not read_only:
            return list(self._tools or [])
        return [t for t in (self._tools or [])
                if t["function"]["name"] in self._readonly]

    @staticmethod
    def _to_function_schema(tool: Any) -> dict[str, Any]:
        """MCP tool -> the function shape OpenAI-compatible models expect.

        The description is the tool's docstring. Passing it through unchanged is
        the point: the same text that guides Claude guides this agent.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": (tool.description or "").strip(),
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Invoke a tool. Returns (text, is_error).

        A tool error is returned, not raised: the model needs to read the failure
        and decide what to do next, which is the whole point of tool errors being
        a result type in MCP rather than a transport exception.
        """
        async with session() as s:
            result = await s.call_tool(name, arguments)
        text = "\n".join(c.text for c in (result.content or [])
                         if getattr(c, "text", None))
        return text or "(the tool returned no content)", bool(result.is_error)


mcp_client = InProcessMCP()
