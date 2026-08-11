"""
Self-test for the MCP server: connects exactly as a real client would and
exercises the handshake, every listing, and the tool calls.

Local (stdio — spawns app.mcp_server as a child process):
    .venv\\Scripts\\python scripts\\test_mcp.py           # fast checks only
    .venv\\Scripts\\python scripts\\test_mcp.py --rag     # also run real RAG queries
                                                          # (slow, uses LLM credits)

Hosted (streamable HTTP — verifies a deployed service):
    .venv\\Scripts\\python scripts\\test_mcp.py --url https://<service>.onrender.com/mcp
    ... --url <url> --token <MCP_AUTH_TOKEN>              # if auth is enabled

Exit code 0 = everything passed. Run it after changing app/ or .env, and after
every deploy.
"""
from __future__ import annotations

import asyncio
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

failures: list[str] = []


def arg(flag: str) -> str | None:
    """Value following --flag, or None."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


@asynccontextmanager
async def open_session(url: str | None, token: str | None):
    """One session, either transport — the checks below don't care which."""
    if url:
        import httpx2
        from mcp.client.streamable_http import streamable_http_client

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx2.AsyncClient(headers=headers, timeout=120.0) as hc:
            async with streamable_http_client(url, http_client=hc) as streams:
                async with ClientSession(streams[0], streams[1]) as s:
                    yield s
    else:
        params = StdioServerParameters(
            command=str(PYTHON), args=["-m", "app.mcp_server"], cwd=str(ROOT))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as s:
                yield s


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{f' — {detail}' if detail else ''}")
    if not ok:
        failures.append(label)


async def run(with_rag: bool, url: str | None = None,
              token: str | None = None) -> None:
    async with open_session(url, token) as s:
        print(f"\n== handshake ({'HTTP ' + url if url else 'stdio'}) ==")
        init = await s.initialize()
        check("server starts and initializes",
              init.server_info.name == "tender-rag",
              f"{init.server_info.name} v{init.server_info.version}, "
              f"protocol {init.protocol_version}")

        print("\n== discovery ==")
        names = {t.name for t in (await s.list_tools()).tools}
        expected = {"ask_tender", "ask_all_tenders", "summarize_tender",
                    "list_tenders", "get_tender", "ingest_tender",
                    "fetch_tender", "ingest_all_tenders", "health_check"}
        check("all 9 tools exposed", names == expected,
              f"missing={expected - names or '-'} extra={names - expected or '-'}")
        check("every tool documented",
              all(t.description for t in (await s.list_tools()).tools))
        tpl = [t.uri_template for t in (await s.list_resource_templates()).resource_templates]
        check("resources + prompt exposed",
              bool((await s.list_resources()).resources) and bool(tpl)
              and bool((await s.list_prompts()).prompts))

        print("\n== dependencies ==")
        r = await s.call_tool("health_check", {})
        text = r.content[0].text if r.content else ""
        check("health_check runs", not r.is_error)
        for dep in ("postgres", "pgvector"):
            check(f"{dep} reachable", f'"{dep}": "ok"' in text,
                  "" if f'"{dep}": "ok"' in text else "see health_check output")
        check("embeddings load", '"embed_model": "ok"' in text or
              '"embed_provider": "ollama"' in text)

        print("\n== catalogue ==")
        r = await s.call_tool("list_tenders", {"limit": 3})
        check("list_tenders returns rows", not r.is_error and bool(r.content))
        first = None
        if not r.is_error and r.content:
            for line in r.content[0].text.splitlines():
                if line.startswith("- **"):
                    first = line.split("**")[1]
                    break
        check("at least one tender loaded", first is not None,
              first or "corpus is empty — run scripts/ingest_all.py")

        if first:
            src, _, tid = first.partition("/")
            r = await s.call_tool("get_tender", {"source": src, "tender_id": tid})
            check("get_tender returns that tender", not r.is_error)
            check("get_tender returns structured data",
                  r.structured_content is not None)
            rr = await s.read_resource(f"tender://{src}/{tid}")
            check("resource read works", bool(rr.contents))

        print("\n== error handling ==")
        r = await s.call_tool("get_tender", {"tender_id": "__does_not_exist__"})
        msg = r.content[0].text if r.content else ""
        check("missing tender -> readable error",
              bool(r.is_error) and "not loaded" in msg, msg[:60])
        r = await s.call_tool("ask_tender", {"question": "  ", "tender_id": "x"})
        check("empty question rejected", bool(r.is_error))

        if with_rag and first:
            print("\n== live RAG (slow) ==")
            src, _, tid = first.partition("/")
            r = await s.call_tool("ask_tender", {
                "question": "What is the closing date?",
                "source": src, "tender_id": tid, "auto_fetch": False})
            body = r.content[0].text if r.content else ""
            check("ask_tender answers", not r.is_error, body[:80].replace("\n", " "))
            check("answer cites its sources", "**Sources**" in body)

            r = await s.call_tool("ask_all_tenders",
                                  {"question": "Which tenders are open?"})
            check("ask_all_tenders answers", not r.is_error)
        elif not with_rag:
            print("\n(skipping live RAG queries — pass --rag to include them)")


def main() -> int:
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    url, token = arg("--url"), arg("--token")
    # Only the stdio path spawns the local interpreter; HTTP talks to a remote.
    if not url and not PYTHON.exists():
        print(f"venv python not found at {PYTHON}")
        return 2
    asyncio.run(run("--rag" in sys.argv, url, token))
    print("\n" + ("=" * 50))
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
