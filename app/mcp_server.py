"""
MCP server (stdio) — exposes the tender RAG corpus to MCP clients.

This replaces the old FastAPI/uvicorn HTTP layer. Every tool is a thin adapter
over app.services, exactly as the routers used to be, so the RAG pipeline
(embed -> hybrid retrieve -> rerank -> grounded LLM answer) is unchanged.

Endpoint -> tool mapping from the previous REST API:
    POST /chat   (with tender_id)  -> ask_tender
    POST /chat   (no tender_id)    -> ask_all_tenders
    POST /summary                  -> summarize_tender
    GET  /tenders                  -> list_tenders
    GET  /tenders/{id}             -> get_tender
    POST /ingest                   -> ingest_tender
    POST /fetch                    -> fetch_tender
    POST /ingest-all               -> ingest_all_tenders
    GET  /health                   -> health_check

Two transports:
    python -m app.mcp_server            stdio  — a local client owns the process
    python -m app.mcp_server --http     hosted — public endpoint; open unless
                                        MCP_AUTH_TOKEN is set (then bearer required)

--http also serves a browser chat page at / (see _http_app): the same RAG pipeline
over POST /api/chat, for people without an MCP client.
"""
from __future__ import annotations

import logging
import os
import secrets
import sys
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.types import ToolAnnotations
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import Chunk, Document, Tender
from app.schemas import ChatResponse, IngestResponse, TenderOut
from app.services import health, ingest_service, normalize, rag
from app.services.embeddings import EmbeddingError
from app.services.llm import LLMError
from app.services.scrape import ScrapeError

# stdio speaks JSON-RPC over stdout, so logs MUST go to stderr.
logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
log = logging.getLogger("tender_rag.mcp")


# --------------------------------------------------------------------------- #
# session + error plumbing (the MCP equivalent of get_db + HTTPException)
# --------------------------------------------------------------------------- #
@contextmanager
def _db() -> Iterator[Session]:
    """A session per tool call, with domain errors mapped to ToolError.

    ToolError reaches the client as an error result it can read and retry from,
    which is the MCP analogue of the routers' HTTPException mapping.
    """
    db = SessionLocal()
    try:
        yield db
    except EmbeddingError as e:
        raise ToolError(f"embedding backend unavailable: {e}") from e
    except LLMError as e:
        raise ToolError(f"chat model unavailable: {e}") from e
    except ScrapeError as e:
        raise ToolError(f"scrape failed: {e}") from e
    except normalize.TenderNotFound as e:
        raise ToolError(str(e)) from e
    finally:
        db.close()


def _load_tender(db: Session, source: str, tender_id: str,
                 auto_fetch: bool) -> Tender:
    """Fetch/ensure a tender or raise a ToolError explaining why it's missing."""
    tender = ingest_service.ensure_tender(db, source, tender_id,
                                          allow_scrape=auto_fetch)
    if tender is None:
        reason = ("this server is query-only (scraping disabled), so only pre-loaded "
                  "tenders can be answered" if not settings.enable_scraping else
                  "unknown source, or auto_fetch off")
        raise ToolError(
            f"tender {source}/{tender_id} is not loaded ({reason}). Load it with "
            "ingest_tender/fetch_tender, or use ask_all_tenders to search every "
            "loaded tender instead.")
    return tender


def _format_answer(res: ChatResponse) -> str:
    """Render a ChatResponse as markdown — what the client actually displays."""
    out = [res.answer.strip()]
    if res.references:
        out.append(f"\n---\n**Sources** ({res.chunks_used} chunks retrieved)")
        for i, ref in enumerate(res.references, 1):
            where = " · ".join(p for p in (
                ref.tender, ref.document,
                f"p.{ref.page}" if ref.page else None) if p)
            snippet = " ".join(ref.snippet.split())
            out.append(f"{i}. {where or 'unknown source'} "
                       f"(score {ref.score:.3f})\n   > {snippet}")
    return "\n".join(out)


def _to_out(t: Tender, doc_count: int | None = None) -> TenderOut:
    return TenderOut(
        source=t.source, tender_id=t.tender_id, tender_number=t.tender_number,
        title=t.title, organization=t.organization, description=t.description,
        category=t.category, country=t.country, issue_date=t.issue_date,
        closing_date=t.closing_date,
        estimated_value_amount=float(t.estimated_value_amount)
        if t.estimated_value_amount is not None else None,
        estimated_value_currency=t.estimated_value_currency,
        source_website=t.source_website, tender_url=t.tender_url, status=t.status,
        document_count=doc_count)


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def _lifespan(server: MCPServer) -> AsyncIterator[None]:
    log.info("Tender RAG MCP up | db=%s | chat=%s(%s) | embed=%s(%d)",
             settings.database_url_safe, settings.active_chat_model,
             settings.chat_provider, settings.active_embed_model, settings.embed_dim)

    # Warm models in the background: an MCP client spawns us and expects to
    # initialize immediately, so loading fastembed/reranker inline would risk a
    # client-side startup timeout.
    def _warm() -> None:
        try:
            health.warmup()
            log.info("embedding + reranker + chat warmed up")
        except Exception as e:  # noqa: BLE001 - non-fatal
            log.warning("warmup skipped: %s", e)

    threading.Thread(target=_warm, name="warmup", daemon=True).start()
    yield


mcp = MCPServer(
    name="tender-rag",
    title="Tender RAG",
    version="0.2.0",
    instructions=(
        "Grounded question answering over scraped public-procurement tenders "
        "(RAG on PostgreSQL + pgvector).\n"
        "Pick a tool by scope: ask_tender when the user names one tender, "
        "ask_all_tenders when they don't, summarize_tender for a whole-tender "
        "brief. Use list_tenders first to discover what is loaded and to find a "
        "tender_id. Answers are grounded in the indexed documents and cite their "
        "sources — do not supplement them with outside knowledge about a tender."
    ),
    lifespan=_lifespan,
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False)
# Tools that return ready-to-read markdown opt out of structured output: the SDK
# would otherwise repeat the entire answer in structured_content as {"result": …},
# doubling the payload the client has to carry for no gain.
_TEXT = {"annotations": _READ_ONLY, "structured_output": False}
# Ingest replaces that tender's own rows; re-running is safe, so idempotent
# rather than destructive. open_world=True where a portal is scraped live.
_INGEST = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                          idempotent_hint=True)
_SCRAPE = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                          idempotent_hint=True, open_world_hint=True)


# --------------------------------------------------------------------------- #
# tools — question answering
# --------------------------------------------------------------------------- #
@mcp.tool(**_TEXT)
def ask_tender(question: str, tender_id: str, source: str = "etenders",
               top_k: int | None = None, auto_fetch: bool = True) -> str:
    """Answer a question about ONE specific tender, grounded in its documents.

    Use this whenever the user names or implies a single tender (e.g. "what
    documents does tender 162660 require?"). Returns the answer followed by the
    document/page sources it was drawn from.

    Args:
        question: The natural-language question to answer.
        tender_id: Portal id of the tender, e.g. "162660" or "RW10414443-26".
        source: Portal the tender came from, e.g. "etenders", "randwater",
            "ppadb", "transnet", "sadc", "zppa", "capetown", "nra", "cpbn".
        top_k: How many retrieved chunks to ground the answer in. Defaults to
            the server's configured value.
        auto_fetch: If the tender isn't loaded yet, scrape it on demand first.
    """
    if not question.strip():
        raise ToolError("question is empty")
    with _db() as db:
        tender = _load_tender(db, source, tender_id, auto_fetch)
        return _format_answer(rag.answer_question(db, tender, question, top_k))


@mcp.tool(**_TEXT)
def ask_all_tenders(question: str, top_k: int | None = None) -> str:
    """Answer a question across EVERY loaded tender, attributing each finding.

    Use this for cross-corpus questions where no single tender is named — e.g.
    "which tenders close in March?" or "who is procuring water infrastructure?".
    Each source in the answer is labelled with the tender it came from.

    Args:
        question: The natural-language question to answer.
        top_k: How many retrieved chunks to ground the answer in. Defaults to
            the server's configured value.
    """
    if not question.strip():
        raise ToolError("question is empty")
    with _db() as db:
        return _format_answer(rag.answer_across_corpus(db, question, top_k))


@mcp.tool(**_TEXT)
def summarize_tender(tender_id: str, source: str = "etenders",
                     auto_fetch: bool = True) -> str:
    """Produce a grounded brief of one tender: scope, dates, fees, eligibility,
    required documents, evaluation method, contract period and contacts.

    Prefer this over ask_tender when the user wants an overview of a tender
    rather than one specific fact.

    Args:
        tender_id: Portal id of the tender, e.g. "PR/PPADB/055".
        source: Portal the tender came from, e.g. "ppadb".
        auto_fetch: If the tender isn't loaded yet, scrape it on demand first.
    """
    with _db() as db:
        tender = _load_tender(db, source, tender_id, auto_fetch)
        return _format_answer(rag.summarize(db, tender))


# --------------------------------------------------------------------------- #
# tools — catalogue
# --------------------------------------------------------------------------- #
@mcp.tool(**_TEXT)
def list_tenders(query: str | None = None, limit: int = 50) -> str:
    """List the tenders currently loaded, newest first.

    Call this first to discover what the corpus holds and to find the exact
    source + tender_id pair the other tools need.

    Args:
        query: Optional case-insensitive filter on title, organization or
            tender number.
        limit: Maximum rows to return (1-500). Kept small by default because
            the full catalogue can be long.
    """
    limit = max(1, min(limit, 500))
    stmt = select(Tender).order_by(Tender.created_at.desc())
    if query and query.strip():
        like = f"%{query.strip()}%"
        stmt = stmt.where(or_(Tender.title.ilike(like),
                              Tender.organization.ilike(like),
                              Tender.tender_number.ilike(like)))
    with _db() as db:
        total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
        rows = db.execute(stmt.limit(limit)).scalars().all()

    if not rows:
        return ("No tenders match." if query else
                "No tenders are loaded yet — run ingest_all_tenders, or "
                "fetch_tender for a specific one.")

    out = [f"{total} tender(s){f' matching {query!r}' if query else ''}"
           f"{f', showing {len(rows)}' if total > len(rows) else ''}:", ""]
    for t in rows:
        out.append(f"- **{t.source}/{t.tender_id}** — {t.title or '(untitled)'}")
        detail = " · ".join(p for p in (
            t.organization, t.tender_number,
            f"closes {t.closing_date}" if t.closing_date else None,
            t.status) if p)
        if detail:
            out.append(f"  {detail}")
    return "\n".join(out)


@mcp.tool(annotations=_READ_ONLY)
def get_tender(tender_id: str, source: str = "etenders") -> TenderOut:
    """Return the full stored metadata for one tender, plus its document count.

    This reads only the tender record — it does not search document text. Use
    ask_tender or summarize_tender for anything requiring the documents.

    Args:
        tender_id: Portal id of the tender.
        source: Portal the tender came from.
    """
    with _db() as db:
        t = db.execute(select(Tender).where(Tender.source == source,
                                            Tender.tender_id == tender_id)
                       ).scalar_one_or_none()
        if t is None:
            raise ToolError(f"tender {source}/{tender_id} is not loaded")
        doc_count = db.execute(
            select(func.count(Document.id)).where(Document.tender_pk == t.id)).scalar()
        return _to_out(t, doc_count)


# --------------------------------------------------------------------------- #
# tools — loading
# --------------------------------------------------------------------------- #
@mcp.tool(annotations=_INGEST)
def ingest_tender(tender_id: str, source: str = "etenders") -> IngestResponse:
    """Index an already-scraped tender from disk into the vector store.

    Requires the scraper output to exist locally. Re-running replaces that
    tender's documents and chunks. Use fetch_tender instead if the tender has
    not been scraped yet.

    Args:
        tender_id: Portal id of the tender to index.
        source: Portal the tender came from.
    """
    with _db() as db:
        return ingest_service.ingest_tender(db, source, tender_id)


@mcp.tool(annotations=_SCRAPE)
def fetch_tender(tender_id: str, source: str) -> IngestResponse:
    """Scrape a tender from its portal on demand, then index it.

    Hits the live procurement portal, so it is slow (up to several minutes) and
    needs local scraper binaries — unavailable when the server runs query-only.

    Args:
        tender_id: Portal id of the tender to scrape, e.g. "RW10414443-26".
        source: Portal to scrape from, e.g. "randwater".
    """
    with _db() as db:
        tender = ingest_service.ensure_tender(db, source, tender_id,
                                              allow_scrape=True)
        if tender is None:
            raise ToolError(f"auto-scrape is not supported for source {source!r}")
        docs = db.execute(select(func.count(Document.id))
                          .where(Document.tender_pk == tender.id)).scalar()
        chunks = db.execute(select(func.count(Chunk.id))
                            .where(Chunk.tender_pk == tender.id)).scalar()
        return IngestResponse(source=tender.source, tender_id=tender.tender_id,
                              tender_pk=tender.id, documents=docs, chunks=chunks,
                              embedded=chunks, reingested=False)


@mcp.tool(annotations=_INGEST, structured_output=False)
def ingest_all_tenders() -> str:
    """Index every scraped tender found across all scraper outputs.

    Slow — it embeds every document chunk in the corpus. Prefer ingest_tender
    for a single tender. Files that cannot be parsed are skipped and reported.
    """
    with _db() as db:
        res = ingest_service.ingest_all(db)
    totals = res["totals"]
    out = [f"Ingested {totals['tenders']} tender(s): "
           f"{totals['documents']} documents, {totals['chunks']} chunks."]
    if res["skipped"]:
        out.append(f"\nSkipped {len(res['skipped'])}:")
        out += [f"- {s}" for s in res["skipped"][:20]]
        if len(res["skipped"]) > 20:
            out.append(f"- …and {len(res['skipped']) - 20} more")
    return "\n".join(out)


@mcp.tool(annotations=_READ_ONLY)
def health_check() -> dict:
    """Report server health: PostgreSQL, pgvector, and the embedding/chat
    providers actually in use, plus the active configuration.

    Use this to diagnose why another tool is failing.
    """
    return health.check()


# --------------------------------------------------------------------------- #
# resources — the read-only GETs, for clients that browse context
# --------------------------------------------------------------------------- #
@mcp.resource("tender://catalogue", name="Tender catalogue",
              description="Every tender currently loaded.", mime_type="text/markdown")
def catalogue_resource() -> str:
    return list_tenders()


@mcp.resource("tender://{source}/{tender_id}", name="Tender record",
              description="Stored metadata for one tender.",
              mime_type="application/json")
def tender_resource(source: str, tender_id: str) -> str:
    with _db() as db:
        t = db.execute(select(Tender).where(Tender.source == source,
                                            Tender.tender_id == tender_id)
                       ).scalar_one_or_none()
        if t is None:
            raise ResourceNotFoundError(f"tender {source}/{tender_id} is not loaded")
        doc_count = db.execute(
            select(func.count(Document.id)).where(Document.tender_pk == t.id)).scalar()
        return _to_out(t, doc_count).model_dump_json(indent=2)


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
@mcp.prompt(name="bid_assessment",
            description="Assess whether a tender is worth bidding for.")
def bid_assessment(tender_id: str, source: str = "etenders") -> str:
    return (
        f"Use summarize_tender on {source}/{tender_id}, then assess it for a "
        "prospective bidder. Cover: scope of work, submission deadline and any "
        "briefing session, mandatory returnable documents, eligibility and "
        "compliance requirements, evaluation criteria, and contract period. "
        "Call out anything that would disqualify a bid. Ground every claim in "
        "the tender's own documents and cite them; if something is not stated "
        "in the documents, say so rather than assuming."
    )


# --------------------------------------------------------------------------- #
# HTTP transport (optional) — for hosting the server instead of running it local
# --------------------------------------------------------------------------- #
class BearerAuth:
    """Raw ASGI middleware enforcing a shared bearer token.

    Only wrapped around the app when MCP_AUTH_TOKEN is set; with no token the
    endpoint is served open (see _http_app).

    Deliberately not a Starlette BaseHTTPMiddleware: that buffers responses and
    would break the streaming (SSE) leg of the streamable-HTTP transport.
    """

    def __init__(self, app, token: str, exempt: frozenset[str] = frozenset()):
        self.app, self.token, self.exempt = app, token, exempt

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.exempt:
            return await self.app(scope, receive, send)

        header = dict(scope.get("headers") or {}).get(b"authorization", b"")
        value = header.decode("latin-1")
        offered = value[7:] if value[:7].lower() == "bearer " else ""
        # compare_digest keeps the check constant-time (no token-length leak).
        if not secrets.compare_digest(offered, self.token):
            log.warning("rejected unauthenticated request to %s", scope.get("path"))
            body = b'{"error":"unauthorized"}'
            await send({"type": "http.response.start", "status": 401, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b'Bearer realm="tender-rag"')]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


class _HttpError(Exception):
    """A domain failure with the HTTP status the browser chat should see."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


@contextmanager
def _web_db() -> Iterator[Session]:
    """Session per web request. The HTTP sibling of _db(): same domain errors, but
    mapped to status codes instead of ToolError (the chat page is not an MCP client).
    """
    db = SessionLocal()
    try:
        yield db
    except EmbeddingError as e:
        raise _HttpError(503, f"embedding backend unavailable: {e}") from e
    except LLMError as e:
        raise _HttpError(503, f"chat model unavailable: {e}") from e
    except ScrapeError as e:
        raise _HttpError(502, f"scrape failed: {e}") from e
    except normalize.TenderNotFound as e:
        raise _HttpError(404, str(e)) from e
    finally:
        db.close()


def _http_app(path: str, stateless: bool):
    """Build the ASGI app: MCP endpoint, the browser chat page, and /healthz."""
    from starlette.concurrency import run_in_threadpool
    from starlette.requests import Request
    from starlette.responses import FileResponse, JSONResponse
    from starlette.routing import Route

    from mcp.server.transport_security import TransportSecuritySettings

    token = settings.mcp_auth_token.strip()
    _page = Path(__file__).resolve().parent / "static" / "index.html"

    async def healthz(request: Request) -> JSONResponse:
        # Hosts need an unauthenticated liveness probe; keep it free of secrets
        # and of anything that touches the database on every ping.
        return JSONResponse({"status": "alive", "server": "tender-rag"})

    async def home(request: Request) -> FileResponse:
        return FileResponse(str(_page))

    def health_route(request: Request) -> JSONResponse:
        # Same probe the health_check tool reports, so the two can't disagree.
        return JSONResponse(health.check())

    # The DB/RAG work below is blocking (a question costs ~5s: embed -> retrieve ->
    # LLM), so it must never run on the event loop that also serves the MCP stream.
    # Starlette threadpools plain `def` endpoints; `chat` needs `await request.json()`
    # first, so it hands the blocking half off explicitly.
    def tenders(request: Request) -> JSONResponse:
        with _web_db() as db:
            rows = db.execute(select(Tender).order_by(Tender.source,
                                                      Tender.tender_id)).scalars()
            return JSONResponse([
                {"source": t.source, "tender_id": t.tender_id,
                 "tender_number": t.tender_number, "title": t.title} for t in rows])

    async def chat(request: Request) -> JSONResponse:
        body = await request.json()
        return await run_in_threadpool(_chat_sync, body)

    async def summary(request: Request) -> JSONResponse:
        # Same path as chat, with the whole-tender brief flag forced on.
        body = await request.json()
        return await run_in_threadpool(_chat_sync, {**body, "summary": True})

    async def ingest(request: Request) -> JSONResponse:
        body = await request.json()
        return await run_in_threadpool(_ingest_sync, body)

    def _ingest_sync(body: dict) -> JSONResponse:
        tender_id = (body.get("tender_id") or "").strip()
        if not tender_id:
            return JSONResponse({"detail": "tender_id is required"}, status_code=422)
        source = (body.get("source") or "").strip() or "etenders"
        with _web_db() as db:
            # Mirrors the ingest_tender tool: load from disk, scraping only if the
            # server is configured for it (cloud hosts have no scraper binaries).
            tender = ingest_service.ensure_tender(
                db, source, tender_id, allow_scrape=settings.enable_scraping)
            if tender is None:
                raise _HttpError(
                    404, f"{source}/{tender_id} is not on disk and could not be "
                         "fetched (scraping disabled or source unsupported)")
            docs = db.execute(select(func.count(Document.id))
                              .where(Document.tender_pk == tender.id)).scalar()
            chunks = db.execute(select(func.count(Chunk.id))
                                .where(Chunk.tender_pk == tender.id)).scalar()
        return JSONResponse({"source": tender.source, "tender_id": tender.tender_id,
                             "documents": docs, "chunks": chunks})

    def _chat_sync(body: dict) -> JSONResponse:
        question = (body.get("question") or "").strip()
        tender_id = (body.get("tender_id") or "").strip()
        want_summary = bool(body.get("summary"))
        if not question and not want_summary:
            return JSONResponse({"detail": "question is empty"}, status_code=422)
        with _web_db() as db:
            if tender_id:
                source = (body.get("source") or "").strip() or "etenders"
                tender = ingest_service.ensure_tender(
                    db, source, tender_id, allow_scrape=settings.enable_scraping)
                if tender is None:
                    raise _HttpError(404, f"tender {source}/{tender_id} is not loaded")
                res = (rag.summarize(db, tender) if want_summary else
                       rag.answer_question(db, tender, question, body.get("top_k")))
            else:
                res = rag.answer_across_corpus(db, question, body.get("top_k"))
        return JSONResponse(res.model_dump())

    app = mcp.streamable_http_app(
        streamable_http_path=path,
        stateless_http=stateless,
        transport_security=TransportSecuritySettings(
            allowed_hosts=settings.allowed_hosts,
            allowed_origins=settings.allowed_hosts),
    )
    async def on_http_error(request: Request, exc: _HttpError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    app.add_exception_handler(_HttpError, on_http_error)
    app.router.routes += [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/", home, methods=["GET"]),
        Route("/api/tenders", tenders, methods=["GET"]),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/summary", summary, methods=["POST"]),
        Route("/api/ingest", ingest, methods=["POST"]),
        Route("/api/health", health_route, methods=["GET"]),
    ]
    if not token:
        # Open endpoint: anyone who knows the URL can call every tool, including
        # the expensive scrape/ingest ones. Set MCP_AUTH_TOKEN to require a token.
        log.warning("MCP_AUTH_TOKEN is not set — serving %s WITHOUT authentication. "
                    "Anyone with the URL can call every tool.", path)
        return app
    # "/" is inert markup, so it loads without a token and its script then prompts
    # for one; /api/* carries corpus data and stays behind the same check as /mcp.
    return BearerAuth(app, token, exempt=frozenset({"/healthz", "/"}))


def main() -> None:
    """Entry point. stdio by default; --http serves the streamable-HTTP transport."""
    import argparse

    parser = argparse.ArgumentParser(description="Tender RAG MCP server")
    parser.add_argument("--http", action="store_true",
                        help="serve streamable HTTP instead of stdio (open unless "
                             "MCP_AUTH_TOKEN is set)")
    parser.add_argument("--host", default=settings.mcp_host)
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT") or settings.mcp_port),
                        help="defaults to $PORT when the host injects one")
    parser.add_argument("--path", default=settings.mcp_path)
    parser.add_argument("--stateful", action="store_true",
                        help="keep MCP sessions in memory (default is stateless, "
                             "so a restart or a second instance can't 404 a client)")
    args = parser.parse_args()

    if not args.http:
        mcp.run(transport="stdio")
        return

    import uvicorn
    log.info("serving MCP over HTTP on %s:%d%s | auth: %s | allowed hosts: %s",
             args.host, args.port, args.path,
             "bearer token" if settings.mcp_auth_token.strip() else "NONE (open)",
             ", ".join(settings.allowed_hosts))
    uvicorn.run(_http_app(args.path, stateless=not args.stateful),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
