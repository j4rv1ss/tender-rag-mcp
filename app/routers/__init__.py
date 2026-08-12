"""FastAPI routers — the browser/REST adapter over app.services.

Thin by design: every route resolves a tender and delegates to services, the
same way app.mcp_server's tools do, so HTTP and MCP cannot answer differently.
"""
