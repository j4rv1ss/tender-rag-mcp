"""GET /health — thin wrapper over the shared health service.

The probe logic lives in app.services.health so this route and the MCP
`health_check` tool report from one implementation and cannot drift apart.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.services import health

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    return health.check()
