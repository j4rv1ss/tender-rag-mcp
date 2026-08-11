"""SQLAlchemy engine, session, and Base."""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,          # drop dead connections rather than erroring mid-request
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False,
                            future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """A scoped session per unit of work (used by scripts/).

    The MCP server has its own wrapper in app.mcp_server._db, which adds the
    domain-error -> ToolError mapping on top of this.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
