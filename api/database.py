"""
api/database.py
=================
Async SQLAlchemy engine + session factory, backed by SQLite (aiosqlite driver).
Standing in for the Postgres+asyncpg stack in phase_03_tasks.md — same ORM
layer (SQLAlchemy async), swap DATABASE_URL for Phase 4 if a real Postgres
instance is available.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from api.config import DATABASE_URL

engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables if they don't exist yet. Called on app startup."""
    from api import models  # noqa: F401  (ensure models are registered on Base)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """FastAPI dependency — yields a DB session per request."""
    async with SessionLocal() as session:
        yield session
