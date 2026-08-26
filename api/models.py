"""
api/models.py
==============
SQLAlchemy ORM model for a processed contract.

Entities and clauses are stored as JSON text columns (SQLite has no native
JSON type before 3.45 / limited support) — parsed back into dicts by the
Pydantic schemas in api/schemas.py before returning to the client.
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base


def _new_id() -> str:
    return str(uuid.uuid4())


class Contract(Base):
    __tablename__ = "contracts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    filename: Mapped[str] = mapped_column(String(512))
    file_path: Mapped[str] = mapped_column(String(1024))

    # pending -> processing -> complete | failed
    status: Mapped[str] = mapped_column(String(32), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    entities_json: Mapped[str | None] = mapped_column(Text, nullable=True)   # list[Entity] as JSON
    clauses_json: Mapped[str | None] = mapped_column(Text, nullable=True)    # list[ClauseResult] as JSON

    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)  # LOW / MEDIUM / HIGH

    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )
