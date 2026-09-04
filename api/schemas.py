"""api/schemas.py — Pydantic request/response models for the API."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


# ============================================================
# UPLOAD
# ============================================================

class UploadResponse(BaseModel):
    id: str
    filename: str
    status: str


# ============================================================
# STATUS
# ============================================================

class StatusResponse(BaseModel):
    id: str
    status: str
    error: str | None = None


# ============================================================
# CONTRACT LIST
# ============================================================

class ContractListItem(BaseModel):
    id: str
    filename: str
    status: str

    risk_score: float | None = None
    risk_level: str | None = None

    created_at: dt.datetime


# ============================================================
# ENTITY
# ============================================================

class EntityOut(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float


# ============================================================
# CLAUSE
# ============================================================

class ClauseOut(BaseModel):
    """
    Single detected clause.

    `text` contains the actual clause/paragraph detected
    by the classifier.
    """

    clause_type: str

    present: bool

    confidence: float

    # Actual clause text
    text: str = ""

    # Evidence extracted by classifier
    evidence_spans: list[str] = Field(default_factory=list)


# ============================================================
# CONTRACT RESULTS
# ============================================================

class ResultsResponse(BaseModel):
    id: str
    filename: str
    status: str

    entities: list[EntityOut] = Field(
        default_factory=list
    )

    clauses: list[ClauseOut] = Field(
        default_factory=list
    )

    risk_score: float | None = None
    risk_level: str | None = None

    created_at: dt.datetime
    updated_at: dt.datetime


# ============================================================
# RISK REPORT
# ============================================================

class RiskReportResponse(BaseModel):
    id: str
    filename: str

    risk_score: float
    risk_level: str

    flagged_clauses: list[ClauseOut]

    protective_clauses: list[ClauseOut]


# ============================================================
# SEMANTIC SEARCH
# ============================================================

class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SemanticSearchResult(BaseModel):
    id: str
    filename: str
    score: float

    risk_level: str | None = None


# ============================================================
# CLAUSE SEARCH
# ============================================================

class ClauseSearchRequest(BaseModel):
    clause_type: str | None = None
    contract_id: str | None = None
    contract_name: str | None = None

    min_confidence: float = 0.5


class ClauseSearchResult(BaseModel):
    id: str
    filename: str

    clause_type: str | None = None

    confidence: float

    # Actual clause text
    text: str = ""

    evidence_spans: list[str] = Field(
        default_factory=list
    )