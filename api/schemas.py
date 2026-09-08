"""api/schemas.py — Pydantic request/response models for the API."""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel


class UploadResponse(BaseModel):
    id: str
    filename: str
    status: str


class StatusResponse(BaseModel):
    id: str
    status: str
    error: str | None = None


class ContractListItem(BaseModel):
    id: str
    filename: str
    status: str
    risk_score: float | None = None
    risk_level: str | None = None
    created_at: dt.datetime


class EntityOut(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float


class ClauseOut(BaseModel):
    clause_type: str
    present: bool
    confidence: float
    evidence_spans: list[str] = []


class ResultsResponse(BaseModel):
    id: str
    filename: str
    status: str
    entities: list[EntityOut] = []
    clauses: list[ClauseOut] = []
    risk_score: float | None = None
    risk_level: str | None = None
    created_at: dt.datetime
    updated_at: dt.datetime


class RiskReportResponse(BaseModel):
    id: str
    filename: str
    risk_score: float
    risk_level: str
    flagged_clauses: list[ClauseOut]
    protective_clauses: list[ClauseOut]


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SemanticSearchResult(BaseModel):
    id: str
    filename: str
    score: float
    risk_level: str | None = None


class ClauseSearchRequest(BaseModel):
    clause_type: str
    min_confidence: float = 0.5


class ClauseSearchResult(BaseModel):
    id: str
    filename: str
    confidence: float
    evidence_spans: list[str] = []
