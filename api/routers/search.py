"""api/routers/search.py — semantic (vector) search and structured clause search."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import CLAUSE_LABELS_PATH
from api.database import get_session
from api.models import Contract
from api.schemas import (
    ClauseSearchRequest,
    ClauseSearchResult,
    SemanticSearchRequest,
    SemanticSearchResult,
)

router = APIRouter(prefix="/v1/search", tags=["search"])


@router.get("/clause-types", response_model=list[str])
async def clause_types():
    """All known clause type labels — powers the UI's clause-search dropdown."""
    return json.loads(CLAUSE_LABELS_PATH.read_text())


@router.post("/semantic", response_model=list[SemanticSearchResult])
async def semantic_search(req: SemanticSearchRequest, session: AsyncSession = Depends(get_session)):
    """Embeds the query and finds the most similar contracts via FAISS."""
    from embeddings.embedder import embed_text
    from embeddings.vector_store import search as vs_search

    query_vector = embed_text(req.query)
    hits = vs_search(query_vector, top_k=req.top_k)

    results = []
    for contract_id, score in hits:
        result = await session.execute(select(Contract).where(Contract.id == contract_id))
        contract = result.scalar_one_or_none()
        if contract is None:
            continue
        results.append(SemanticSearchResult(
            id=contract.id, filename=contract.filename, score=score, risk_level=contract.risk_level
        ))
    return results


@router.post("/clause", response_model=list[ClauseSearchResult])
async def clause_search(req: ClauseSearchRequest, session: AsyncSession = Depends(get_session)):
    """Finds all processed contracts containing a given clause type above
    a confidence threshold. Simple in-memory filter over the DB (fine at
    the scale of a local demo — swap for a proper indexed query if this
    ever needs to run over thousands of contracts)."""
    result = await session.execute(
        select(Contract).where(Contract.status == "complete", Contract.clauses_json.is_not(None))
    )
    contracts = result.scalars().all()

    matches = []
    for contract in contracts:
        clauses = json.loads(contract.clauses_json)
        for c in clauses:
            if c["clause_type"] == req.clause_type and c["present"] and c["confidence"] >= req.min_confidence:
                matches.append(ClauseSearchResult(
                    id=contract.id, filename=contract.filename,
                    confidence=c["confidence"], evidence_spans=c["evidence_spans"],
                ))
                break
    return sorted(matches, key=lambda m: m.confidence, reverse=True)
