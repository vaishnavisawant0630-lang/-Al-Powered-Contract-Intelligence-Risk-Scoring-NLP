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

@router.post("/clause", response_model=list[ClauseSearchResult])
async def clause_search(
    req: ClauseSearchRequest,
    session: AsyncSession = Depends(get_session)
):
    """
    Search detected clauses by contract, clause type,
    or both.
    """

    result = await session.execute(
        select(Contract).where(
            Contract.status == "completed",
            Contract.clauses_json.is_not(None)
        )
    )

    contracts = result.scalars().all()

    requested_type = (
        req.clause_type.strip().lower()
        if req.clause_type
        else None
    )

    requested_contract_id = (
        req.contract_id.strip()
        if req.contract_id
        else None
    )

    requested_contract_name = (
        req.contract_name.strip().lower()
        if req.contract_name
        else None
    )

    matches = []

    for contract in contracts:

        # ------------------------------------------
        # CONTRACT FILTER
        # ------------------------------------------

        if requested_contract_id:

            if contract.id != requested_contract_id:
                continue

        if requested_contract_name:

            filename = (
                contract.filename or ""
            ).lower()

            if requested_contract_name not in filename:
                continue

        # ------------------------------------------
        # LOAD CLAUSES
        # ------------------------------------------

        try:

            clauses = json.loads(
                contract.clauses_json
            )

        except (json.JSONDecodeError, TypeError):

            continue

        if not isinstance(clauses, list):
            continue

        # ------------------------------------------
        # CLAUSE FILTER
        # ------------------------------------------

        for clause in clauses:

            if not isinstance(clause, dict):
                continue

            clause_type = str(
                clause.get(
                    "clause_type",
                    ""
                )
            ).strip().lower()

            present = bool(
                clause.get(
                    "present",
                    False
                )
            )

            confidence = float(
                clause.get(
                    "confidence",
                    0.0
                )
            )

            # Clause type filter
            if requested_type:

                if clause_type != requested_type:
                    continue

            # Present filter
            if not present:
                continue

            # Confidence filter
            if confidence < req.min_confidence:
                continue

            matches.append(
                ClauseSearchResult(
                    id=contract.id,
                    filename=contract.filename,
                    confidence=confidence,
                    evidence_spans=clause.get(
                        "evidence_spans",
                        []
                    )
                )
            )

    return sorted(
        matches,
        key=lambda x: x.confidence,
        reverse=True
    )