"""api/routers/risk.py — risk report endpoint."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import HIGH_RISK_CLAUSES, PROTECTIVE_CLAUSES
from api.database import get_session
from api.models import Contract
from api.schemas import ClauseOut, RiskReportResponse

router = APIRouter(prefix="/v1/contracts", tags=["risk"])


@router.get("/{contract_id}/risk-report", response_model=RiskReportResponse)
async def risk_report(contract_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Contract).where(Contract.id == contract_id))
    contract = result.scalar_one_or_none()
    if contract is None:
        raise HTTPException(404, f"Contract {contract_id} not found")
    if contract.status != "complete":
        raise HTTPException(409, f"Contract is not yet processed (status={contract.status})")

    clauses = [ClauseOut(**c) for c in json.loads(contract.clauses_json)]
    flagged     = [c for c in clauses if c.present and c.clause_type in HIGH_RISK_CLAUSES]
    protective  = [c for c in clauses if c.present and c.clause_type in PROTECTIVE_CLAUSES]

    return RiskReportResponse(
        id=contract.id, filename=contract.filename,
        risk_score=contract.risk_score, risk_level=contract.risk_level,
        flagged_clauses=sorted(flagged, key=lambda c: c.confidence, reverse=True),
        protective_clauses=sorted(protective, key=lambda c: c.confidence, reverse=True),
    )
