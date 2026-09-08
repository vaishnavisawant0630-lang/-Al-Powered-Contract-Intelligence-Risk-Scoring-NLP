"""api/routers/contracts.py — upload, status, results, delete."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import UPLOAD_DIR
from api.database import get_session
from api.models import Contract
from api.pipeline import process_contract
from api.schemas import (
    ClauseOut,
    ContractListItem,
    EntityOut,
    ResultsResponse,
    StatusResponse,
    UploadResponse,
)

router = APIRouter(prefix="/v1/contracts", tags=["contracts"])

ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}


@router.get("", response_model=list[ContractListItem])
async def list_contracts(session: AsyncSession = Depends(get_session)):
    """All uploaded contracts, most recent first — powers the UI's contract list."""
    result = await session.execute(select(Contract).order_by(Contract.created_at.desc()))
    contracts = result.scalars().all()
    return [
        ContractListItem(
            id=c.id, filename=c.filename, status=c.status,
            risk_score=c.risk_score, risk_level=c.risk_level, created_at=c.created_at,
        )
        for c in contracts
    ]


@router.post("/upload", response_model=UploadResponse)
async def upload_contract(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {sorted(ALLOWED_SUFFIXES)}")

    contract = Contract(filename=file.filename, file_path="", status="pending")
    session.add(contract)
    await session.flush()  # populates contract.id

    dest = UPLOAD_DIR / f"{contract.id}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    contract.file_path = str(dest)
    await session.commit()

    background_tasks.add_task(process_contract, contract.id, str(dest))

    return UploadResponse(id=contract.id, filename=contract.filename, status=contract.status)


@router.get("/{contract_id}/status", response_model=StatusResponse)
async def get_status(contract_id: str, session: AsyncSession = Depends(get_session)):
    contract = await _get_or_404(session, contract_id)
    return StatusResponse(id=contract.id, status=contract.status, error=contract.error)


@router.get("/{contract_id}/results", response_model=ResultsResponse)
async def get_results(contract_id: str, session: AsyncSession = Depends(get_session)):
    contract = await _get_or_404(session, contract_id)
    entities = [EntityOut(**e) for e in json.loads(contract.entities_json)] if contract.entities_json else []
    clauses = [ClauseOut(**c) for c in json.loads(contract.clauses_json)] if contract.clauses_json else []
    return ResultsResponse(
        id=contract.id, filename=contract.filename, status=contract.status,
        entities=entities, clauses=clauses,
        risk_score=contract.risk_score, risk_level=contract.risk_level,
        created_at=contract.created_at, updated_at=contract.updated_at,
    )


@router.get("/{contract_id}", response_model=StatusResponse)
async def get_contract(contract_id: str, session: AsyncSession = Depends(get_session)):
    return await get_status(contract_id, session)


@router.delete("/{contract_id}")
async def delete_contract(contract_id: str, session: AsyncSession = Depends(get_session)):
    contract = await _get_or_404(session, contract_id)
    if contract.file_path and Path(contract.file_path).exists():
        Path(contract.file_path).unlink()
    await session.delete(contract)
    await session.commit()
    return {"deleted": contract_id}


async def _get_or_404(session: AsyncSession, contract_id: str) -> Contract:
    result = await session.execute(select(Contract).where(Contract.id == contract_id))
    contract = result.scalar_one_or_none()
    if contract is None:
        raise HTTPException(404, f"Contract {contract_id} not found")
    return contract
