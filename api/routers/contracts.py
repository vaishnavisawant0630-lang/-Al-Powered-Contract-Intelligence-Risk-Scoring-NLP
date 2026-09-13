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


# ============================================================
# LIST CONTRACTS
# ============================================================

@router.get("", response_model=list[ContractListItem])
async def list_contracts(
    session: AsyncSession = Depends(get_session),
):
    """
    Return all uploaded contracts, newest first.
    Used by the frontend contract list.
    """

    result = await session.execute(
        select(Contract).order_by(Contract.created_at.desc())
    )

    contracts = result.scalars().all()

    return [
        ContractListItem(
            id=c.id,
            filename=c.filename,
            status=c.status,
            risk_score=c.risk_score,
            risk_level=c.risk_level,
            created_at=c.created_at,
        )
        for c in contracts
    ]


# ============================================================
# UPLOAD CONTRACT
# ============================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_contract(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    """
    Upload PDF/DOCX/TXT contract and start background processing.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {suffix}. "
                f"Allowed: {sorted(ALLOWED_SUFFIXES)}"
            ),
        )

    # Create database record first so we get an ID.
    contract = Contract(
        filename=file.filename,
        file_path="",
        status="pending",
    )

    session.add(contract)

    await session.flush()

    # Save uploaded file using contract ID.
    dest = UPLOAD_DIR / f"{contract.id}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    contract.file_path = str(dest)

    await session.commit()

    # Process asynchronously.
    background_tasks.add_task(
        process_contract,
        contract.id,
        str(dest),
    )

    return UploadResponse(
        id=contract.id,
        filename=contract.filename,
        status=contract.status,
    )


# ============================================================
# CONTRACT STATUS
# ============================================================

@router.get("/{contract_id}/status", response_model=StatusResponse)
async def get_status(
    contract_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Return processing status for a contract.
    """

    contract = await _get_or_404(
        session,
        contract_id,
    )

    return StatusResponse(
        id=contract.id,
        status=contract.status,
        error=contract.error,
    )


# ============================================================
# CONTRACT RESULTS
# ============================================================

@router.get("/{contract_id}/results", response_model=ResultsResponse)
async def get_results(
    contract_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Return complete analysis results for ONE contract.

    Important:
    Only clauses with present=True are returned.
    """

    contract = await _get_or_404(
        session,
        contract_id,
    )

    # --------------------------------------------------------
    # ENTITIES
    # --------------------------------------------------------

    entities = []

    if contract.entities_json:
        try:
            raw_entities = json.loads(contract.entities_json)

            if isinstance(raw_entities, list):
                for entity in raw_entities:

                    if not isinstance(entity, dict):
                        continue

                    try:
                        entities.append(
                            EntityOut(**entity)
                        )
                    except Exception:
                        # Ignore malformed old entity records.
                        continue

        except (json.JSONDecodeError, TypeError):
            entities = []

    # --------------------------------------------------------
    # CLAUSES
    # --------------------------------------------------------

    clauses = []

    if contract.clauses_json:
        try:
            raw_clauses = json.loads(contract.clauses_json)

            if isinstance(raw_clauses, list):

                for clause in raw_clauses:

                    if not isinstance(clause, dict):
                        continue

                    # ------------------------------------------------
                    # IMPORTANT:
                    # Do NOT return absent classifier labels.
                    # ------------------------------------------------
                    present = clause.get("present", False)

                    if not present:
                        continue

                    # ------------------------------------------------
                    # Normalize clause fields.
                    # ------------------------------------------------

                    clause_type = (
                        clause.get("clause_type")
                        or clause.get("type")
                        or "Unknown"
                    )

                    confidence = clause.get(
                        "confidence",
                        0.0,
                    )

                    text = (
                        clause.get("text")
                        or clause.get("clause_text")
                        or ""
                    )

                    evidence_spans = clause.get(
                        "evidence_spans",
                        [],
                    )

                    if evidence_spans is None:
                        evidence_spans = []

                    if not isinstance(evidence_spans, list):
                        evidence_spans = [str(evidence_spans)]

                    # ------------------------------------------------
                    # If text is missing, use evidence span.
                    # ------------------------------------------------

                    if not text.strip() and evidence_spans:
                        text = " ".join(
                            str(span).strip()
                            for span in evidence_spans
                            if str(span).strip()
                        )

                    normalized_clause = {
                        "clause_type": str(clause_type),
                        "present": True,
                        "confidence": float(confidence),
                        "text": str(text),
                        "evidence_spans": evidence_spans,
                    }

                    try:
                        clauses.append(
                            ClauseOut(**normalized_clause)
                        )
                    except Exception:
                        # Don't allow one malformed clause
                        # to destroy the entire API response.
                        continue

        except (json.JSONDecodeError, TypeError):
            clauses = []

    # --------------------------------------------------------
    # RETURN RESULTS FOR THIS CONTRACT ONLY
    # --------------------------------------------------------

    return ResultsResponse(
        id=contract.id,
        filename=contract.filename,
        status=contract.status,
        entities=entities,
        clauses=clauses,
        risk_score=contract.risk_score,
        risk_level=contract.risk_level,
        created_at=contract.created_at,
        updated_at=contract.updated_at,
    )


# ============================================================
# GET CONTRACT
# ============================================================

@router.get("/{contract_id}", response_model=StatusResponse)
async def get_contract(
    contract_id: str,
    session: AsyncSession = Depends(get_session),
):
    return await get_status(
        contract_id,
        session,
    )


# ============================================================
# DELETE CONTRACT
# ============================================================

@router.delete("/{contract_id}")
async def delete_contract(
    contract_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Delete contract database record and uploaded file.
    """

    contract = await _get_or_404(
        session,
        contract_id,
    )

    if contract.file_path:
        file_path = Path(contract.file_path)

        if file_path.exists():
            file_path.unlink()

    await session.delete(contract)
    await session.commit()

    return {
        "deleted": contract_id
    }


# ============================================================
# HELPER
# ============================================================

async def _get_or_404(
    session: AsyncSession,
    contract_id: str,
) -> Contract:

    result = await session.execute(
        select(Contract).where(
            Contract.id == contract_id
        )
    )

    contract = result.scalar_one_or_none()

    if contract is None:
        raise HTTPException(
            status_code=404,
            detail=f"Contract {contract_id} not found",
        )

    return contract