"""
api/pipeline.py
=================
Orchestrator: ties together every model from Phases 1 & 2 plus the new
Phase 3 embedding/vector-search layer into one end-to-end flow.

    file on disk
        -> ingestion.DocumentRouter          (OCR / PDF / DOCX text extraction)
        -> ner.inference.extract_entities    (Phase 1 model)
        -> classification.inference.classify_clauses  (Phase 2 model)
        -> risk score (heuristic, see api.config)
        -> embeddings.embedder + vector_store (Phase 3 semantic search)
        -> persisted to the Contract row via api.database

Runs inside a FastAPI BackgroundTask (see api/routers/contracts.py) so the
upload endpoint returns immediately and processing happens after the
response is sent — standing in for the Celery worker in phase_03_tasks.md.
"""
from __future__ import annotations

import json
import logging
import re

from sqlalchemy import select

from api.config import (
    CLASSIFIER_MODEL_PATH,
    CLAUSE_LABELS_PATH,
    HIGH_RISK_CLAUSES,
    NER_MODEL_PATH,
    PROTECTIVE_CLAUSES,
    RISK_THRESHOLDS,
)
from api.database import SessionLocal
from api.models import Contract

logger = logging.getLogger(__name__)

_MIN_PARAGRAPH_CHARS = 40


def _split_into_paragraphs(text: str) -> list[str]:
    """Split a contract into clause-sized chunks.

    The classifier was trained on individual CUAD clause spans (short,
    single-topic excerpts), not whole multi-clause documents. Classifying
    the full document as one input dilutes the signal across many topics
    at once. Splitting on blank lines / numbered-clause boundaries and
    classifying each chunk separately much more closely matches the
    training distribution.

    Falls back to the whole text as a single "paragraph" if the document
    has no blank-line structure (e.g. OCR output with no paragraph breaks).
    """
    # Split on blank lines, or on "<number>. WORDS:" clause headers.
    chunks = re.split(r"\n\s*\n|\n(?=\d+\.\s+[A-Z])", text)
    chunks = [c.strip() for c in chunks if len(c.strip()) >= _MIN_PARAGRAPH_CHARS]
    return chunks if chunks else [text]


def _classify_document(raw_text: str, ner_entities: list) -> list:
    """Classify each paragraph separately and merge by max confidence per label."""
    from classification.inference import classify_clauses

    paragraphs = _split_into_paragraphs(raw_text)
    logger.info("classifying %d paragraph(s)", len(paragraphs))

    best: dict[str, object] = {}
    for para in paragraphs:
        para_results = classify_clauses(para, ner_entities=ner_entities)
        for r in para_results:
            current = best.get(r.clause_type)
            if current is None or r.confidence > current.confidence:
                best[r.clause_type] = r

    return list(best.values())


def _compute_risk(clauses: list) -> tuple[float, str]:
    """Simple weighted-sum heuristic over which clauses are present.
    This is intentionally transparent (not a learned model) so every score
    is explainable in the risk report — see api/routers/risk.py."""
    score = 0.0
    for c in clauses:
        if not c.present:
            continue
        weight = HIGH_RISK_CLAUSES.get(c.clause_type) or PROTECTIVE_CLAUSES.get(c.clause_type)
        if weight:
            score += weight * c.confidence

    score = max(0.0, score)
    if score < RISK_THRESHOLDS["LOW"]:
        level = "LOW"
    elif score < RISK_THRESHOLDS["MEDIUM"]:
        level = "MEDIUM"
    else:
        level = "HIGH"
    return round(score, 2), level


async def process_contract(contract_id: str, file_path: str) -> None:
    """Runs the full pipeline for one contract and persists results.
    Any exception is caught and written to the row's `error` column so the
    status endpoint can report a clean failure instead of hanging forever."""
    async with SessionLocal() as session:
        result = await session.execute(select(Contract).where(Contract.id == contract_id))
        contract = result.scalar_one_or_none()
        if contract is None:
            logger.error("process_contract: contract %s not found", contract_id)
            return

        contract.status = "processing"
        await session.commit()

        try:
            # 1. Extract text (auto-routes PDF/DOCX/TXT, OCR fallback for scans)
            from ingestion.document_router import DocumentRouter
            extraction = DocumentRouter().route(file_path)
            raw_text = extraction.raw_text

            # 2. NER
            from ner.inference import extract_entities, load_model
            load_model(str(NER_MODEL_PATH))
            entities = extract_entities(raw_text)

            # 3. Clause classification (per-paragraph, then merged — see _classify_document)
            from classification.inference import load_classifier
            load_classifier(model_dir=str(CLASSIFIER_MODEL_PATH), labels_path=str(CLAUSE_LABELS_PATH))
            clauses = _classify_document(raw_text, entities)

            # 4. Risk score
            risk_score, risk_level = _compute_risk(clauses)

            # 5. Embedding + vector index
            from embeddings.embedder import embed_text
            from embeddings.vector_store import add as vs_add
            vector = embed_text(raw_text)
            vs_add(contract_id, vector)

            # 6. Persist
            contract.raw_text = raw_text
            contract.entities_json = json.dumps([
                {"text": e.text, "label": e.label, "start_char": e.start_char,
                 "end_char": e.end_char, "confidence": e.confidence}
                for e in entities
            ])
            contract.clauses_json = json.dumps([
                {"clause_type": c.clause_type, "present": c.present,
                 "confidence": c.confidence, "evidence_spans": c.evidence_spans}
                for c in clauses
            ])
            contract.risk_score = risk_score
            contract.risk_level = risk_level
            contract.status = "complete"

        except Exception as exc:  # noqa: BLE001 — deliberately broad: this is a background job
            logger.exception("Pipeline failed for contract %s", contract_id)
            contract.status = "failed"
            contract.error = str(exc)

        await session.commit()
