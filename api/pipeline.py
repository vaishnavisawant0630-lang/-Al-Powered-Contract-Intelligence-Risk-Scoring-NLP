"""
api/pipeline.py
================

End-to-end contract processing pipeline:

    File
      ↓
    Document extraction / OCR
      ↓
    NER
      ↓
    Clause classification
      ↓
    Risk scoring
      ↓
    Embedding / vector store
      ↓
    Database persistence

Runs as a FastAPI BackgroundTask.
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


# ============================================================
# TEXT SPLITTING
# ============================================================

def _split_into_paragraphs(
    text: str,
) -> list[str]:
    """
    Split contract text into clause-sized chunks.

    The classifier was trained on individual clause spans,
    so classifying smaller chunks gives better results than
    sending the entire contract at once.
    """

    if not text or not text.strip():
        return []

    chunks = re.split(
        r"\n\s*\n|\n(?=\d+\.\s+[A-Z])",
        text,
    )

    chunks = [
        chunk.strip()
        for chunk in chunks
        if len(chunk.strip())
        >= _MIN_PARAGRAPH_CHARS
    ]

    return (
        chunks
        if chunks
        else [text.strip()]
    )


# ============================================================
# CLAUSE CLASSIFICATION
# ============================================================

def _classify_document(
    raw_text: str,
    ner_entities: list,
) -> list:
    """
    Classify each paragraph separately.

    If the same clause type appears multiple times,
    keep the result with the highest confidence.

    Each ClauseResult already contains the actual text
    classified by the model.
    """

    from classification.inference import (
        classify_clauses,
    )

    paragraphs = _split_into_paragraphs(
        raw_text
    )

    logger.info(
        "Classifying %d paragraph(s)",
        len(paragraphs),
    )

    best: dict[str, object] = {}

    for index, para in enumerate(
        paragraphs,
        start=1,
    ):

        logger.info(
            "Classifying paragraph %d/%d",
            index,
            len(paragraphs),
        )

        # ----------------------------------------------------
        # Classify paragraph
        # ----------------------------------------------------

        para_results = classify_clauses(
            para,
            ner_entities=ner_entities,
        )

        # ----------------------------------------------------
        # Store highest-confidence result
        # ----------------------------------------------------

        for result in para_results:

            current = best.get(
                result.clause_type
            )

            if (
                current is None
                or result.confidence
                > current.confidence
            ):

                best[
                    result.clause_type
                ] = result

    return list(
        best.values()
    )


# ============================================================
# RISK CALCULATION
# ============================================================

def _compute_risk(
    clauses: list,
) -> tuple[float, str]:
    """
    Calculate contract risk on a 0-100 scale.

    Risk levels:
        0-29   = LOW
        30-59  = MEDIUM
        60-79  = HIGH
        80-100 = CRITICAL
    """

    score = 0.0

    logger.info(
        "Calculating risk from %d clause(s)",
        len(clauses),
    )

    for clause in clauses:

        if not clause.present:
            continue

        clause_type = (
            clause.clause_type
        )

        confidence = float(
            clause.confidence or 0.0
        )

        # ----------------------------------------------------
        # High-risk clause
        # ----------------------------------------------------

        if clause_type in HIGH_RISK_CLAUSES:

            weight = (
                HIGH_RISK_CLAUSES[
                    clause_type
                ]
            )

            contribution = (
                weight * confidence
            )

            score += contribution

            logger.info(
                "High-risk clause: %s | "
                "weight=%s | "
                "confidence=%.3f | "
                "contribution=%.3f",
                clause_type,
                weight,
                confidence,
                contribution,
            )

        # ----------------------------------------------------
        # Protective clause
        # ----------------------------------------------------

        elif clause_type in PROTECTIVE_CLAUSES:

            weight = (
                PROTECTIVE_CLAUSES[
                    clause_type
                ]
            )

            contribution = (
                weight * confidence
            )

            score -= contribution

            logger.info(
                "Protective clause: %s | "
                "weight=%s | "
                "confidence=%.3f | "
                "contribution=-%.3f",
                clause_type,
                weight,
                confidence,
                contribution,
            )

    # --------------------------------------------------------
    # Keep score between 0 and 100
    # --------------------------------------------------------

    score = max(
        0.0,
        min(100.0, score),
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if score < RISK_THRESHOLDS["LOW"]:

        level = "LOW"

    elif score < RISK_THRESHOLDS["MEDIUM"]:

        level = "MEDIUM"

    elif score < RISK_THRESHOLDS["HIGH"]:

        level = "HIGH"

    else:

        level = "CRITICAL"

    score = round(
        score,
        2,
    )

    logger.info(
        "FINAL RISK SCORE = %.2f | LEVEL = %s",
        score,
        level,
    )

    return score, level


# ============================================================
# MAIN PIPELINE
# ============================================================

async def process_contract(
    contract_id: str,
    file_path: str,
) -> None:
    """
    Process one uploaded contract.

    Pipeline:

        1. Document extraction
        2. NER
        3. Clause classification
        4. Risk scoring
        5. Embedding
        6. Vector store
        7. Database persistence
    """

    async with SessionLocal() as session:

        # ----------------------------------------------------
        # Find contract
        # ----------------------------------------------------

        result = await session.execute(
            select(Contract).where(
                Contract.id == contract_id
            )
        )

        contract = (
            result.scalar_one_or_none()
        )

        if contract is None:

            logger.error(
                "Contract %s not found",
                contract_id,
            )

            return

        # ----------------------------------------------------
        # Processing status
        # ----------------------------------------------------

        contract.status = "processing"

        contract.error = None

        await session.commit()

        logger.info(
            "Started processing contract %s",
            contract_id,
        )

        try:

            # =================================================
            # 1. DOCUMENT EXTRACTION
            # =================================================

            logger.info(
                "[1/6] Extracting document text..."
            )

            from ingestion.document_router import (
                DocumentRouter,
            )

            extraction = (
                DocumentRouter().route(
                    file_path
                )
            )

            raw_text = (
                extraction.raw_text
            )

            if (
                not raw_text
                or not raw_text.strip()
            ):

                raise ValueError(
                    "No text could be extracted "
                    "from the uploaded document."
                )

            logger.info(
                "Extracted %d characters",
                len(raw_text),
            )

            # =================================================
            # 2. NER
            # =================================================

            logger.info(
                "[2/6] Running NER..."
            )

            from ner.inference import (
                extract_entities,
                load_model,
            )

            load_model(
                str(NER_MODEL_PATH)
            )

            entities = extract_entities(
                raw_text
            )

            logger.info(
                "NER extracted %d entities",
                len(entities),
            )

            # =================================================
            # 3. CLAUSE CLASSIFICATION
            # =================================================

            logger.info(
                "[3/6] Running clause classification..."
            )

            from classification.inference import (
                load_classifier,
            )

            load_classifier(
                model_dir=str(
                    CLASSIFIER_MODEL_PATH
                ),
                labels_path=str(
                    CLAUSE_LABELS_PATH
                ),
            )

            clauses = (
                _classify_document(
                    raw_text,
                    entities,
                )
            )

            logger.info(
                "Detected %d clause types",
                len(clauses),
            )

            # =================================================
            # 4. RISK SCORE
            # =================================================

            logger.info(
                "[4/6] Calculating risk score..."
            )

            (
                risk_score,
                risk_level,
            ) = _compute_risk(
                clauses
            )

            logger.info(
                "Risk score = %.2f | "
                "Risk level = %s",
                risk_score,
                risk_level,
            )

            # =================================================
            # 5. EMBEDDING / VECTOR STORE
            # =================================================

            logger.info(
                "[5/6] Creating document embedding..."
            )

            from embeddings.embedder import (
                embed_text,
            )

            from embeddings.vector_store import (
                add as vs_add,
            )

            vector = embed_text(
                raw_text
            )

            vs_add(
                contract_id,
                vector,
            )

            logger.info(
                "Embedding stored for contract %s",
                contract_id,
            )

            # =================================================
            # 6. DATABASE PERSISTENCE
            # =================================================

            logger.info(
                "[6/6] Saving analysis results..."
            )

            # -------------------------------------------------
            # Raw extracted text
            # -------------------------------------------------

            contract.raw_text = raw_text

            # -------------------------------------------------
            # NER results
            # -------------------------------------------------

            contract.entities_json = (
                json.dumps(
                    [
                        {
                            "text": entity.text,
                            "label": entity.label,
                            "start_char": entity.start_char,
                            "end_char": entity.end_char,
                            "confidence": entity.confidence,
                        }
                        for entity in entities
                    ]
                )
            )

            # -------------------------------------------------
            # Clause results
            # -------------------------------------------------

            contract.clauses_json = (
                json.dumps(
                    [
                        {
                            "clause_type":
                                clause.clause_type,

                            "present":
                                clause.present,

                            "confidence":
                                float(
                                    clause.confidence
                                    or 0.0
                                ),

                            # =================================
                            # IMPORTANT:
                            # Actual clause text
                            # =================================

                            "text":
                                clause.clause_text,

                            # =================================
                            # Evidence spans
                            # =================================

                            "evidence_spans":
                                clause.evidence_spans,
                        }
                        for clause in clauses
                    ]
                )
            )

            # -------------------------------------------------
            # Risk
            # -------------------------------------------------

            contract.risk_score = (
                risk_score
            )

            contract.risk_level = (
                risk_level
            )

            # -------------------------------------------------
            # Completed
            # -------------------------------------------------

            contract.status = "completed"

            contract.error = None

            await session.commit()

            # =================================================
            # LOGGING
            # =================================================

            logger.info(
                "================================================"
            )

            logger.info(
                "CONTRACT PROCESSING COMPLETED"
            )

            logger.info(
                "Contract ID : %s",
                contract_id,
            )

            logger.info(
                "Risk Score  : %.2f",
                risk_score,
            )

            logger.info(
                "Risk Level  : %s",
                risk_level,
            )

            logger.info(
                "Clauses     : %d",
                len(clauses),
            )

            logger.info(
                "Entities    : %d",
                len(entities),
            )

            logger.info(
                "Status      : completed"
            )

            logger.info(
                "================================================"
            )

        except Exception as exc:

            # ------------------------------------------------
            # Pipeline failed
            # ------------------------------------------------

            logger.exception(
                "Pipeline failed for contract %s",
                contract_id,
            )

            contract.status = "failed"

            contract.error = str(exc)

            await session.commit()