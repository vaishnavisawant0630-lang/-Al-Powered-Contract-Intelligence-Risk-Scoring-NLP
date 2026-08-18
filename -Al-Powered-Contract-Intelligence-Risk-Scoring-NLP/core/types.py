"""
core/types.py
=============
Shared data structures for the entire Contract Intelligence platform.

PURPOSE
-------
This file is the single source of truth for every data shape that crosses
a module boundary. If data is passed between ingestion → ner → api, its type
is defined here — never inside the producing module.

RULES
-----
- NO business logic in this file (no methods that do computation)
- NO imports from any other module inside this project
- Only stdlib + dataclasses + enum + typing

TYPES DEFINED
-------------

Enums:
    DocumentType        — recognised input file formats
    ExtractionMethod    — how text was obtained from a document
    EntityLabel         — the 41 CUAD clause labels + 4 core NER labels

Dataclasses (immutable by default via frozen=True):
    Entity              — a single extracted named entity span
    ExtractionResult    — the output of any BaseExtractor.extract() call
    NERSample           — one training example: text + list of Entity spans
    ClauseSample        — one training example for Phase-2 clause classification
    SpanConflict        — record of a span overlap that was resolved

USAGE EXAMPLE
-------------
    from core.types import Entity, ExtractionResult, DocumentType
    result = ExtractionResult(
        source_path="contracts/acme.pdf",
        raw_text="This Agreement is between Acme Corp ...",
        method=ExtractionMethod.PDF_DIRECT,
        page_count=12,
        metadata={"producer": "Adobe"},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentType(Enum):
    """
    Recognised input document formats.

    Used by DocumentRouter to select the correct extractor.
    Adding a new format here is step 1 of adding a new extractor.
    """
    PDF_DIGITAL = auto()   # Selectable text layer present → PdfExtractor
    PDF_SCANNED = auto()   # Image-only PDF → OcrExtractor
    DOCX = auto()          # Microsoft Word .docx → DocxExtractor
    TXT = auto()           # Plain text / markdown → TextExtractor
    UNKNOWN = auto()       # Could not determine; Router will raise IngestionError


class ExtractionMethod(Enum):
    """
    Records how text was extracted, for audit-trail and metrics.
    Stored in ExtractionResult.method.
    """
    PDF_DIRECT = "pdf_direct"        # pdfminer.six — no OCR needed
    PDF_OCR = "pdf_ocr"              # pdf2image + pytesseract
    DOCX_PARSE = "docx_parse"        # python-docx paragraph stitching
    TXT_READ = "txt_read"            # Direct UTF-8 / latin-1 file read


class EntityLabel(str, Enum):
    """
    All entity / clause labels recognised by the NER model.

    GROUP 1 — Core spaCy NER labels (4)
        Used by baseline model to extract universal legal primitives.

    GROUP 2 — CUAD 41 clause-type labels
        All 41 question categories from the CUAD dataset, mapped to
        short snake_case identifiers. Used for both NER span labels
        and Phase-2 clause classification targets.

    Naming convention: CUAD question index preserved in comment for traceability.
    """
    # --- Core NER (Group 1) ---
    ORG = "ORG"                                     # Organisation names
    DATE = "DATE"                                   # Dates and durations
    MONEY = "MONEY"                                 # Monetary values
    GPE = "GPE"                                     # Geopolitical entity / Jurisdiction

    # --- CUAD Clause Labels (Group 2, Q1–Q41) ---
    DOCUMENT_NAME = "DOCUMENT_NAME"                 # Q1
    PARTIES = "PARTIES"                             # Q2
    AGREEMENT_DATE = "AGREEMENT_DATE"               # Q3
    EFFECTIVE_DATE = "EFFECTIVE_DATE"               # Q4
    EXPIRATION_DATE = "EXPIRATION_DATE"             # Q5
    RENEWAL_TERM = "RENEWAL_TERM"                   # Q6
    NOTICE_PERIOD_TO_TERMINATE = "NOTICE_PERIOD_TO_TERMINATE"  # Q7
    GOVERNING_LAW = "GOVERNING_LAW"                 # Q8
    MOST_FAVORED_NATION = "MOST_FAVORED_NATION"     # Q9
    NON_COMPETE = "NON_COMPETE"                     # Q10
    EXCLUSIVITY = "EXCLUSIVITY"                     # Q11
    NO_SOLICIT_OF_CUSTOMERS = "NO_SOLICIT_OF_CUSTOMERS"  # Q12
    NO_SOLICIT_OF_EMPLOYEES = "NO_SOLICIT_OF_EMPLOYEES"  # Q13
    NON_DISPARAGEMENT = "NON_DISPARAGEMENT"         # Q14
    TERMINATION_FOR_CONVENIENCE = "TERMINATION_FOR_CONVENIENCE"  # Q15
    ROFR_ROFO_ROFN = "ROFR_ROFO_ROFN"              # Q16
    CHANGE_OF_CONTROL = "CHANGE_OF_CONTROL"         # Q17
    ANTI_ASSIGNMENT = "ANTI_ASSIGNMENT"             # Q18
    REVENUE_PROFIT_SHARING = "REVENUE_PROFIT_SHARING"  # Q19
    PRICE_RESTRICTION = "PRICE_RESTRICTION"         # Q20
    MINIMUM_COMMITMENT = "MINIMUM_COMMITMENT"       # Q21
    VOLUME_RESTRICTION = "VOLUME_RESTRICTION"       # Q22
    IP_OWNERSHIP_ASSIGNMENT = "IP_OWNERSHIP_ASSIGNMENT"  # Q23
    JOINT_IP_OWNERSHIP = "JOINT_IP_OWNERSHIP"       # Q24
    LICENSE_GRANT = "LICENSE_GRANT"                 # Q25
    NON_TRANSFERABLE_LICENSE = "NON_TRANSFERABLE_LICENSE"  # Q26
    AFFILIATE_LICENSE_LICENSOR = "AFFILIATE_LICENSE_LICENSOR"  # Q27
    AFFILIATE_LICENSE_LICENSEE = "AFFILIATE_LICENSE_LICENSEE"  # Q28
    UNLIMITED_LICENSE = "UNLIMITED_LICENSE"         # Q29
    IRREVOCABLE_OR_PERPETUAL = "IRREVOCABLE_OR_PERPETUAL"  # Q30
    SOURCE_CODE_ESCROW = "SOURCE_CODE_ESCROW"       # Q31
    POST_TERMINATION_SERVICES = "POST_TERMINATION_SERVICES"  # Q32
    AUDIT_RIGHTS = "AUDIT_RIGHTS"                   # Q33
    UNCAPPED_LIABILITY = "UNCAPPED_LIABILITY"        # Q34
    CAP_ON_LIABILITY = "CAP_ON_LIABILITY"           # Q35
    LIQUIDATED_DAMAGES = "LIQUIDATED_DAMAGES"       # Q36
    WARRANTY_DURATION = "WARRANTY_DURATION"         # Q37
    INSURANCE = "INSURANCE"                         # Q38
    COVENANT_NOT_TO_SUE = "COVENANT_NOT_TO_SUE"     # Q39
    THIRD_PARTY_BENEFICIARY = "THIRD_PARTY_BENEFICIARY"  # Q40
    LIMITATION_OF_LIABILITY = "LIMITATION_OF_LIABILITY"  # Q41


# ---------------------------------------------------------------------------
# Dataclasses — immutable, hashable, serialisation-safe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    """
    A single extracted named-entity span from a contract.

    Produced by: ner/inference.py → NERModel.extract_entities()
    Consumed by: API response serialiser (Phase 3), risk scorer (Phase 4)

    Attributes
    ----------
    label       Entity type from EntityLabel enum
    text        The matched surface form exactly as it appears in the document
    start       Character offset (inclusive) in the source text
    end         Character offset (exclusive) in the source text
    score       Confidence score in [0.0, 1.0]; -1.0 means not available
    """
    label: str          # EntityLabel value
    text: str
    start: int
    end: int
    score: float = -1.0


@dataclass(frozen=True)
class ExtractionResult:
    """
    The output of any BaseExtractor.extract() call.

    Produced by: PdfExtractor / OcrExtractor / DocxExtractor / TextExtractor
    Consumed by: TextCleaner, NERModel, data pipeline, API layer

    Attributes
    ----------
    source_path     Absolute path to the original document
    raw_text        Extracted text BEFORE cleaning (preserves original for audit)
    method          How text was obtained (ExtractionMethod enum)
    page_count      Number of pages processed; -1 if unknown
    metadata        Extractor-specific key/value pairs (e.g. PDF producer, OCR conf)
    """
    source_path: str
    raw_text: str
    method: str               # ExtractionMethod.value
    page_count: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NERSample:
    """
    One spaCy NER training / evaluation example.

    Produced by: data_processing/cuad_to_ner.py after span validation
    Consumed by: ner/train.py (written to DocBin), ner/evaluate.py

    Attributes
    ----------
    text        The raw contract paragraph text
    entities    List of Entity spans aligned to `text` (non-overlapping)
    doc_id      Optional CUAD document identifier for traceability
    """
    text: str
    entities: tuple[Entity, ...]  # tuple for hashability / frozen compat
    doc_id: str = ""


@dataclass(frozen=True)
class ClauseSample:
    """
    One clause-classification training example (used in Phase 2).

    Produced by: data_processing/cuad_to_classification.py
    Consumed by: Phase-2 transformer fine-tuning pipeline

    Attributes
    ----------
    text        The clause text snippet
    label       One of the 41 CUAD EntityLabel values (string)
    is_present  True = clause is present; False = negative example
    doc_id      Optional CUAD document identifier
    meta        Any extra fields (e.g., question_index, answer_start)
    """
    text: str
    label: str            # EntityLabel.value
    is_present: bool
    doc_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpanConflict:
    """
    Audit record emitted by SpanValidator when two spans overlap.

    Produced by: data_processing/span_validator.py
    Consumed by: dataset_stats.py (for reporting), structlog (for warning)

    Resolution strategy (Phase 1 decision): keep the LONGER span.

    Attributes
    ----------
    doc_id          Source document
    kept            The span that was retained
    discarded       The span that was removed
    reason          Human-readable explanation e.g. "longer span wins"
    """
    doc_id: str
    kept: Entity
    discarded: Entity
    reason: str = "longer span wins"
