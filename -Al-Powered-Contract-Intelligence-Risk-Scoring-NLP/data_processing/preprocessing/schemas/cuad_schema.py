"""
data_processing/preprocessing/schemas/cuad_schema.py
=====================================================
Typed dataclass schemas for every artefact produced by the 41-stage pipeline.

Every stage reads a typed input and writes a typed output.
This makes the pipeline inspectable, JSON-serialisable, and testable in isolation.

SCHEMA HIERARCHY
----------------
DocumentManifest        → Stage 01  (immutable ground truth)
IntegrityReport         → Stage 02
DeduplicationReport     → Stage 03
PdfInspection           → Stage 05
PageClassification      → Stage 06
RenderedPage            → Stage 07
OcrDocument / OcrWord   → Stage 28–29
CuadAnnotation          → raw CUAD input
AlignedAnnotation       → Stage 32
LabelledSample          → Stage 34  (final training record)
DatasetSplit            → Stage 37
ValidationReport        → Stage 38–40
ReleaseGateReport       → Stage 41  (hard release gate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    NATIVE_TEXT = "native_text"
    SCANNED     = "scanned"
    HYBRID      = "hybrid"


class PageType(str, Enum):
    NATIVE  = "native"
    SCANNED = "scanned"


class RegionType(str, Enum):
    TITLE     = "TITLE"
    HEADING   = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    TABLE     = "TABLE"
    LIST      = "LIST"
    FOOTNOTE  = "FOOTNOTE"
    HEADER    = "HEADER"
    FOOTER    = "FOOTER"
    SIGNATURE = "SIGNATURE"
    PAGE_NUM  = "PAGE_NUM"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    page: int


# ─────────────────────────────────────────────────────────────────────────────
# Stage 01 — Immutable document manifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentManifest:
    """
    Created once from CUADv1.json. Never modified.
    Every downstream stage receives this record and adds its own output alongside it.
    """
    document_id:  str    # Canonical: CUAD_000001 … CUAD_000510
    source_file:  str    # 'title' field from CUADv1.json
    sha256:       str    # SHA-256 of the context string
    char_count:   int    # len(context)
    qa_count:     int    # Always 41
    created_at:   str    # ISO-8601 UTC timestamp


# ─────────────────────────────────────────────────────────────────────────────
# Stage 02 — File integrity
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntegrityCheck:
    document_id: str
    passes:      bool
    issues:      list[str] = field(default_factory=list)


@dataclass
class IntegrityReport:
    total_documents: int
    passed:          int
    failed:          int
    checks:          list[IntegrityCheck] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return self.failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 03 — Deduplication
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DuplicatePair:
    doc_id_a:   str
    doc_id_b:   str
    similarity: float      # 0.0–1.0
    method:     str        # "sha256" | "simhash" | "minhash" | "ngram"


@dataclass
class DeduplicationReport:
    total_scanned:    int
    exact_duplicates: int
    near_duplicates:  int
    removed:          list[str]
    pairs:            list[DuplicatePair] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 05–06 — PDF structural inspection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageClassification:
    page_number:      int
    page_type:        PageType
    char_count:       int      # Chars extracted by pdfminer on this page
    image_area_ratio: float    # 0.0–1.0 (1.0 = pure image)
    confidence:       float    # Decision confidence


@dataclass
class PdfInspection:
    document_id:    str
    document_type:  DocumentType
    total_pages:    int
    pages:          list[PageClassification] = field(default_factory=list)
    has_tables:     bool = False
    has_multicolumn: bool = False
    has_forms:      bool = False

    @property
    def scanned_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.page_type == PageType.SCANNED]

    @property
    def native_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.page_type == PageType.NATIVE]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 07–20 — Image preprocessing record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessingStep:
    """Records one image transformation applied to a page."""
    name:       str            # "deskew" | "sauvola" | "clahe" | ...
    applied:    bool           # False = skipped (not needed)
    params:     dict[str, Any] = field(default_factory=dict)
    conf_before: float | None = None
    conf_after:  float | None = None


@dataclass
class RenderedPage:
    document_id:     str
    page_number:     int
    dpi:             int
    width_px:        int
    height_px:       int
    skew_angle_deg:  float
    binarization:    str       # "sauvola" | "otsu" | "adaptive_gaussian" | "none"
    steps_applied:   list[PreprocessingStep] = field(default_factory=list)
    image_path:      str | None = None   # Path in data/intermediate/


# ─────────────────────────────────────────────────────────────────────────────
# Stage 21–27 — Layout
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayoutRegion:
    region_type:   RegionType
    bbox:          BoundingBox
    text:          str
    reading_order: int


@dataclass
class PageLayout:
    document_id:      str
    page_number:      int
    column_count:     int
    regions:          list[LayoutRegion] = field(default_factory=list)
    headers:          list[str]          = field(default_factory=list)
    footers:          list[str]          = field(default_factory=list)
    page_number_text: str | None         = None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 28–29 — OCR output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OcrWord:
    """
    Atomic OCR unit. Every downstream operation traces back to OcrWord.
    """
    text:           str
    confidence:     float      # 0.0–1.0 (normalised from Tesseract 0–100)
    bbox:           BoundingBox
    is_legal_token: bool = False   # Protected by Stage 31 vocabulary


@dataclass
class OcrPage:
    document_id:     str
    page_number:     int
    raw_text:        str
    words:           list[OcrWord]
    mean_confidence: float
    low_conf_ratio:  float     # Fraction of words with conf < 0.80
    ocr_method:      str       # "tesseract" | "native_pdf"


@dataclass
class OcrDocument:
    document_id:     str
    pages:           list[OcrPage]
    full_text:       str       # Concatenated with "\n\n--- PAGE {n} ---\n\n"
    mean_confidence: float
    total_words:     int

    @property
    def low_confidence_pages(self) -> list[int]:
        return [p.page_number for p in self.pages if p.mean_confidence < 0.85]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 32–35 — CUAD alignment and labelling
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CuadAnnotation:
    """Raw annotation from CUADv1.json — read-only."""
    document_id:   str
    question_id:   str
    clause_type:   str
    question_text: str
    answer_texts:  list[str]
    answer_starts: list[int]   # Char offsets in original context
    is_present:    bool        # True if answer_texts is non-empty


@dataclass
class AlignedAnnotation:
    """CUAD annotation aligned from original context offsets → OCR text offsets."""
    annotation:      CuadAnnotation
    ocr_answer_text: str | None
    ocr_char_start:  int | None
    ocr_char_end:    int | None
    alignment_score: float    # IoU vs original span (0.0–1.0)
    alignment_method: str     # "exact" | "fuzzy" | "failed"
    bbox:            BoundingBox | None = None


@dataclass
class LabelledSample:
    """
    Final training record — what enters the model.

    NER:            text + char_start/end per entity
    Classification: text + binary label (41-dim vector assembled at Stage 34)
    """
    sample_id:        str     # "{doc_id}__{clause_type}__{idx}"
    document_id:      str
    clause_type:      str
    text:             str
    char_start:       int
    char_end:         int
    label:            int     # 1 = present, 0 = absent
    ocr_confidence:   float
    page_number:      int | None
    is_hard_negative: bool = False
    split:            str  = "train"   # "train" | "val" | "test"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 37 — Dataset split
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSplit:
    train_doc_ids: list[str]
    val_doc_ids:   list[str]
    test_doc_ids:  list[str]
    train_samples: int
    val_samples:   int
    test_samples:  int

    @property
    def total_samples(self) -> int:
        return self.train_samples + self.val_samples + self.test_samples


# ─────────────────────────────────────────────────────────────────────────────
# Stage 38–40 — Validation metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OcrQualityMetrics:
    document_id:       str
    mean_confidence:   float
    median_confidence: float
    p10_confidence:    float
    low_conf_ratio:    float
    cer:               float | None = None
    wer:               float | None = None


@dataclass
class SpanMetrics:
    clause_type: str
    exact_match: float
    iou_mean:    float
    precision:   float
    recall:      float
    f1:          float
    support:     int


@dataclass
class ValidationReport:
    ocr_metrics:     list[OcrQualityMetrics]
    span_metrics:    list[SpanMetrics]
    macro_f1:        float
    micro_f1:        float
    class_imbalance: dict[str, float]   # clause_type → positive_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Stage 41 — Release gate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GateCheck:
    name:    str
    status:  GateStatus
    message: str = ""


@dataclass
class ReleaseGateReport:
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return all(c.status == GateStatus.PASS for c in self.checks)

    @property
    def failed_checks(self) -> list[GateCheck]:
        return [c for c in self.checks if c.status == GateStatus.FAIL]

    def summary(self) -> str:
        total  = len(self.checks)
        passed = sum(1 for c in self.checks if c.status == GateStatus.PASS)
        failed = sum(1 for c in self.checks if c.status == GateStatus.FAIL)
        warned = sum(1 for c in self.checks if c.status == GateStatus.WARN)
        icon   = "✅ RELEASED" if self.all_pass else "❌ BLOCKED"
        return (f"Release Gate: {passed}/{total} PASS | "
                f"{failed} FAIL | {warned} WARN | {icon}")
