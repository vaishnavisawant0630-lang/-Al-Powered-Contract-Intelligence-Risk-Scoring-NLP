"""
core/exceptions.py
==================
Typed exception hierarchy for the Contract Intelligence platform.

PURPOSE
-------
Centralise all custom exceptions so callers can catch specific error types
rather than bare Exception. Every module raises from this hierarchy.

DESIGN
------
All exceptions inherit from ContractIntelligenceError (the root).
This allows callers to catch all platform errors with one except clause,
or selectively catch subsystem-specific errors.

HIERARCHY
---------
ContractIntelligenceError               ← root; never raise this directly
│
├── IngestionError                       ← raised by ingestion/ modules
│   ├── UnsupportedFileTypeError         ← DocumentRouter: unknown extension
│   ├── ExtractionError                  ← PdfExtractor / DocxExtractor failure
│   └── OCRError                         ← OcrExtractor failure (Tesseract crash)
│
├── DataProcessingError                  ← raised by data_processing/ modules
│   ├── CuadLoadError                    ← HuggingFace dataset download / parse fail
│   ├── SpanValidationError              ← span_validator: unrecoverable span issue
│   └── ConversionError                  ← cuad_to_ner / cuad_to_classification fail
│
├── NERError                             ← raised by ner/ modules
│   ├── ModelNotFoundError               ← saved model path doesn't exist
│   ├── ModelLoadError                   ← model exists but fails to load
│   └── InferenceError                   ← extract_entities() runtime failure
│
└── ConfigurationError                   ← raised by core/config.py
    └── MissingEnvVarError               ← required env var not set

USAGE EXAMPLES
--------------
    # In pdf_extractor.py:
    from core.exceptions import ExtractionError
    raise ExtractionError("pdfminer failed", path=str(path), page=3)

    # In a caller:
    from core.exceptions import IngestionError
    try:
        result = extractor.extract(path)
    except IngestionError as exc:
        log.error("ingestion_failed", error=str(exc), **exc.context)
        raise

CONTEXT DICT
------------
All exceptions carry a `context` dict of structured key-value metadata.
This is forwarded to structlog so error logs are machine-queryable.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class ContractIntelligenceError(Exception):
    """
    Root exception for all platform errors.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    **context : Any
        Structured key-value pairs added to log output.
        Example: path="contracts/acme.pdf", page=3
    """
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context


# ---------------------------------------------------------------------------
# Ingestion errors  (raised by ingestion/ package)
# ---------------------------------------------------------------------------

class IngestionError(ContractIntelligenceError):
    """Base for all document ingestion failures."""


class UnsupportedFileTypeError(IngestionError):
    """
    Raised by DocumentRouter when no extractor can handle the given file.

    Context keys typically included:
        path (str)          : path to the file
        detected_type (str) : MIME type or extension that was detected
    """


class ExtractionError(IngestionError):
    """
    Raised when a text extractor fails to produce output.

    Context keys typically included:
        path (str)          : source file path
        extractor (str)     : class name of the failing extractor
        page (int)          : page number if applicable
    """


class OCRError(IngestionError):
    """
    Raised when Tesseract or pdf2image raises an unexpected error.

    Context keys typically included:
        path (str)          : source PDF path
        page (int)          : page being processed when failure occurred
        tesseract_exit (int): tesseract process exit code
    """


# ---------------------------------------------------------------------------
# Data processing errors  (raised by data_processing/ package)
# ---------------------------------------------------------------------------

class DataProcessingError(ContractIntelligenceError):
    """Base for all data-processing pipeline failures."""


class CuadLoadError(DataProcessingError):
    """
    Raised when HuggingFace datasets fails to load or parse CUAD.

    Context keys typically included:
        dataset_path (str)  : local cache path or hub identifier
        split (str)         : "train" | "test"
    """


class SpanValidationError(DataProcessingError):
    """
    Raised by SpanValidator for an unrecoverable span issue.

    Note: Overlapping spans are NOT raised — they are resolved and logged.
    This is raised only when a span has start > end or references out-of-range chars.

    Context keys typically included:
        doc_id (str)        : CUAD document identifier
        start (int)         : span start character offset
        end (int)           : span end character offset
        text_length (int)   : length of the source text
    """


class ConversionError(DataProcessingError):
    """
    Raised when a converter (cuad_to_ner, cuad_to_classification) fails
    on a specific sample.

    Context keys typically included:
        doc_id (str)        : failing document identifier
        question_idx (int)  : CUAD question index being processed
    """


# ---------------------------------------------------------------------------
# NER errors  (raised by ner/ package)
# ---------------------------------------------------------------------------

class NERError(ContractIntelligenceError):
    """Base for all NER model failures."""


class ModelNotFoundError(NERError):
    """
    Raised when the saved model directory does not exist.

    Context keys typically included:
        model_path (str)    : path that was expected to contain the model
    """


class ModelLoadError(NERError):
    """
    Raised when spaCy fails to load a model that does exist on disk.

    Context keys typically included:
        model_path (str)    : path to the corrupted / incompatible model
    """


class InferenceError(NERError):
    """
    Raised when extract_entities() encounters a runtime failure.

    Context keys typically included:
        text_snippet (str)  : first 100 chars of the failing input
    """


# ---------------------------------------------------------------------------
# Configuration errors  (raised by core/config.py)
# ---------------------------------------------------------------------------

class ConfigurationError(ContractIntelligenceError):
    """Base for configuration failures."""


class MissingEnvVarError(ConfigurationError):
    """
    Raised when a required environment variable is not set.

    Context keys typically included:
        var_name (str)      : name of the missing variable
    """
