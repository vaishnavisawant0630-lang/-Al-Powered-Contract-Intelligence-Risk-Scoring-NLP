"""
ingestion/base.py
=================
BaseExtractor Protocol — the contract that ALL extractors must implement.

PURPOSE
-------
Defines the interface that DocumentRouter depends on. No extractor concrete
class is ever directly imported by the router — only this Protocol.

This is the key to the Open/Closed principle in the ingestion layer:
    - OPEN for extension: add a new extractor → implement BaseExtractor
    - CLOSED for modification: router never changes

WHY PROTOCOL NOT ABC
--------------------
Python's typing.Protocol enables structural subtyping (duck typing with
type-checker support). An extractor does NOT need to explicitly inherit
from BaseExtractor — it just needs to implement the two methods.
This means third-party extractors can also be plugged in without
modification.

CONTRACT
--------
Any class that:
    1. Has a method extract(path: Path) → ExtractionResult
    2. Has a method can_handle(path: Path) → bool
...automatically satisfies BaseExtractor at runtime and type-check time.

IMPLEMENTORS IN THIS PACKAGE
----------------------------
    PdfExtractor       pdf_extractor.py
    OcrExtractor       ocr_extractor.py
    DocxExtractor      docx_extractor.py
    TextExtractor      text_extractor.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from core.types import ExtractionResult


@runtime_checkable
class BaseExtractor(Protocol):
    """
    Protocol (interface) for all document text extractors.

    The @runtime_checkable decorator allows isinstance() checks,
    which DocumentRouter uses to validate registered extractors.
    """

    def extract(self, path: Path) -> ExtractionResult:
        """
        Extract all text from the document at `path`.

        Parameters
        ----------
        path : Path
            Absolute path to the source document.

        Returns
        -------
        ExtractionResult
            Contains raw_text, extraction method, page_count, and metadata.

        Raises
        ------
        ExtractionError
            If the extractor cannot produce text from the document.
            Implementors should wrap lower-level exceptions in ExtractionError
            and attach structured context (path, page number, etc.).
        OCRError
            Only from OcrExtractor: if Tesseract fails at the OS level.

        IMPLEMENTATION CONTRACT
        -----------------------
        - Must always return an ExtractionResult (never None, never "")
        - raw_text may contain artefacts — TextCleaner handles normalisation
        - Must log extraction start, completion, and any per-page warnings
        - Must NOT swallow exceptions silently
        """
        ...

    def can_handle(self, path: Path) -> bool:
        """
        Return True if this extractor is capable of processing `path`.

        Used by DocumentRouter to select the correct extractor.
        Should be a fast, cheap check (MIME type or extension only).
        Must NOT open or read the file content.

        Parameters
        ----------
        path : Path
            Path to the candidate file (may not exist yet — check only type).

        Returns
        -------
        bool
            True  → this extractor should be tried for this file.
            False → skip this extractor, try the next one.

        IMPLEMENTATION CONTRACT
        -----------------------
        - Must be synchronous and near-instant (no I/O)
        - Must not raise exceptions (return False on unknown input)
        - Extension check alone is sufficient; MIME check is a bonus
        """
        ...
