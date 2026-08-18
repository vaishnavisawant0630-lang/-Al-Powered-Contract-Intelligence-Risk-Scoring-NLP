"""
ingestion/document_router.py
==============================
Routes any input document to the correct extractor.

ROUTING LOGIC (from spec §2.4)
---------------------------------
1. Detect filetype via magic bytes first, then file extension
2. Route to correct extractor:
     .pdf  → PdfExtractor (digital) → if scanned pages flagged → OcrExtractor for those pages
     .docx → DocxExtractor
     .txt  → TextExtractor
     other → raise IngestionError
3. For PDFs with mixed pages:
     a. Run PdfExtractor on the full file
     b. If metadata["scanned_pages"] is non-empty, run OcrExtractor.extract_pages()
        for those specific pages and splice the OCR text back in
4. Always run TextCleaner on the result before returning

MAGIC BYTES
-----------
  PDF:  b"%PDF"  (first 4 bytes)
  DOCX: b"PK\x03\x04" (ZIP header — .docx/.xlsx/.pptx are all ZIP)
  Fallback to file extension if magic bytes unrecognised.
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.types import ExtractionMethod, ExtractionResult

logger = logging.getLogger(__name__)

# Magic byte signatures
_MAGIC_PDF  = b"%PDF"
_MAGIC_ZIP  = b"PK\x03\x04"   # ZIP/DOCX

# Min bytes to read for magic detection
_MAGIC_READ_BYTES = 8


class DocumentRouter:
    """
    Routes a document file to the correct extractor.

    Usage
    -----
        router = DocumentRouter()
        result = router.route("contracts/acme.pdf")
        print(result.raw_text[:500])
        print(result.method)           # ExtractionMethod.PDF_DIRECT or PDF_OCR
    """

    def __init__(self) -> None:
        from ingestion.pdf_extractor  import PdfExtractor
        from ingestion.ocr_extractor  import OcrExtractor
        from ingestion.docx_extractor import DocxExtractor
        from ingestion.text_extractor import TextExtractor
        from ingestion.text_cleaner   import TextCleaner

        self._pdf_extractor  = PdfExtractor()
        self._ocr_extractor  = OcrExtractor()
        self._docx_extractor = DocxExtractor()
        self._txt_extractor  = TextExtractor()
        self._cleaner        = TextCleaner()

    # ── Public API ────────────────────────────────────────────────────────

    def route(self, path: str | Path, clean: bool = True) -> ExtractionResult:
        """
        Extract text from any supported document.

        Parameters
        ----------
        path : str | Path
            Path to the document file.
        clean : bool
            If True (default), run TextCleaner after extraction.

        Returns
        -------
        ExtractionResult
            raw_text contains the extracted (and optionally cleaned) text.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file type is not supported.
        RuntimeError
            If extraction fails at a lower level.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        doc_type = self._detect_type(path)
        logger.info("DocumentRouter: %s → %s", path.name, doc_type)

        result = self._extract(path, doc_type)

        if clean:
            from ingestion.text_cleaner import TextCleaner, CleanResult
            clean_result: CleanResult = TextCleaner.clean(result)
            # Replace raw_text with cleaned text
            result = ExtractionResult(
                source_path=result.source_path,
                raw_text=clean_result.text,
                method=result.method,
                page_count=result.page_count,
                metadata={
                    **result.metadata,
                    "word_count":         clean_result.word_count,
                    "ligatures_replaced": clean_result.ligatures_replaced,
                    "headers_removed":    clean_result.lines_removed,
                },
            )

        logger.info(
            "DocumentRouter: done — method=%s chars=%d",
            result.method.value, len(result.raw_text),
        )
        return result

    # ── Type detection ────────────────────────────────────────────────────

    def _detect_type(self, path: Path) -> str:
        """
        Detect document type using magic bytes, then file extension.

        Returns
        -------
        str — one of: "pdf", "docx", "txt"

        Raises
        ------
        ValueError
            If the type cannot be determined.
        """
        # Magic bytes (most reliable)
        try:
            with open(path, "rb") as f:
                magic = f.read(_MAGIC_READ_BYTES)

            if magic[:4] == _MAGIC_PDF:
                return "pdf"
            if magic[:4] == _MAGIC_ZIP:
                # Could be .docx, .xlsx, etc. — trust the extension
                suffix = path.suffix.lower()
                if suffix in {".docx", ".doc"}:
                    return "docx"
                # Other ZIP-based formats not supported
        except OSError:
            pass   # Fall through to extension-based detection

        # Extension fallback
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".docx", ".doc"}:
            return "docx"
        if suffix in {".txt", ".text", ".md"}:
            return "txt"

        raise ValueError(
            f"Unsupported document type: {path.suffix!r} ({path.name})\n"
            f"Supported: .pdf, .docx, .doc, .txt, .text, .md"
        )

    # ── Extraction dispatch ───────────────────────────────────────────────

    def _extract(self, path: Path, doc_type: str) -> ExtractionResult:
        """Dispatch to the correct extractor(s)."""
        if doc_type == "pdf":
            return self._extract_pdf(path)
        if doc_type == "docx":
            return self._docx_extractor.extract(path)
        if doc_type == "txt":
            return self._txt_extractor.extract(path)
        raise ValueError(f"Unknown doc_type: {doc_type!r}")

    def _extract_pdf(self, path: Path) -> ExtractionResult:
        """
        Extract a PDF with hybrid PDF-direct + OCR fallback.

        Algorithm:
          1. Run PdfExtractor on the full file
          2. Check metadata["scanned_pages"]
          3. If any scanned pages: run OcrExtractor.extract_pages() for those
          4. Splice OCR text back into the full text at the page markers
        """
        # Step 1: digital extraction
        result = self._pdf_extractor.extract(path)
        scanned_pages: list[int] = result.metadata.get("scanned_pages", [])

        if not scanned_pages:
            # Fully digital PDF — done
            return result

        logger.info(
            "PDF has %d scanned page(s) — running OCR for pages %s",
            len(scanned_pages), scanned_pages,
        )

        # Step 2: OCR just the scanned pages
        try:
            ocr_page_map = self._ocr_extractor.extract_pages(path, scanned_pages)
        except RuntimeError as exc:
            # Tesseract not installed — log warning and return digital-only result
            logger.warning(
                "OCR unavailable (%s). Returning digital-only extraction for %s.",
                exc, path.name,
            )
            return result

        # Step 3: Splice OCR text back at page markers
        full_text = result.raw_text
        for page_idx, ocr_text in ocr_page_map.items():
            marker = f"--- PAGE {page_idx + 1} ---"
            if marker in full_text:
                # Replace the (nearly empty) page text with OCR output
                parts = full_text.split(marker, 1)
                if len(parts) == 2:
                    # Find end of this page (next marker or EOF)
                    next_marker_match = __import__("re").search(
                        r"--- PAGE \d+ ---", parts[1]
                    )
                    if next_marker_match:
                        before_next = parts[1][: next_marker_match.start()]
                        after_next  = parts[1][next_marker_match.start():]
                        full_text   = (
                            parts[0] + marker
                            + f"\n\n{ocr_text}\n\n"
                            + after_next
                        )
                    else:
                        full_text = parts[0] + marker + f"\n\n{ocr_text}\n\n"

        return ExtractionResult(
            source_path=result.source_path,
            raw_text=full_text,
            method=ExtractionMethod.PDF_OCR,   # Mixed — report as OCR
            page_count=result.page_count,
            metadata={
                **result.metadata,
                "ocr_pages_processed": list(ocr_page_map.keys()),
                "hybrid_extraction":   True,
            },
        )
