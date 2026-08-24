"""
ingestion/pdf_extractor.py
===========================
Extracts text from digital (text-layer) PDF files using pdfminer.six.

BEHAVIOUR
---------
- Extracts text page by page using pdfminer's high-level API
- Computes character density per page (chars / page area)
- Any page with < MIN_CHAR_DENSITY chars is flagged as likely scanned
- Flagged page numbers are included in ExtractionResult.metadata
  so DocumentRouter can hand them to OcrExtractor for re-processing

USAGE
-----
    extractor = PdfExtractor()
    result = extractor.extract(Path("contracts/acme.pdf"))
    print(result.raw_text[:500])
    print(result.metadata["scanned_pages"])   # list of 0-indexed page numbers
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.types import ExtractionMethod, ExtractionResult

logger = logging.getLogger(__name__)

# Pages with fewer characters than this are treated as scanned/image pages
MIN_CHAR_DENSITY: int = 50


class PdfExtractor:
    """
    Extracts text from digital PDF files via pdfminer.six.

    Satisfies the BaseExtractor Protocol (can_handle + extract).
    """

    SUPPORTED_SUFFIXES = {".pdf"}

    def can_handle(self, path: Path) -> bool:
        """Return True for .pdf files."""
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def extract(self, path: Path) -> ExtractionResult:
        """
        Extract text from a digital PDF.

        Parameters
        ----------
        path : Path
            Path to a .pdf file.

        Returns
        -------
        ExtractionResult
            raw_text: all page texts joined with page markers
            method: ExtractionMethod.PDF_DIRECT
            page_count: number of pages in the document
            metadata:
                scanned_pages (list[int]): 0-indexed page numbers with < MIN_CHAR_DENSITY chars
                char_density_per_page (list[int]): char count per page
                producer (str): PDF producer string if available
        """
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTAnno, LTChar, LTTextBox, LTTextLine
        except ImportError:
            raise ImportError(
                "pdfminer.six is required for PDF text extraction.\n"
                "Install with: pip install pdfminer.six"
            )

        logger.info("PdfExtractor: extracting %s", path.name)

        page_texts:        list[str] = []
        scanned_pages:     list[int] = []
        density_per_page:  list[int] = []

        try:
            for page_idx, page_layout in enumerate(extract_pages(str(path))):
                page_chars: list[str] = []
                for element in page_layout:
                    if isinstance(element, LTTextBox):
                        for line in element:
                            if isinstance(line, LTTextLine):
                                for char in line:
                                    if isinstance(char, (LTChar, LTAnno)):
                                        page_chars.append(char.get_text())

                page_text = "".join(page_chars)
                char_count = len(page_text.strip())
                density_per_page.append(char_count)

                if char_count < MIN_CHAR_DENSITY:
                    scanned_pages.append(page_idx)
                    logger.debug(
                        "Page %d appears scanned (chars=%d < %d)",
                        page_idx, char_count, MIN_CHAR_DENSITY,
                    )

                page_texts.append(f"\n\n--- PAGE {page_idx + 1} ---\n\n{page_text}")

        except Exception as exc:
            raise RuntimeError(
                f"PdfExtractor failed on {path}: {exc}"
            ) from exc

        full_text  = "".join(page_texts)
        page_count = len(page_texts)

        if scanned_pages:
            logger.warning(
                "%s: %d/%d pages appear scanned → flagged for OCR",
                path.name, len(scanned_pages), page_count,
            )

        logger.info(
            "PdfExtractor done: pages=%d scanned=%d chars=%d",
            page_count, len(scanned_pages), len(full_text),
        )

        return ExtractionResult(
            source_path=str(path),
            raw_text=full_text,
            method=ExtractionMethod.PDF_DIRECT,
            page_count=page_count,
            metadata={
                "scanned_pages":        scanned_pages,
                "char_density_per_page": density_per_page,
                "extractor":            "pdfminer.six",
            },
        )
