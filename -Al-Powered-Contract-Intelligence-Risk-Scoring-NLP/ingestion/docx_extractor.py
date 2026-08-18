"""
ingestion/docx_extractor.py
=============================
Extracts text from Microsoft Word (.docx) files using python-docx.

EXTRACTION STRATEGY (from spec §2.3)
--------------------------------------
1. Paragraph text — extracted in document order, joined with newlines
2. Table cell text — row by row, cells pipe-delimited ("|")
   Tables appear inline at their document position

READING ORDER
-------------
python-docx exposes the document body as a flat list of blocks that
are either Paragraph or Table objects, in their original document
order. We iterate this list to preserve reading order.

USAGE
-----
    extractor = DocxExtractor()
    result = extractor.extract(Path("contracts/acme.docx"))
    print(result.raw_text[:500])
"""

from __future__ import annotations

import logging
from pathlib import Path

from core.types import ExtractionMethod, ExtractionResult

logger = logging.getLogger(__name__)


class DocxExtractor:
    """
    Extracts text from .docx files using python-docx.

    Satisfies the BaseExtractor Protocol (can_handle + extract).
    """

    SUPPORTED_SUFFIXES = {".docx", ".doc"}

    def can_handle(self, path: Path) -> bool:
        """Return True for .docx and .doc files."""
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def extract(self, path: Path) -> ExtractionResult:
        """
        Extract all text from a .docx file in reading order.

        Parameters
        ----------
        path : Path
            Path to the .docx file.

        Returns
        -------
        ExtractionResult
            raw_text: paragraphs + tables in document order
            method: ExtractionMethod.DOCX_PARSE
            page_count: 0 (page count not available from python-docx)
            metadata:
                paragraph_count (int)
                table_count (int)
                word_count (int)
        """
        try:
            import docx
        except ImportError:
            raise ImportError(
                "python-docx is required for Word document extraction.\n"
                "Install with: pip install python-docx"
            )

        logger.info("DocxExtractor: reading %s", path.name)

        try:
            document = docx.Document(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"DocxExtractor: failed to open {path}: {exc}"
            ) from exc

        blocks:          list[str] = []
        paragraph_count: int       = 0
        table_count:     int       = 0

        # Iterate document body in reading order (paragraphs + tables interleaved)
        for block in document.element.body:
            tag = block.tag.split("}")[-1] if "}" in block.tag else block.tag

            if tag == "p":
                # Paragraph
                para = docx.text.paragraph.Paragraph(block, document)
                text = para.text.strip()
                if text:
                    blocks.append(text)
                    paragraph_count += 1

            elif tag == "tbl":
                # Table — render each row as "| cell1 | cell2 | ... |"
                table = docx.table.Table(block, document)
                table_lines: list[str] = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    # Deduplicate merged cells (python-docx repeats merged cells)
                    deduped = []
                    prev    = None
                    for c in cells:
                        if c != prev:
                            deduped.append(c)
                            prev = c
                    table_lines.append("| " + " | ".join(deduped) + " |")
                if table_lines:
                    blocks.append("\n".join(table_lines))
                    table_count += 1

        full_text  = "\n\n".join(blocks)
        word_count = len(full_text.split())

        logger.info(
            "DocxExtractor done: paragraphs=%d tables=%d words=%d",
            paragraph_count, table_count, word_count,
        )

        return ExtractionResult(
            source_path=str(path),
            raw_text=full_text,
            method=ExtractionMethod.DOCX_PARSE,
            page_count=0,          # python-docx doesn't expose page count
            metadata={
                "paragraph_count": paragraph_count,
                "table_count":     table_count,
                "word_count":      word_count,
                "extractor":       "python-docx",
            },
        )
