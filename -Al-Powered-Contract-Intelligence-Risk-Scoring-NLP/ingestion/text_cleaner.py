"""
ingestion/text_cleaner.py
==========================
Post-extraction normalisation — runs after any extractor.

CLEANING STEPS (from spec §2.5)
---------------------------------
1. Ligature fix       — ﬁ→fi, ﬂ→fl, ﬃ→ffi, ﬀ→ff, ﬄ→ffl, etc.
2. Whitespace         — collapse multiple blank lines, trim line-trailing spaces
3. Header/footer removal — heuristic: short lines (< 60 chars) that appear
                           on 3+ pages are treated as running headers/footers
4. Encoding artefacts — replace common mojibake sequences
5. Metadata returned  — {pages, word_count, extraction_method}

DESIGN
------
- Stateless: TextCleaner.clean() is a pure function
- Non-destructive for legal text: does NOT lowercase, does NOT remove
  punctuation (amounts, dates, section refs must be preserved exactly)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core.types import ExtractionResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Ligature table  (Unicode typographic ligatures → ASCII equivalents)
# ─────────────────────────────────────────────────────────────────────────────

LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\ufb05": "st",   # ﬅ (long-s t)
    "\ufb06": "st",   # ﬆ
    "\u00e6": "ae",   # æ  (common in older legal text)
    "\u0153": "oe",   # œ
    "\u2019": "'",    # right single quotation mark → apostrophe
    "\u2018": "'",    # left single quotation mark
    "\u201c": '"',    # left double quotation mark
    "\u201d": '"',    # right double quotation mark
    "\u2013": "-",    # en dash
    "\u2014": "--",   # em dash
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",    # non-breaking space
    "\u200b": "",     # zero-width space
}

# Header/footer heuristics
MAX_HEADER_FOOTER_LEN:  int = 60   # lines shorter than this are candidates
MIN_PAGES_FOR_PATTERN:  int = 3    # must appear on at least this many pages
MIN_PAGES_TO_USE_HEURISTIC: int = 4  # don't bother on very short docs


@dataclass
class CleanResult:
    """Result of a TextCleaner.clean() call."""
    text:               str
    word_count:         int
    char_count:         int
    ligatures_replaced: int
    lines_removed:      int        # header/footer lines removed
    extraction_method:  str        # from ExtractionResult.method


class TextCleaner:
    """
    Post-extraction text normalisation.

    All methods are @staticmethod — no instantiation needed.

    Usage
    -----
        cleaned = TextCleaner.clean(extraction_result)
        print(cleaned.text[:300])
    """

    @staticmethod
    def clean(result: ExtractionResult) -> CleanResult:
        """
        Run the full cleaning pipeline on an ExtractionResult.

        Steps:
          1. Fix ligatures and encoding artefacts
          2. Collapse excessive whitespace
          3. Remove running headers/footers (heuristic)

        Parameters
        ----------
        result : ExtractionResult

        Returns
        -------
        CleanResult
        """
        text = result.raw_text

        # Step 1 — Ligature + encoding fix
        text, ligature_count = TextCleaner._fix_ligatures(text)

        # Step 2 — Whitespace normalisation
        text = TextCleaner._normalise_whitespace(text)

        # Step 3 — Header/footer removal (only for multi-page docs)
        lines_removed = 0
        if result.page_count >= MIN_PAGES_TO_USE_HEURISTIC:
            text, lines_removed = TextCleaner._remove_headers_footers(
                text, result.page_count
            )

        word_count = len(text.split())

        logger.debug(
            "TextCleaner: ligatures=%d removed_lines=%d words=%d",
            ligature_count, lines_removed, word_count,
        )

        return CleanResult(
            text=text,
            word_count=word_count,
            char_count=len(text),
            ligatures_replaced=ligature_count,
            lines_removed=lines_removed,
            extraction_method=result.method.value,
        )

    @staticmethod
    def clean_text(raw_text: str, page_count: int = 1) -> str:
        """
        Convenience: clean a raw string directly (no ExtractionResult needed).

        Used by run_preprocessing.py and tests.
        """
        from core.types import ExtractionMethod
        dummy = ExtractionResult(
            source_path="",
            raw_text=raw_text,
            method=ExtractionMethod.TXT_READ,
            page_count=page_count,
            metadata={},
        )
        return TextCleaner.clean(dummy).text

    # ── Step 1: Ligatures ─────────────────────────────────────────────────

    @staticmethod
    def _fix_ligatures(text: str) -> tuple[str, int]:
        """
        Replace Unicode ligatures and typographic characters with ASCII.

        Returns
        -------
        (cleaned_text, number_of_replacements)
        """
        count = 0
        for lig, replacement in LIGATURE_MAP.items():
            occurrences = text.count(lig)
            if occurrences:
                text   = text.replace(lig, replacement)
                count += occurrences

        # Also fix common mojibake patterns
        mojibake = {
            "\u00e2\u20ac\u2122": "'",    # â€™ → '
            "\u00e2\u20ac\u0153": '"',    # â€œ → "
            "\u00e2\u20ac\u009d": '"',    # â€  → "
            "\u00e2\u20ac\u201c": "--",   # â€" → --
            "\u00e2\u20ac\u201d": "-",    # â€" → -
            "\u00c2 ": " ",               # Â   → space
            "\u00c2":  "",                # Â   → empty
        }
        for bad, good in mojibake.items():
            occurrences = text.count(bad)
            if occurrences:
                text   = text.replace(bad, good)
                count += occurrences

        return text, count

    # ── Step 2: Whitespace ────────────────────────────────────────────────

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        """
        Collapse excessive whitespace without removing meaningful structure.

        Rules:
          - Strip trailing spaces from each line
          - Collapse 3+ consecutive blank lines into 2
          - Normalise Windows line endings (\\r\\n → \\n)
          - Strip leading/trailing whitespace from the whole document
        """
        # Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]

        # Collapse 3+ consecutive blank lines into 2
        normalised: list[str] = []
        blank_run = 0
        for line in lines:
            if line == "":
                blank_run += 1
                if blank_run <= 2:
                    normalised.append(line)
            else:
                blank_run = 0
                normalised.append(line)

        return "\n".join(normalised).strip()

    # ── Step 3: Headers / footers ─────────────────────────────────────────

    @staticmethod
    def _remove_headers_footers(text: str, page_count: int) -> tuple[str, int]:
        """
        Remove running page headers and footers.

        Heuristic:
          - Split text at page markers ("--- PAGE N ---")
          - Collect the first 3 lines and last 3 lines of each page
          - Any short line (< MAX_HEADER_FOOTER_LEN chars) that appears
            on MIN_PAGES_FOR_PATTERN or more pages is a header/footer candidate
          - Remove all occurrences of those lines

        Parameters
        ----------
        text : str
            Already whitespace-normalised text with page markers.
        page_count : int
            Number of pages (used to calibrate the threshold).

        Returns
        -------
        tuple[str, int]
            (cleaned_text, number_of_lines_removed)
        """
        pages = re.split(r"\n*--- PAGE \d+ ---\n*", text)
        pages = [p for p in pages if p.strip()]

        if len(pages) < MIN_PAGES_FOR_PATTERN:
            return text, 0

        candidate_lines: list[str] = []
        for page in pages:
            page_lines = [l for l in page.split("\n") if l.strip()]
            # First 3 + last 3 lines
            candidates = page_lines[:3] + page_lines[-3:]
            for line in candidates:
                if 0 < len(line.strip()) < MAX_HEADER_FOOTER_LEN:
                    candidate_lines.append(line.strip())

        # Count frequency
        freq      = Counter(candidate_lines)
        threshold = max(MIN_PAGES_FOR_PATTERN, page_count // 3)
        bad_lines = {line for line, count in freq.items() if count >= threshold}

        if not bad_lines:
            return text, 0

        removed = 0
        clean_lines: list[str] = []
        for line in text.split("\n"):
            if line.strip() in bad_lines:
                removed += 1
            else:
                clean_lines.append(line)

        logger.debug(
            "Header/footer heuristic removed %d lines (%d unique patterns)",
            removed, len(bad_lines),
        )
        return "\n".join(clean_lines), removed
