from __future__ import annotations

from pathlib import Path

from core.exceptions import ExtractionError  # noqa: F401
from core.logging import get_logger
from core.types import ExtractionMethod, ExtractionResult

log = get_logger(__name__)


class TextExtractor:
    """
    Reads plain text files with multi-encoding fallback.

    Implements the BaseExtractor Protocol.
    Also used as the fallback extractor in DocumentRouter.

    THREAD SAFETY
    -------------
    Stateless — safe to share across threads.
    """

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".text", ".rst"})
    _ENCODING_FALLBACK_CHAIN: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin-1")

    def can_handle(self, path: Path) -> bool:
        """Return True for .txt, .md, .text, .rst extensions."""
        # TODO: return path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        pass

    def extract(self, path: Path) -> ExtractionResult:
        """
        Read the file content using the encoding fallback chain.

        Algorithm
        ---------
        1. For each encoding in _ENCODING_FALLBACK_CHAIN:
            a. Attempt path.read_text(encoding=enc)
            b. On success: record encoding, break loop
            c. On UnicodeDecodeError: try next encoding
        2. If all encodings fail: raise ExtractionError
        3. Count lines in the decoded text
        4. Return ExtractionResult

        Parameters
        ----------
        path : Path

        Returns
        -------
        ExtractionResult
            raw_text: file contents as-is
            method: ExtractionMethod.TXT_READ
            page_count: -1 (not applicable)
            metadata: {encoding_used, file_size_bytes, line_count}

        Raises
        ------
        ExtractionError
            If path does not exist or all encodings fail.
        """
        # TODO (implementation)
        pass