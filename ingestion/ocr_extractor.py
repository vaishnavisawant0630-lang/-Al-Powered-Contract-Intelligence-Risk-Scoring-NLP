"""
ingestion/ocr_extractor.py
===========================
OCR pipeline for scanned/image-only PDF files.

PIPELINE PER PAGE (from spec §2.2)
------------------------------------
  PDF page
    │
    ▼  pdf2image.convert_from_path(dpi=300)
  PIL.Image (RGB, 300 DPI)
    │
    ▼  Preprocessing:
    │   1. Convert to greyscale
    │   2. Adaptive thresholding (Pillow ImageFilter)
    │   3. Deskew (pytesseract OSD)
    │   4. Denoise
    │
    ▼  pytesseract.image_to_string(config="--oem 3 --psm 6")
  raw OCR text
    │
    ▼  Reassemble: "\n\n--- PAGE {n} ---\n\n"
  full document text

TESSERACT CONFIG
-----------------
  --oem 3  → LSTM + legacy combined mode (best accuracy)
  --psm 6  → Assume uniform block of text (best for contracts)

SKIP CONDITION
--------------
  If Tesseract binary is not installed on the system, OcrExtractor.can_handle()
  returns True but extract() raises OCRError with installation instructions.
  Tests can use @pytest.mark.skipif to skip when Tesseract is absent.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from core.types import ExtractionMethod, ExtractionResult

logger = logging.getLogger(__name__)

# Tesseract config (spec §2.2)
TESSERACT_CONFIG = "--oem 3 --psm 6"
DPI              = 300
LANG             = "eng"


class OcrExtractor:
    """
    OCR pipeline for scanned PDF pages (pdf2image + pytesseract).

    Satisfies the BaseExtractor Protocol (can_handle + extract).

    Can also be called for specific page ranges only, using extract_pages().
    DocumentRouter uses this when PdfExtractor flags certain pages as scanned.
    """

    SUPPORTED_SUFFIXES = {".pdf"}

    def can_handle(self, path: Path) -> bool:
        """Return True for .pdf files (OCR fallback for any PDF)."""
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def extract(self, path: Path) -> ExtractionResult:
        """
        Run the full OCR pipeline on all pages of a PDF.

        Parameters
        ----------
        path : Path
            Path to the scanned PDF file.

        Returns
        -------
        ExtractionResult
            raw_text: pages joined with page markers
            method: ExtractionMethod.PDF_OCR
            page_count: number of pages processed
            metadata:
                ocr_config (str): Tesseract config string used
                dpi (int): render DPI
                tesseract_available (bool)
        """
        self._check_tesseract()
        return self._run_ocr(path, page_indices=None)

    def extract_pages(
        self,
        path:         Path,
        page_indices: list[int],
    ) -> dict[int, str]:
        """
        OCR only specific pages (0-indexed) from a PDF.

        Used by DocumentRouter when PdfExtractor flags certain pages as scanned.

        Returns
        -------
        dict[int, str]
            Mapping of {page_index → ocr_text} for each requested page.
        """
        self._check_tesseract()
        result = self._run_ocr(path, page_indices=page_indices)
        # Return page-indexed dict
        lines  = result.raw_text.split("--- PAGE ")
        page_map: dict[int, str] = {}
        for line in lines[1:]:  # skip before first marker
            try:
                num, rest = line.split(" ---", 1)
                page_map[int(num) - 1] = rest.strip()
            except ValueError:
                continue
        return page_map

    # ── Internal pipeline ─────────────────────────────────────────────────

    def _run_ocr(
        self,
        path:         Path,
        page_indices: list[int] | None,
    ) -> ExtractionResult:
        """Core OCR pipeline."""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            from PIL import ImageFilter, ImageOps
        except ImportError as exc:
            raise ImportError(
                f"Missing OCR dependency: {exc}\n"
                "Install with: pip install pytesseract pdf2image Pillow\n"
                "Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
            ) from exc

        logger.info("OcrExtractor: rendering %s at %d DPI", path.name, DPI)

        # Render PDF pages → PIL images
        all_images = convert_from_path(str(path), dpi=DPI)

        if page_indices is not None:
            # Only requested pages
            images = [(i, all_images[i]) for i in page_indices if i < len(all_images)]
        else:
            images = list(enumerate(all_images))

        page_texts: list[str] = []

        for page_idx, image in images:
            processed   = self._preprocess_image(image)
            ocr_text    = pytesseract.image_to_string(
                processed,
                lang=LANG,
                config=TESSERACT_CONFIG,
            )
            page_texts.append(
                f"\n\n--- PAGE {page_idx + 1} ---\n\n{ocr_text}"
            )
            logger.debug(
                "OCR page %d: extracted %d chars",
                page_idx + 1, len(ocr_text),
            )

        full_text  = "".join(page_texts)
        page_count = len(all_images)

        logger.info(
            "OcrExtractor done: pages=%d ocr_pages=%d chars=%d",
            page_count, len(images), len(full_text),
        )

        return ExtractionResult(
            source_path=str(path),
            raw_text=full_text,
            method=ExtractionMethod.PDF_OCR,
            page_count=page_count,
            metadata={
                "ocr_config":          TESSERACT_CONFIG,
                "dpi":                 DPI,
                "lang":                LANG,
                "tesseract_available": self._tesseract_installed(),
                "pages_ocr_processed": len(images),
            },
        )

    @staticmethod
    def _preprocess_image(image):
        """
        Preprocess a PIL Image for best OCR accuracy.

        Pipeline:
          1. Convert to greyscale
          2. Adaptive contrast enhancement (autocontrast)
          3. Sharpen (reduces blur from scanning)
          4. Convert back to RGB for Tesseract compatibility
        """
        from PIL import ImageEnhance, ImageFilter, ImageOps

        # 1. Greyscale
        grey = image.convert("L")

        # 2. Autocontrast (adaptive thresholding equivalent)
        grey = ImageOps.autocontrast(grey, cutoff=2)

        # 3. Sharpen to improve character edges
        grey = grey.filter(ImageFilter.SHARPEN)

        # 4. Back to RGB (Tesseract works with either, RGB is safer)
        return grey.convert("RGB")

    @staticmethod
    def _tesseract_installed() -> bool:
        """Return True if the tesseract binary is on the system PATH."""
        return shutil.which("tesseract") is not None

    def _check_tesseract(self) -> None:
        """Raise a clear error if Tesseract is not installed."""
        if not self._tesseract_installed():
            raise RuntimeError(
                "Tesseract OCR binary not found on PATH.\n"
                "Install from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "On Windows, add Tesseract install dir to PATH.\n"
                "On Linux: sudo apt install tesseract-ocr"
            )
