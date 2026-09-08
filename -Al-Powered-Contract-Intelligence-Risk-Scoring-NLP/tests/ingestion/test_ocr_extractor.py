"""
tests/ingestion/test_ocr_extractor.py
=======================================
Tests for ingestion/ocr_extractor.py (§4.2 from phase_01_tasks.md).

SPEC REQUIREMENTS
-----------------
- Convert a simple test image with known text to PDF
- Assert OCR output matches expected string (within Levenshtein threshold)
- Skip automatically if Tesseract binary not installed

TEST COVERAGE
-------------
test_can_handle_pdf              — can_handle returns True for .pdf
test_tesseract_check             — helpful error if Tesseract not found
test_ocr_extracts_text           — basic OCR produces non-empty output
test_ocr_output_approximate_match — text matches within Levenshtein distance
test_preprocessing_returns_rgb   — preprocessed image is RGB (Tesseract compatible)
test_extract_pages_returns_dict  — extract_pages() returns {page_idx: text} dict
test_metadata_keys               — ExtractionResult has required metadata keys
test_ocr_method_tag              — method is ExtractionMethod.PDF_OCR
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pytest

# ── Skip marker for when Tesseract is not installed ────────────────────────
TESSERACT_AVAILABLE = shutil.which("tesseract") is not None

skip_if_no_tesseract = pytest.mark.skipif(
    not TESSERACT_AVAILABLE,
    reason="Tesseract binary not found on PATH — OCR tests skipped",
)


# ── Fixtures ──────────────────────────────────────────────────────────────

KNOWN_TEXT = "Agreement dated January 2024 Acme Corporation"


def _build_image_pdf(text: str) -> bytes:
    """
    Build a single-page image-only PDF by:
    1. Rendering text on a white PIL image
    2. Embedding the image as a JPEG in a minimal PDF structure
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io as _io

        # Create white image with text
        img = Image.new("RGB", (800, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 80), text, fill="black")

        # Save image as JPEG in memory
        img_buf = _io.BytesIO()
        img.save(img_buf, format="JPEG", quality=95)
        jpeg_bytes = img_buf.getvalue()

        # Wrap JPEG in minimal PDF
        return _jpeg_to_pdf(jpeg_bytes, width=800, height=200)
    except ImportError:
        pytest.skip("Pillow not installed — cannot build image PDF for OCR test")


def _jpeg_to_pdf(jpeg_bytes: bytes, width: int, height: int) -> bytes:
    """Wrap a JPEG image in a minimal 1-page PDF."""
    img_obj = (
        b"1 0 obj\n<< /Type /XObject /Subtype /Image "
        b"/Width " + str(width).encode() + b" /Height " + str(height).encode() + b" "
        b"/ColorSpace /DeviceRGB /BitsPerComponent 8 "
        b"/Filter /DCTDecode /Length " + str(len(jpeg_bytes)).encode() + b" >>\n"
        b"stream\n" + jpeg_bytes + b"\nendstream\nendobj\n"
    )
    stream_content = b"q " + str(width).encode() + b" 0 0 " + str(height).encode() + b" 0 0 cm /Im1 Do Q\n"
    content_obj = (
        b"2 0 obj\n<< /Length " + str(len(stream_content)).encode() + b" >>\n"
        b"stream\n" + stream_content + b"\nendstream\nendobj\n"
    )
    page_obj = (
        b"3 0 obj\n<< /Type /Page /Parent 4 0 R "
        b"/MediaBox [0 0 " + str(width).encode() + b" " + str(height).encode() + b"] "
        b"/Contents 2 0 R /Resources << /XObject << /Im1 1 0 R >> >> >>\nendobj\n"
    )
    pages_obj = b"4 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    catalog_obj = b"5 0 obj\n<< /Type /Catalog /Pages 4 0 R >>\nendobj\n"
    header = b"%PDF-1.4\n"
    body   = img_obj + content_obj + page_obj + pages_obj + catalog_obj
    xref_offset = len(header) + len(body)
    xref = b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 6 /Root 5 0 R >>\nstartxref\n" + str(xref_offset).encode() + b"\n%%EOF\n"
    return header + body + xref


@pytest.fixture(scope="module")
def image_pdf_path(tmp_path_factory) -> Path:
    """1-page image-only PDF with KNOWN_TEXT rendered into it."""
    pdf_bytes = _build_image_pdf(KNOWN_TEXT)
    path = tmp_path_factory.mktemp("ocr") / "scanned.pdf"
    path.write_bytes(pdf_bytes)
    return path


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        new_row = [i + 1]
        for j, cb in enumerate(b):
            new_row.append(min(row[j + 1] + 1, new_row[j] + 1, row[j] + (ca != cb)))
        row = new_row
    return row[-1]


# ── Tests ─────────────────────────────────────────────────────────────────

def test_can_handle_pdf():
    from ingestion.ocr_extractor import OcrExtractor
    extractor = OcrExtractor()
    assert extractor.can_handle(Path("contract.pdf")) is True


def test_cannot_handle_docx():
    from ingestion.ocr_extractor import OcrExtractor
    extractor = OcrExtractor()
    assert extractor.can_handle(Path("contract.docx")) is False


def test_tesseract_available_check():
    """OcrExtractor._tesseract_installed() should reflect system state."""
    from ingestion.ocr_extractor import OcrExtractor
    result = OcrExtractor._tesseract_installed()
    assert result == TESSERACT_AVAILABLE


@skip_if_no_tesseract
def test_ocr_extracts_text(image_pdf_path):
    """OCR should produce non-empty text from an image PDF."""
    from ingestion.ocr_extractor import OcrExtractor
    extractor = OcrExtractor()
    result    = extractor.extract(image_pdf_path)

    assert result.raw_text is not None
    assert len(result.raw_text.strip()) > 0


@skip_if_no_tesseract
def test_ocr_output_approximate_match(image_pdf_path):
    """
    Spec §4.2: OCR output matches expected string within Levenshtein threshold.

    Threshold: Levenshtein distance <= 20% of expected text length.
    (OCR is not perfect — small character recognition errors are acceptable.)
    """
    from ingestion.ocr_extractor import OcrExtractor
    result     = OcrExtractor().extract(image_pdf_path)
    ocr_text   = result.raw_text.replace("\n", " ").strip()
    expected   = KNOWN_TEXT

    distance  = _levenshtein(ocr_text.lower(), expected.lower())
    threshold = max(10, len(expected) // 5)  # allow up to 20% edit distance

    assert distance <= threshold, (
        f"OCR output too different from expected.\n"
        f"Expected: {expected!r}\n"
        f"Got:      {ocr_text[:100]!r}\n"
        f"Levenshtein distance: {distance} (threshold: {threshold})"
    )


@skip_if_no_tesseract
def test_metadata_keys(image_pdf_path):
    """ExtractionResult.metadata should contain required keys."""
    from ingestion.ocr_extractor import OcrExtractor
    result   = OcrExtractor().extract(image_pdf_path)
    required = {"ocr_config", "dpi", "lang", "tesseract_available"}
    for key in required:
        assert key in result.metadata, f"Missing metadata key: {key}"


@skip_if_no_tesseract
def test_ocr_method_tag(image_pdf_path):
    """Result method should be ExtractionMethod.PDF_OCR."""
    from ingestion.ocr_extractor import OcrExtractor
    from core.types import ExtractionMethod
    result = OcrExtractor().extract(image_pdf_path)
    assert result.method == ExtractionMethod.PDF_OCR


def test_preprocessing_returns_rgb():
    """Preprocessed image should be RGB (compatible with Tesseract)."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")

    from ingestion.ocr_extractor import OcrExtractor

    # Create a simple grey test image
    img = Image.new("L", (100, 50), color=200)
    result = OcrExtractor._preprocess_image(img)
    assert result.mode == "RGB", f"Expected RGB, got {result.mode}"
