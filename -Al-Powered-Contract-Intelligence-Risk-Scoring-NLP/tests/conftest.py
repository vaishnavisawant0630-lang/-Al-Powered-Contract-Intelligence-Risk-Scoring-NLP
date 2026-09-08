"""
tests/conftest.py
==================
Shared pytest fixtures for the entire test suite.

PURPOSE
-------
Central location for fixtures that are used across multiple test modules.
Keeps individual test files lean — they import from here via pytest's
automatic conftest discovery.

FIXTURE CATEGORIES
------------------
1. Sample Documents
   - sample_pdf_bytes: minimal valid PDF bytes for testing PdfExtractor
   - sample_scanned_pdf_path: path to a fixture scanned PDF (image only)
   - sample_docx_path: path to a fixture .docx file
   - sample_txt_path: path to a plain text contract fixture

2. CUAD Data Stubs
   - cuad_sample: a single synthetic CUAD-format dict (no network call)
   - cuad_train_samples: 5-sample list for testing CuadLoader splits
   - cuad_ner_sample: a NERSample with known entities for assertion

3. Settings
   - test_settings: overridden Settings with temp directories
   - tmp_output_dir: a tmp_path-based output directory

4. Model Stubs
   - mock_ner_model: MagicMock satisfying BaseNERModel Protocol

FIXTURE FILES LOCATION
-----------------------
    tests/fixtures/
        minimal_digital.pdf       — 1-page PDF with selectable text
        scanned_contract.pdf      — 1-page image-only PDF
        sample_contract.docx      — .docx with 2 paragraphs + 1 table
        sample_contract.txt       — plain text with ligatures and whitespace noise

HOW TO USE IN TESTS
-------------------
    def test_pdf_extractor(sample_pdf_bytes, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(sample_pdf_bytes)
        extractor = PdfExtractor()
        result = extractor.extract(pdf_file)
        assert len(result.raw_text) > 0
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Document fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to the tests/fixtures/ directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def sample_pdf_path() -> Path:
    """
    Path to a minimal digital PDF fixture.
    File: tests/fixtures/minimal_digital.pdf
    Contains: one page of selectable text — "This Agreement..."
    """
    return FIXTURES_DIR / "minimal_digital.pdf"


@pytest.fixture(scope="session")
def sample_scanned_pdf_path() -> Path:
    """
    Path to a scanned (image-only) PDF fixture.
    File: tests/fixtures/scanned_contract.pdf
    Contains: one page rendered as JPEG — no text layer.
    char_density will be < 50 → should trigger OcrExtractor.
    """
    return FIXTURES_DIR / "scanned_contract.pdf"


@pytest.fixture(scope="session")
def sample_docx_path() -> Path:
    """
    Path to a .docx fixture with 2 paragraphs and 1 table.
    File: tests/fixtures/sample_contract.docx
    """
    return FIXTURES_DIR / "sample_contract.docx"


@pytest.fixture(scope="session")
def sample_txt_path() -> Path:
    """
    Path to a plain text contract fixture.
    File: tests/fixtures/sample_contract.txt
    Contains: intentional ligatures (ﬁ), extra whitespace, and en-dashes.
    Used to test TextCleaner in real-world conditions.
    """
    return FIXTURES_DIR / "sample_contract.txt"


# ---------------------------------------------------------------------------
# CUAD data stubs (no network, no disk)
# ---------------------------------------------------------------------------

@pytest.fixture
def cuad_sample() -> dict:
    """
    A single synthetic CUAD-format sample dict.

    Used to test CuadToNer.convert() and CuadToClassification.convert()
    without loading the full dataset.
    """
    return {
        "id": "test_contract_0",
        "title": "SOFTWARE LICENSE AGREEMENT",
        "context": (
            "This Software License Agreement ('Agreement') is entered into as of "
            "January 15, 2024, by and between Acme Corporation, a Delaware corporation "
            "('Licensor'), and Beta Technologies LLC, a California limited liability "
            "company ('Licensee'). The governing law shall be the State of California."
        ),
        "question": "Highlight the parts (if any) of this contract related to governing law.",
        "answers": {
            "text": ["the State of California"],
            "answer_start": [232],
        },
    }


@pytest.fixture
def cuad_train_samples(cuad_sample) -> list[dict]:
    """
    5 synthetic CUAD samples for testing split and conversion logic.
    Constructed from cuad_sample with varied titles (for stratification testing).
    """
    samples = []
    for i in range(5):
        sample = dict(cuad_sample)
        sample["id"] = f"test_contract_{i}"
        sample["title"] = f"Agreement Type {i % 3}"  # 3 unique titles for stratification
        samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_settings(tmp_path) -> object:
    """
    A Settings-like object with all paths pointing to tmp_path.
    Used to prevent tests from reading/writing to real data directories.

    Returns
    -------
    Settings-like object (namespace or MagicMock with required attrs)
    """
    # TODO (implementation): return a configured Settings instance or SimpleNamespace
    settings = MagicMock()
    settings.data_raw_dir = str(tmp_path / "raw")
    settings.data_processed_dir = str(tmp_path / "processed")
    settings.models_dir = str(tmp_path / "models")
    settings.ocr_dpi = 300
    settings.ocr_lang = "eng"
    settings.ocr_char_density_threshold = 50
    settings.cuad_train_split = 0.8
    settings.cuad_random_seed = 42
    settings.max_text_length = 256
    return settings


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ner_model() -> MagicMock:
    """
    A MagicMock satisfying the BaseNERModel Protocol.

    extract_entities() returns two fixed Entity objects.
    Used to test code that depends on BaseNERModel without a real model.
    """
    from core.types import Entity
    mock = MagicMock()
    mock.extract_entities.return_value = [
        Entity(label="ORG", text="Acme Corporation", start=0, end=16, score=0.95),
        Entity(label="GOVERNING_LAW", text="State of California", start=200, end=219),
    ]
    mock.batch_extract.return_value = [[
        Entity(label="ORG", text="Acme Corporation", start=0, end=16, score=0.95),
    ]]
    mock.model_info.return_value = {
        "model_path": "models/test",
        "labels": ["ORG", "DATE", "GOVERNING_LAW"],
        "loaded_at": "2025-08-05T12:00:00Z",
    }
    return mock
