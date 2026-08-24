"""
ingestion/
==========
Document ingestion package — converts raw files into clean text strings.

PURPOSE
-------
This package is responsible for ONE thing: given a file path, return the
text content of that document as a clean Python string.

It does NOT:
    - Do NER, entity extraction, or any ML
    - Store results to disk
    - Know anything about CUAD or training data
    - Modify the file on disk

ARCHITECTURE — Strategy Pattern
--------------------------------
Each file format is handled by a dedicated extractor class that implements
the BaseExtractor protocol (defined in ingestion/base.py).

DocumentRouter inspects the file and delegates to the correct extractor.
New formats are added by: (1) creating a new extractor, (2) registering it
in the router — zero changes to existing extractors.

PUBLIC API (what callers import)
---------------------------------
    from ingestion import DocumentRouter

    router = DocumentRouter()
    result = router.route("contracts/acme.pdf")   # → ExtractionResult
    clean  = TextCleaner.clean(result.raw_text)   # → str

INTERNAL MODULES
----------------
    base.py             BaseExtractor Protocol (the contract all extractors fulfil)
    pdf_extractor.py    Digital PDFs via pdfminer.six
    ocr_extractor.py    Scanned PDFs via pdf2image + pytesseract
    docx_extractor.py   .docx files via python-docx
    text_extractor.py   .txt / .md passthrough
    document_router.py  Routes file → extractor (Strategy pattern)
    text_cleaner.py     Post-extraction text normalisation (pure functions)
"""

from ingestion.document_router import DocumentRouter
from ingestion.text_cleaner import TextCleaner

__all__ = ["DocumentRouter", "TextCleaner"]
