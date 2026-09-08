"""
data_processing/
================
CUAD dataset processing pipeline — raw dataset → training artifacts.

PURPOSE
-------
Transforms the raw CUAD (Contract Understanding Atticus Dataset) into two
training artifact formats:

    1. spaCy DocBin  (.spacy binary) → consumed by ner/train.py
    2. Clause JSON   (.json)         → consumed by Phase-2 transformer fine-tuning

PIPELINE OVERVIEW
-----------------
    HuggingFace CUAD dataset  (or local CUADv1.json)
            │
            ▼
    CuadLoader.load()              → (train_samples, dev_samples)
            │
            ├──▶ CuadToNer.convert()         → cuad_ner_train.spacy / cuad_ner_dev.spacy
            │         └── SpanValidator      → cleaned, non-overlapping NER spans
            │
            └──▶ CuadToClassification.convert()
                        └── cuad_clauses_train.json / cuad_clauses_dev.json

PUBLIC API
----------
    from data_processing import load_cuad, build_ner_corpus, build_clause_corpus
"""

from data_processing.cuad_loader import load_cuad, CuadLoader
from data_processing.cuad_to_ner import CuadToNer
from data_processing.cuad_to_classification import CuadToClassification
from data_processing.span_validator import SpanValidator, Entity, SpanConflict


def build_ner_corpus(
    train_samples: list[dict],
    dev_samples:   list[dict],
    output_dir:    str = "data/processed",
) -> dict:
    """Convert CUAD samples → spaCy DocBin (.spacy files)."""
    return CuadToNer().convert(train_samples, dev_samples, output_dir)


def build_clause_corpus(
    train_samples: list[dict],
    dev_samples:   list[dict],
    output_dir:    str = "data/processed",
) -> dict:
    """Convert CUAD samples → clause classification JSON Lines."""
    return CuadToClassification().convert(train_samples, dev_samples, output_dir)


__all__ = [
    "load_cuad",
    "CuadLoader",
    "CuadToNer",
    "CuadToClassification",
    "SpanValidator",
    "Entity",
    "SpanConflict",
    "build_ner_corpus",
    "build_clause_corpus",
]
