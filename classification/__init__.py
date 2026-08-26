"""Clause classification package (Phase 2).

Exports:
    load_classifier    - load a fine-tuned model + calibrator bundle
    classify_clauses    - run inference on raw contract text
"""
from .inference import load_classifier, classify_clauses  # noqa: F401

__all__ = ["load_classifier", "classify_clauses"]
