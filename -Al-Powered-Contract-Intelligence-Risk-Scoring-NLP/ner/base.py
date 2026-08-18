"""
ner/base.py
============
BaseNERModel Protocol — the contract all NER model classes must implement.

PURPOSE
-------
Defines the interface that the API layer (Phase 3) and risk scorer (Phase 4)
depend on. Phase 2's transformer model and Phase 1's spaCy model both
implement this same interface.

BENEFIT
-------
Phase 3's API handler imports BaseNERModel, not NERModel or TransformerNERModel.
Swapping the underlying model is a one-line config change, not a code change.

PROTOCOL DEFINITION
-------------------
Any class that implements:
    - extract_entities(text: str) → list[Entity]
    - batch_extract(texts: list[str]) → list[list[Entity]]
    - model_info() → dict
satisfies BaseNERModel at type-check time without explicit inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.types import Entity


@runtime_checkable
class BaseNERModel(Protocol):
    """
    Protocol (interface) for all NER model implementations.

    Implementors:
        ner/inference.py → NERModel (spaCy)
        Phase 2 → TransformerNERModel (RoBERTa-legal)
    """

    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract named entities from a single text string.

        Parameters
        ----------
        text : str
            Clean contract text (should be passed through TextCleaner first).
            May be up to max_text_length chars; longer texts are chunked internally.

        Returns
        -------
        list[Entity]
            Entities sorted by start position.
            May be empty list if no entities found (never None).

        Raises
        ------
        InferenceError
            If the model encounters a runtime failure.
            Must never propagate lower-level spaCy exceptions directly.

        CONTRACT
        --------
        - Returned entities must be non-overlapping
        - score field should be populated where the model provides confidence
        - Calling with empty string → returns [] (no error)
        """
        ...

    def batch_extract(self, texts: list[str]) -> list[list[Entity]]:
        """
        Extract entities from multiple texts efficiently.

        Parameters
        ----------
        texts : list[str]
            List of clean contract text strings.

        Returns
        -------
        list[list[Entity]]
            One list of Entity per input text, in the same order.
            Result length == len(texts) always.

        PERFORMANCE CONTRACT
        --------------------
        Implementations should use the model's native batching (e.g.,
        spaCy's nlp.pipe()) for throughput. Do not implement as a loop
        over extract_entities() unless the model has no batch API.
        """
        ...

    def model_info(self) -> dict:
        """
        Return metadata about the loaded model.

        Returns
        -------
        dict
            {
                "model_path": str,
                "spacy_version": str | None,
                "labels": list[str],   # all entity labels the model knows
                "loaded_at": str,      # ISO timestamp
            }

        Used by the health-check endpoint in Phase 3.
        """
        ...
