"""
data_processing/base.py
========================
BaseConverter Protocol — the contract all converter classes must implement.

PURPOSE
-------
Defines a common interface for CUAD → training-format converters.
CuadToNer and CuadToClassification both implement this protocol.

BENEFIT
-------
The prepare_data.sh script and any orchestration code can iterate over
a list of BaseConverter instances without knowing concrete types.
Adding a new output format (e.g., JSONL for LLM fine-tuning) means
creating a new converter — nothing else changes.

CONTRACT
--------
A converter takes a raw list of CUAD sample dicts and returns
a list of strongly-typed training samples (NERSample or ClauseSample).
It is stateless — the same instance can be called multiple times.

CUAD SAMPLE DICT SCHEMA
------------------------
Each raw CUAD sample looks like:
{
    "id": "CUAD_v1/full_contract_pdf/N-1_4.pdf_0",
    "title": "STRATEGIC ALLIANCE AGREEMENT",
    "context": "This Strategic Alliance Agreement... (full contract text)",
    "question": "Highlight the parts (if any) of this contract related to...",
    "answers": {
        "text": ["Acme Corp", "Beta LLC"],
        "answer_start": [145, 892]
    }
}

The `question` field determines which EntityLabel the answer maps to.
CUAD has 41 distinct question templates, one per clause type.
"""

from __future__ import annotations

from typing import Protocol, Union

from core.types import ClauseSample, NERSample

# Union type for all possible converter outputs
ConversionOutput = list[NERSample] | list[ClauseSample]


class BaseConverter(Protocol):
    """
    Protocol for CUAD dataset converters.

    Implementors: CuadToNer, CuadToClassification
    """

    def convert(self, samples: list[dict]) -> ConversionOutput:
        """
        Convert a list of raw CUAD sample dicts to training examples.

        Parameters
        ----------
        samples : list[dict]
            Raw CUAD samples as returned by CuadLoader.load().
            Each dict follows the CUAD sample schema (see module docstring).

        Returns
        -------
        list[NERSample] | list[ClauseSample]
            Training samples in the appropriate format.
            Never returns an empty list without logging a warning.

        Raises
        ------
        ConversionError
            If a sample is malformed and cannot be recovered.
            Implementations should try/except per sample and only
            raise if more than 5% of samples fail (configurable).
        """
        ...

    def get_label_set(self) -> set[str]:
        """
        Return the set of all label strings this converter produces.

        Used by training scripts to validate that the model's label
        set matches the training data.

        Returns
        -------
        set[str]
            Set of EntityLabel string values.
        """
        ...
