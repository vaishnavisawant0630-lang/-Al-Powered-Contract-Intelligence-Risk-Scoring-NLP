"""
data_processing/cuad_to_classification.py
==========================================
Converts CUAD QA annotations → clause classification JSON Lines format.

OUTPUT FORMAT (per record, from spec §1.4)
-------------------------------------------
{
  "contract_name": "N-1_4.pdf",
  "clause_type":   "GOVERNING_LAW",
  "text_span":     "This Agreement shall be governed by the laws of California.",
  "label":         1
}

RULES
-----
- label=1  if the CUAD answer is non-empty (clause is present)
- label=0  if answers are empty (clause is absent in this contract)
- text_span: for label=1 → the answer text (first answer if multiple)
             for label=0 → the full contract context (the model must predict absence)
- clause_type: UPPER_SNAKE_CASE version of the CUAD question category

SPLIT
-----
  80% train → cuad_clauses_train.json  (JSON Lines)
  20% dev   → cuad_clauses_dev.json    (JSON Lines)
  Split is document-level (same titles go to same split).

CLAUSE TYPE NORMALISATION
--------------------------
"Governing Law" → "GOVERNING_LAW"
"IP Ownership Assignment" → "IP_OWNERSHIP_ASSIGNMENT"
(replaces spaces/slashes/hyphens with underscore, uppercases)
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Split ratio for this module (spec: 80/20)
TRAIN_RATIO  = 0.80
RANDOM_SEED  = 42

# Max characters for negative (absent) text_span samples
# We use the first N chars of the contract to give the model context
MAX_CONTEXT_CHARS_FOR_NEGATIVE = 512


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_clause_type(question: str) -> str:
    """
    Extract clause category name from a CUAD question and normalise to
    UPPER_SNAKE_CASE for use as clause_type field.

    Example:
        'Highlight the parts ... "Governing Law" ...' → "GOVERNING_LAW"
        'Highlight the parts ... "IP Ownership Assignment" ...' → "IP_OWNERSHIP_ASSIGNMENT"
    """
    # Extract text between first pair of double quotes
    parts = question.split('"')
    if len(parts) >= 3:
        raw = parts[1].strip()
    else:
        raw = question[:60]

    # Normalise to UPPER_SNAKE_CASE
    # Replace spaces, hyphens, slashes, dots with underscores
    normalised = re.sub(r"[\s\-/\.]+", "_", raw)
    # Remove any remaining non-word characters except underscore
    normalised = re.sub(r"[^\w]", "", normalised)
    return normalised.upper()


# ─────────────────────────────────────────────────────────────────────────────
# Main converter
# ─────────────────────────────────────────────────────────────────────────────

class CuadToClassification:
    """
    Converts CUAD QA samples to clause classification JSON Lines.

    Usage
    -----
        converter = CuadToClassification()
        stats = converter.convert(
            train_samples=train,
            dev_samples=dev,
            output_dir="data/processed/",
        )

    Output files
    ------------
    cuad_clauses_train.json  — JSON Lines, one record per QA pair
    cuad_clauses_dev.json    — JSON Lines, one record per QA pair
    """

    def __init__(
        self,
        max_context_chars: int = MAX_CONTEXT_CHARS_FOR_NEGATIVE,
    ) -> None:
        self.max_context_chars = max_context_chars

    # ── Public API ────────────────────────────────────────────────────────

    def convert(
        self,
        train_samples: list[dict],
        dev_samples:   list[dict],
        output_dir:    str | Path = "data/processed",
    ) -> dict:
        """
        Convert and write both train and dev classification files.

        Returns stats dict with record counts per split and per label.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_records = self._samples_to_records(train_samples)
        dev_records   = self._samples_to_records(dev_samples)

        self._write_jsonl(train_records, output_dir / "cuad_clauses_train.json")
        self._write_jsonl(dev_records,   output_dir / "cuad_clauses_dev.json")

        train_pos = sum(1 for r in train_records if r["label"] == 1)
        dev_pos   = sum(1 for r in dev_records   if r["label"] == 1)

        stats = {
            "train_records":   len(train_records),
            "dev_records":     len(dev_records),
            "train_positive":  train_pos,
            "train_negative":  len(train_records) - train_pos,
            "dev_positive":    dev_pos,
            "dev_negative":    len(dev_records) - dev_pos,
            "clause_types":    len(set(r["clause_type"] for r in train_records)),
        }

        logger.info(
            "Classification conversion complete — "
            "train=%d (pos=%d neg=%d)  dev=%d (pos=%d neg=%d)",
            stats["train_records"], stats["train_positive"], stats["train_negative"],
            stats["dev_records"],   stats["dev_positive"],  stats["dev_negative"],
        )
        return stats

    # ── Internal helpers ──────────────────────────────────────────────────

    def _samples_to_records(self, samples: list[dict]) -> list[dict]:
        """
        Convert a list of raw CUAD QA samples to classification records.

        One record per QA pair (not per contract).
        """
        records: list[dict] = []

        for sample in samples:
            contract_name = sample.get("title", "unknown")
            context       = sample.get("context", "")
            question      = sample.get("question", "")
            answers       = sample.get("answers", {})

            clause_type = _normalise_clause_type(question)
            texts       = answers.get("text", [])
            is_present  = bool(texts and texts[0])

            if is_present:
                # label=1: use the first answer span as text_span
                text_span = texts[0].strip()
                label     = 1
            else:
                # label=0: use the first N chars of contract context
                # This gives the model enough context to predict absence
                text_span = context[: self.max_context_chars].strip()
                label     = 0

            if not text_span:
                continue  # skip empty spans

            records.append({
                "contract_name": contract_name,
                "clause_type":   clause_type,
                "text_span":     text_span,
                "label":         label,
            })

        return records

    @staticmethod
    def _write_jsonl(records: list[dict], path: Path) -> None:
        """Write records as JSON Lines (one JSON object per line)."""
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Written %d records → %s", len(records), path)

    # ── Standalone runner ─────────────────────────────────────────────────

    @classmethod
    def run(cls, output_dir: str | Path = "data/processed") -> dict:
        """
        End-to-end: load CUAD → convert → write .json files.

        Usage
        -----
            python -m data_processing.cuad_to_classification
        """
        from data_processing.cuad_loader import CuadLoader
        loader = CuadLoader(train_ratio=TRAIN_RATIO)
        train_samples, dev_samples = loader.load()

        converter = cls()
        return converter.convert(train_samples, dev_samples, output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    stats = CuadToClassification.run(output_dir=output_dir)

    print("\n=== Classification Conversion Complete ===")
    print(f"  Train records:    {stats['train_records']:,}")
    print(f"    Positive (1):   {stats['train_positive']:,}")
    print(f"    Negative (0):   {stats['train_negative']:,}")
    print(f"  Dev records:      {stats['dev_records']:,}")
    print(f"    Positive (1):   {stats['dev_positive']:,}")
    print(f"    Negative (0):   {stats['dev_negative']:,}")
    print(f"  Clause types:     {stats['clause_types']}")
