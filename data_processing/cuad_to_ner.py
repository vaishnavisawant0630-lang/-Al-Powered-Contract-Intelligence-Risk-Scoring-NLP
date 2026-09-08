"""
data_processing/cuad_to_ner.py
================================
Converts CUAD QA annotations → spaCy NER training format (DocBin).

ENTITY LABEL MAPPING (from phase_01_tasks.md)
----------------------------------------------
CUAD Question Type          → NER Label
------------------------------------------
Parties                     → ORG
Governing Law               → LAW_JURISDICTION
Effective Date              → DATE
Expiration Date             → DATE
Agreement Date              → DATE
Notice Period to Terminate  → DURATION
Minimum Commitment          → MONEY
Revenue/Profit Sharing      → MONEY
Cap On Liability            → MONEY
Liquidated Damages          → MONEY
Warranty Duration           → DURATION
Renewal Term                → DURATION
Non-Compete                 → CLAUSE
Exclusivity                 → CLAUSE
Termination For Convenience → CLAUSE
Anti-Assignment             → CLAUSE
Change Of Control           → CLAUSE
IP Ownership Assignment     → IP_CLAUSE
License Grant               → IP_CLAUSE
Source Code Escrow          → IP_CLAUSE
Governing Law               → LAW_JURISDICTION
(all remaining 35 clause types) → CLAUSE

OUTPUT
------
cuad_ner_train.spacy  — spaCy DocBin (binary format)
cuad_ner_dev.spacy    — spaCy DocBin (binary format)

One spaCy Doc per contract (not per QA pair).
All entity spans from all 41 questions are merged into a single entity list
per contract, then validated by SpanValidator before writing to DocBin.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Entity label mapping  (spec §1.2)
# ─────────────────────────────────────────────────────────────────────────────

# Maps a substring of the CUAD question text → NER label
# Matching is done by checking if the key appears in question.lower()
QUESTION_TO_LABEL: dict[str, str] = {
    # Core labels (spec-required)
    "parties":                          "ORG",
    "governing law":                    "LAW_JURISDICTION",
    "jurisdiction":                     "LAW_JURISDICTION",
    "effective date":                   "DATE",
    "expiration date":                  "DATE",
    "agreement date":                   "DATE",
    "notice period to terminate":       "DURATION",
    "renewal term":                     "DURATION",
    "warranty duration":                "DURATION",
    "minimum commitment":               "MONEY",
    "revenue/profit sharing":           "MONEY",
    "cap on liability":                 "MONEY",
    "liquidated damages":               "MONEY",
    "price restrictions":               "MONEY",
    # IP clauses
    "ip ownership assignment":          "IP_CLAUSE",
    "joint ip ownership":               "IP_CLAUSE",
    "license grant":                    "IP_CLAUSE",
    "non-transferable license":         "IP_CLAUSE",
    "affiliate license":                "IP_CLAUSE",
    "unlimited/all-you-can-eat":        "IP_CLAUSE",
    "irrevocable or perpetual license": "IP_CLAUSE",
    "source code escrow":               "IP_CLAUSE",
    "covenant not to sue":              "IP_CLAUSE",
    # Restrictive covenants
    "non-compete":                      "CLAUSE",
    "exclusivity":                      "CLAUSE",
    "no-solicit of customers":          "CLAUSE",
    "no-solicit of employees":          "CLAUSE",
    "competitive restriction":          "CLAUSE",
    "non-disparagement":                "CLAUSE",
    "most favored nation":              "CLAUSE",
    # Operational clauses
    "termination for convenience":      "CLAUSE",
    "anti-assignment":                  "CLAUSE",
    "change of control":                "CLAUSE",
    "rofr/rofo/rofn":                   "CLAUSE",
    "post-termination services":        "CLAUSE",
    "audit rights":                     "CLAUSE",
    "volume restriction":               "CLAUSE",
    "insurance":                        "CLAUSE",
    "third party beneficiary":          "CLAUSE",
    "uncapped liability":               "CLAUSE",
    # Identity (document-level)
    "document name":                    "CLAUSE",
}

# All 7 distinct NER labels used by the model
ALL_NER_LABELS = [
    "ORG",
    "DATE",
    "MONEY",
    "DURATION",
    "LAW_JURISDICTION",
    "IP_CLAUSE",
    "CLAUSE",
]


def _question_to_label(question: str) -> str:
    """
    Map a CUAD question string to an NER label.

    Iterates QUESTION_TO_LABEL in order and returns the label for the first
    matching key. Falls back to "CLAUSE" if no match found.
    """
    q_lower = question.lower()
    for keyword, label in QUESTION_TO_LABEL.items():
        if keyword in q_lower:
            return label
    return "CLAUSE"


# ─────────────────────────────────────────────────────────────────────────────
# Main converter
# ─────────────────────────────────────────────────────────────────────────────

class CuadToNer:
    """
    Converts CUAD QA samples into spaCy DocBin NER training data.

    Usage
    -----
        converter = CuadToNer()
        converter.convert(
            train_samples=train,
            dev_samples=dev,
            output_dir="data/processed/",
        )
        # Writes: cuad_ner_train.spacy  cuad_ner_dev.spacy

    Design
    ------
    - Groups samples by contract title (one Doc per contract)
    - Merges all 41 QA answers into a single entity list per Doc
    - Runs SpanValidator to clean overlapping/misaligned spans
    - Writes spaCy DocBin binary format
    """

    def __init__(self) -> None:
        from data_processing.span_validator import SpanValidator
        self.validator = SpanValidator()

    # ── Public API ────────────────────────────────────────────────────────

    def convert(
        self,
        train_samples: list[dict],
        dev_samples:   list[dict],
        output_dir:    str | Path = "data/processed",
    ) -> dict:
        """
        Convert and save both splits.

        Returns
        -------
        dict with keys: train_docs, dev_docs, total_entities, total_conflicts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_stats = self._convert_split(
            train_samples,
            output_dir / "cuad_ner_train.spacy",
            split_name="train",
        )
        dev_stats = self._convert_split(
            dev_samples,
            output_dir / "cuad_ner_dev.spacy",
            split_name="dev",
        )

        stats = {
            "train_docs":        train_stats["docs"],
            "dev_docs":          dev_stats["docs"],
            "train_entities":    train_stats["entities"],
            "dev_entities":      dev_stats["entities"],
            "train_conflicts":   train_stats["conflicts"],
            "dev_conflicts":     dev_stats["conflicts"],
            "total_entities":    train_stats["entities"] + dev_stats["entities"],
            "total_conflicts":   train_stats["conflicts"] + dev_stats["conflicts"],
            "label_counts":      train_stats["label_counts"],
        }

        logger.info(
            "NER conversion complete — "
            "train_docs=%d dev_docs=%d "
            "total_entities=%d total_conflicts=%d",
            stats["train_docs"], stats["dev_docs"],
            stats["total_entities"], stats["total_conflicts"],
        )
        return stats

    # ── Internal: convert one split ───────────────────────────────────────

    def _convert_split(
        self,
        samples:    list[dict],
        output_path: Path,
        split_name: str,
    ) -> dict:
        """
        Convert one split (train or dev) and write DocBin to disk.
        """
        try:
            import spacy
            from spacy.tokens import DocBin
        except ImportError:
            raise ImportError(
                "spaCy is required for NER conversion.\n"
                "Install with: pip install spacy\n"
                "Then download model: python -m spacy download en_core_web_lg"
            )

        # Load blank English model for tokenisation
        try:
            nlp = spacy.load("en_core_web_lg", disable=["ner", "parser", "lemmatizer"])
        except OSError:
            logger.warning("en_core_web_lg not found — using blank en model")
            nlp = spacy.blank("en")

        from data_processing.span_validator import SpanValidator, Entity

        # Group samples by contract title → one Doc per contract
        contracts: dict[str, dict] = {}  # title → {context, qas}
        for sample in samples:
            title = sample["title"]
            if title not in contracts:
                contracts[title] = {"context": sample["context"], "qas": []}
            contracts[title]["qas"].append({
                "question": sample["question"],
                "answers":  sample["answers"],
            })

        doc_bin        = DocBin()
        total_entities = 0
        total_conflicts= 0
        label_counts: dict[str, int] = {lbl: 0 for lbl in ALL_NER_LABELS}
        skipped_docs   = 0

        for title, contract in contracts.items():
            context = contract["context"]
            if not context.strip():
                skipped_docs += 1
                continue

            # Build raw entity list from all 41 QA answers
            raw_entities: list[Entity] = []
            for qa in contract["qas"]:
                label    = _question_to_label(qa["question"])
                answers  = qa["answers"]
                texts    = answers.get("text", [])
                starts   = answers.get("answer_start", [])

                for span_text, start in zip(texts, starts):
                    if span_text and span_text.strip():
                        raw_entities.append(Entity(
                            start=start,
                            end=start + len(span_text),
                            label=label,
                            text=span_text,
                        ))

            # Validate spans
            clean_entities, conflicts = SpanValidator.validate(
                text=context,
                entities=raw_entities,
                doc_id=title,
            )
            total_conflicts += len(conflicts)

            # Create spaCy Doc + set entities
            doc = nlp.make_doc(context)
            ents = []
            for ent in clean_entities:
                span = doc.char_span(ent.start, ent.end, label=ent.label)
                if span is not None:
                    ents.append(span)
                    label_counts[ent.label] = label_counts.get(ent.label, 0) + 1
                    total_entities += 1

            doc.ents = ents
            doc_bin.add(doc)

        doc_bin.to_disk(output_path)
        n_docs = len(contracts) - skipped_docs

        logger.info(
            "[%s] DocBin written — docs=%d entities=%d conflicts=%d path=%s",
            split_name, n_docs, total_entities, total_conflicts, output_path,
        )

        return {
            "docs":         n_docs,
            "entities":     total_entities,
            "conflicts":    total_conflicts,
            "label_counts": label_counts,
        }

    # ── Standalone runner ─────────────────────────────────────────────────

    @classmethod
    def run(cls, output_dir: str | Path = "data/processed") -> dict:
        """
        End-to-end: load CUAD → convert → write .spacy files.

        Usage
        -----
            python -m data_processing.cuad_to_ner
        """
        from data_processing.cuad_loader import CuadLoader
        loader = CuadLoader()
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
    stats = CuadToNer.run(output_dir=output_dir)

    print("\n=== NER Conversion Complete ===")
    print(f"  Train docs:       {stats['train_docs']}")
    print(f"  Dev docs:         {stats['dev_docs']}")
    print(f"  Total entities:   {stats['total_entities']}")
    print(f"  Total conflicts:  {stats['total_conflicts']}")
    print("\n  Label breakdown:")
    for label, count in sorted(stats["label_counts"].items(), key=lambda x: -x[1]):
        print(f"    {label:<20} {count:>6}")
