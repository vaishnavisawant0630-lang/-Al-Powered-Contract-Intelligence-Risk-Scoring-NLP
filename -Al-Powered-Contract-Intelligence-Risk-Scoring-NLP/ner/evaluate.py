"""
ner/evaluate.py
================
Evaluates the trained spaCy NER model on the dev set.

Prints a per-entity precision/recall/F1 table matching the spec §3.3 format:

    Label               Precision  Recall     F1      Support
    ──────────────────────────────────────────────────────────
    ORG                   0.912     0.887    0.899      412
    LAW_JURISDICTION      0.871     0.903    0.887      284
    DATE                  0.834     0.761    0.796      201
    MONEY                 0.798     0.743    0.770      156
    DURATION              0.756     0.701    0.728       89
    ──────────────────────────────────────────────────────────
    MICRO AVG             0.856     0.831    0.843     3421

USAGE
-----
    python -m ner.evaluate
    python -m ner.evaluate --model models/ner_baseline --dev data/processed/cuad_ner_dev.spacy
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "ner_baseline" / "model-best"
DEV_DATA  = ROOT / "data" / "processed" / "cuad_ner_dev.spacy"


def evaluate(
    model_path: Path = MODEL_DIR,
    dev_data:   Path = DEV_DATA,
    output_json: Path | None = None,
) -> dict:
    """
    Evaluate the NER model on the dev DocBin.

    Parameters
    ----------
    model_path : Path
        Path to the saved spaCy model directory (model-best/).
    dev_data : Path
        Path to cuad_ner_dev.spacy (DocBin).
    output_json : Path | None
        If set, writes evaluation results to this JSON file.

    Returns
    -------
    dict
        {
          "micro_f1": float,
          "micro_precision": float,
          "micro_recall": float,
          "per_label": {label: {"precision", "recall", "f1", "support"}},
        }
    """
    import spacy
    from spacy.tokens import DocBin
    from spacy.scorer import Scorer

    # ── Load model ────────────────────────────────────────────────────────
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run: python -m ner.train"
        )
    if not dev_data.exists():
        raise FileNotFoundError(
            f"Dev data not found: {dev_data}\n"
            f"Run: python data_processing/run_task1.py"
        )

    logger.info("Loading model from %s", model_path)
    nlp = spacy.load(str(model_path))

    logger.info("Loading dev set from %s", dev_data)
    doc_bin  = DocBin().from_disk(dev_data)
    dev_docs = list(doc_bin.get_docs(nlp.vocab))

    logger.info("Evaluating on %d documents...", len(dev_docs))

    # ── Compute predictions and scores ─────────────────────────────────────
    # Per-label accumulators: {label → (tp, fp, fn)}
    label_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for gold_doc in dev_docs:
        # Run model on the raw text
        pred_doc = nlp(gold_doc.text)

        gold_spans = {(e.start_char, e.end_char, e.label_) for e in gold_doc.ents}
        pred_spans = {(e.start_char, e.end_char, e.label_) for e in pred_doc.ents}

        for span in pred_spans:
            label = span[2]
            if span in gold_spans:
                label_stats[label]["tp"] += 1
            else:
                label_stats[label]["fp"] += 1

        for span in gold_spans:
            label = span[2]
            if span not in pred_spans:
                label_stats[label]["fn"] += 1

    # ── Compute metrics ────────────────────────────────────────────────────
    per_label: dict[str, dict] = {}
    total_tp = total_fp = total_fn = 0

    for label, counts in sorted(label_stats.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        support = tp + fn  # all gold occurrences

        per_label[label] = {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "support":   support,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Micro averages
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0 else 0.0
    )

    results = {
        "micro_precision": round(micro_p, 4),
        "micro_recall":    round(micro_r, 4),
        "micro_f1":        round(micro_f, 4),
        "total_support":   total_tp + total_fn,
        "per_label":       per_label,
    }

    # ── Print table ────────────────────────────────────────────────────────
    _print_table(per_label, micro_p, micro_r, micro_f, total_tp + total_fn)

    # ── Optionally write JSON ─────────────────────────────────────────────
    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results written to %s", output_json)

    return results


def _print_table(
    per_label: dict[str, dict],
    micro_p: float,
    micro_r: float,
    micro_f: float,
    total_support: int,
) -> None:
    """Print the spec-format evaluation table."""
    sep = "─" * 58
    header = f"{'Label':<22}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Support':>7}"
    print()
    print(header)
    print(sep)

    # Sort by F1 descending
    for label, m in sorted(per_label.items(), key=lambda x: -x[1]["f1"]):
        print(
            f"{label:<22}  {m['precision']:>9.3f}  {m['recall']:>6.3f}"
            f"  {m['f1']:>6.3f}  {m['support']:>7}"
        )

    print(sep)
    print(
        f"{'MICRO AVG':<22}  {micro_p:>9.3f}  {micro_r:>6.3f}"
        f"  {micro_f:>6.3f}  {total_support:>7}"
    )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="Evaluate CUAD NER model")
    parser.add_argument("--model",  default=str(MODEL_DIR),
                        help="Path to model-best/ directory")
    parser.add_argument("--dev",    default=str(DEV_DATA),
                        help="Path to cuad_ner_dev.spacy")
    parser.add_argument("--output", default=None,
                        help="Optional path to write JSON results")
    args = parser.parse_args()

    evaluate(
        model_path=Path(args.model),
        dev_data=Path(args.dev),
        output_json=Path(args.output) if args.output else None,
    )
