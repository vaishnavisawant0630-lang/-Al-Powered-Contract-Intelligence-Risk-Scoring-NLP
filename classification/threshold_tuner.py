"""
Per-label threshold tuning.
===========================
The trained classifier uses a single global threshold (0.5) to decide
whether a clause is "present". Because rare clause types have far fewer
positive examples, 0.5 is rarely their optimal operating point — this
script sweeps thresholds per label on the (calibrated) dev-set
probabilities and picks the one that maximizes F1 for that label.

This requires NO retraining — it only changes the decision rule applied
on top of the existing model + calibrators.

Usage
-----
    python -m classification.threshold_tuner
    python -m classification.threshold_tuner --model models/clause_classifier

Writes
------
    <model_dir>/thresholds.json   — {label: optimal_threshold}
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np

from .calibrator import _collect_dev_probs, calibrate
from .config import TrainingConfig

logger = logging.getLogger(__name__)


def _f1_at_threshold(probs_col: np.ndarray, labels_col: np.ndarray, threshold: float) -> tuple[float, float, float]:
    preds = (probs_col >= threshold).astype(int)
    tp = int((preds & labels_col).sum())
    fp = int((preds & (1 - labels_col)).sum())
    fn = int(((1 - preds) & labels_col).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def tune_thresholds(model_dir: str, dev_path: str, labels_path: str,
                     candidate_thresholds: np.ndarray | None = None):
    logging.basicConfig(level=logging.INFO)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    candidate_thresholds = candidate_thresholds if candidate_thresholds is not None \
        else np.arange(0.05, 0.96, 0.01)

    logger.info("Collecting dev-set predictions")
    raw_probs, dev_labels = _collect_dev_probs(model_dir, dev_path, labels)

    try:
        calibrators = joblib.load(f"{model_dir}/calibrators.pkl")
        probs = calibrate(raw_probs, calibrators)
        logger.info("Using calibrated probabilities (calibrators.pkl found)")
    except FileNotFoundError:
        probs = raw_probs
        logger.warning("No calibrators.pkl found — tuning on raw sigmoid probabilities")

    results = {}
    print(f"\n{'Clause Type':<32}{'Old F1 (0.5)':>14}{'Best Thr':>10}{'New F1':>10}{'Delta':>10}{'Support':>10}")
    print("-" * 86)

    total_old_f1, total_new_f1 = 0.0, 0.0
    for i, name in enumerate(labels):
        col_probs = probs[:, i]
        col_labels = dev_labels[:, i]
        support = int(col_labels.sum())

        _, _, old_f1 = _f1_at_threshold(col_probs, col_labels, 0.5)

        best_thr, best_f1 = 0.5, old_f1
        for thr in candidate_thresholds:
            _, _, f1 = _f1_at_threshold(col_probs, col_labels, thr)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        results[name] = round(float(best_thr), 2)
        total_old_f1 += old_f1
        total_new_f1 += best_f1
        delta = best_f1 - old_f1
        marker = "  <-- improved" if delta > 0.005 else ""
        print(f"{name:<32}{old_f1:>14.3f}{best_thr:>10.2f}{best_f1:>10.3f}{delta:>+10.3f}{support:>10}{marker}")

    n = len(labels)
    print("-" * 86)
    print(f"{'MACRO F1':<32}{total_old_f1/n:>14.3f}{'':>10}{total_new_f1/n:>10.3f}{(total_new_f1-total_old_f1)/n:>+10.3f}\n")

    out_path = f"{model_dir}/thresholds.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote per-label thresholds to %s", out_path)
    logger.info("Old macro F1 (threshold=0.5 everywhere): %.4f", total_old_f1 / n)
    logger.info("New macro F1 (per-label tuned thresholds): %.4f", total_new_f1 / n)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/clause_classifier")
    parser.add_argument("--dev", default=None)
    parser.add_argument("--labels", default=None)
    args = parser.parse_args()

    cfg = TrainingConfig()
    tune_thresholds(
        model_dir=args.model,
        dev_path=args.dev or cfg.dev_path,
        labels_path=args.labels or cfg.labels_path,
    )
