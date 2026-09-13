"""Evaluate a saved clause classifier: per-label P/R/F1, macro/micro averages."""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import TrainingConfig
from .dataset_builder import _group_by_span, _read_records  # noqa: F401 (internal reuse)

logger = logging.getLogger(__name__)


def evaluate(model_dir: str, dev_path: str, labels_path: str, max_length: int = 512,
             batch_size: int = 16, threshold: float = 0.5):
    logging.basicConfig(level=logging.INFO)

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    logger.info("Loading model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    records = _read_records(dev_path)
    examples = _group_by_span(records, labels)
    logger.info("Evaluating on %d dev spans", len(examples))

    all_probs = []
    all_labels = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        texts = [ex["text"] for ex in batch]
        vecs = np.array([ex["label_vector"] for ex in batch], dtype=int)

        enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length,
                         return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(vecs)

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    preds = (probs >= threshold).astype(int)

    rows = []
    for i, name in enumerate(labels):
        tp = int((preds[:, i] & y_true[:, i]).sum())
        fp = int((preds[:, i] & (1 - y_true[:, i])).sum())
        fn = int(((1 - preds[:, i]) & y_true[:, i]).sum())
        support = int(y_true[:, i].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "label": name, "precision": precision, "recall": recall,
            "f1": f1, "support": support, "tp": tp, "fp": fp, "fn": fn,
        })

    tp_sum = sum(r["tp"] for r in rows)
    fp_sum = sum(r["fp"] for r in rows)
    fn_sum = sum(r["fn"] for r in rows)
    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    macro_f1 = sum(r["f1"] for r in rows) / len(rows)

    rows_sorted = sorted(rows, key=lambda r: r["f1"])

    print(f"\n{'Clause Type':<32}{'Precision':>10}{'Recall':>10}{'F1':>8}{'Support':>10}")
    print("-" * 72)
    for r in rows_sorted:
        print(f"{r['label']:<32}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['f1']:>8.3f}{r['support']:>10}")
    print("-" * 72)
    print(f"{'MICRO AVG':<32}{micro_p:>10.3f}{micro_r:>10.3f}{micro_f1:>8.3f}{tp_sum + fn_sum:>10}")
    print(f"{'MACRO AVG':<32}{'':>10}{'':>10}{macro_f1:>8.3f}{'':>10}\n")

    out = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_label": {r["label"]: {"precision": r["precision"], "recall": r["recall"],
                                    "f1": r["f1"], "support": r["support"]} for r in rows},
    }
    out_path = f"{model_dir}/metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote metrics to %s", out_path)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/clause_classifier")
    parser.add_argument("--dev", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    cfg = TrainingConfig()
    evaluate(
        model_dir=args.model,
        dev_path=args.dev or cfg.dev_path,
        labels_path=args.labels or cfg.labels_path,
        threshold=args.threshold,
    )
