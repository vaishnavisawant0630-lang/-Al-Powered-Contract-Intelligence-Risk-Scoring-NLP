"""Per-label isotonic regression calibration: raw sigmoid probs -> calibrated probs."""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import TrainingConfig
from .dataset_builder import _group_by_span, _read_records

logger = logging.getLogger(__name__)


def _collect_dev_probs(model_dir: str, dev_path: str, labels: list[str],
                        max_length: int = 512, batch_size: int = 16):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    records = _read_records(dev_path)
    examples = _group_by_span(records, labels)

    all_probs, all_labels = [], []
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

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def fit_calibrators(dev_probs: np.ndarray, dev_labels: np.ndarray) -> list[IsotonicRegression]:
    """Fit one isotonic regressor per label column."""
    num_labels = dev_probs.shape[1]
    calibrators = []
    for i in range(num_labels):
        reg = IsotonicRegression(out_of_bounds="clip")
        col_probs = dev_probs[:, i]
        col_labels = dev_labels[:, i]
        if col_labels.sum() == 0 or col_labels.sum() == len(col_labels):
            # Degenerate column (all-0 or all-1 support) — isotonic regression
            # can't learn anything useful; fall back to identity mapping.
            reg.fit([0.0, 1.0], [0.0, 1.0])
        else:
            reg.fit(col_probs, col_labels)
        calibrators.append(reg)
    return calibrators


def calibrate(raw_probs: np.ndarray, calibrators: list[IsotonicRegression]) -> np.ndarray:
    """Apply fitted calibrators. raw_probs shape (N, L) or (L,)."""
    single = raw_probs.ndim == 1
    probs = raw_probs.reshape(1, -1) if single else raw_probs

    out = np.zeros_like(probs, dtype=float)
    for i, reg in enumerate(calibrators):
        out[:, i] = reg.predict(probs[:, i])
    out = np.clip(out, 0.0, 1.0)

    return out[0] if single else out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def run(model_dir: str, dev_path: str, labels_path: str):
    logging.basicConfig(level=logging.INFO)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    logger.info("Collecting dev-set predictions for calibration")
    dev_probs, dev_labels = _collect_dev_probs(model_dir, dev_path, labels)

    logger.info("Fitting %d isotonic calibrators", len(labels))
    calibrators = fit_calibrators(dev_probs, dev_labels)

    calibrated = calibrate(dev_probs, calibrators)

    print(f"\n{'Label':<32}{'ECE (raw)':>12}{'ECE (calibrated)':>18}")
    print("-" * 62)
    flags = []
    for i, name in enumerate(labels):
        ece_raw = expected_calibration_error(dev_probs[:, i], dev_labels[:, i])
        ece_cal = expected_calibration_error(calibrated[:, i], dev_labels[:, i])
        flag = " <-- review" if ece_cal > 0.10 else ""
        if flag:
            flags.append(name)
        print(f"{name:<32}{ece_raw:>12.4f}{ece_cal:>18.4f}{flag}")
    print()
    if flags:
        logger.warning("Labels with ECE > 0.10 after calibration: %s", flags)

    out_path = f"{model_dir}/calibrators.pkl"
    joblib.dump(calibrators, out_path)
    logger.info("Saved %d calibrators to %s", len(calibrators), out_path)
    return calibrators


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/clause_classifier")
    parser.add_argument("--dev", default=None)
    parser.add_argument("--labels", default=None)
    args = parser.parse_args()

    cfg = TrainingConfig()
    run(
        model_dir=args.model,
        dev_path=args.dev or cfg.dev_path,
        labels_path=args.labels or cfg.labels_path,
    )
