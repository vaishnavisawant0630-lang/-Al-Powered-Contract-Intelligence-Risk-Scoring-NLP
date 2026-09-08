"""Inference interface: load a trained classifier, run it on raw contract text."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .calibrator import calibrate
from .config import TrainingConfig
from .heuristics import apply_heuristics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClauseResult:
    clause_type: str
    present: bool
    confidence: float
    evidence_spans: list[str] = field(default_factory=list)


class ClauseClassifier:
    def __init__(self, model_dir: str, labels: list[str], calibrators=None,
                 max_length: int = 512, stride: int = 256,
                 default_thresholds: dict[str, float] | None = None):
        self.model_dir = model_dir
        self.labels = labels
        self.calibrators = calibrators
        self.max_length = max_length
        self.stride = stride
        self.default_thresholds = default_thresholds  # per-label, from threshold_tuner.py

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def _windowed_probs(self, text: str) -> np.ndarray:
        """Split text into overlapping windows if it exceeds max_length tokens,
        run each window through the model, and aggregate per-label confidence
        as the max probability across windows."""
        enc = self.tokenizer(
            text, truncation=False, return_overflowing_tokens=True,
            max_length=self.max_length, stride=self.stride, return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        all_probs = []
        batch_size = 8
        for i in range(0, input_ids.shape[0], batch_size):
            batch_ids = input_ids[i:i + batch_size].to(self.device)
            batch_mask = attention_mask[i:i + batch_size].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=batch_ids, attention_mask=batch_mask).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

        stacked = np.concatenate(all_probs, axis=0)
        return stacked.max(axis=0)  # (num_labels,) — max confidence across windows

    def _extract_evidence(self, text: str, clause_type: str, window_chars: int = 200,
                           max_spans: int = 3) -> list[str]:
        """Simple lexical-overlap heuristic: split text into overlapping char
        windows and return the ones most likely relevant to this clause type
        (currently just returns the leading windows — Phase 3 replaces this
        with attention-weighted extraction, per phase_02_tasks.md)."""
        if len(text) <= window_chars:
            return [text] if text.strip() else []

        spans = []
        step = window_chars // 2
        for start in range(0, len(text), step):
            chunk = text[start:start + window_chars].strip()
            if chunk:
                spans.append(chunk)
            if len(spans) >= max_spans:
                break
        return spans

    def _resolve_threshold(self, label: str, threshold: float | dict[str, float] | None) -> float:
        if isinstance(threshold, dict):
            return threshold.get(label, 0.5)
        if isinstance(threshold, (int, float)):
            return float(threshold)
        if self.default_thresholds is not None:
            return self.default_thresholds.get(label, 0.5)
        return 0.5

    def classify(self, text: str, threshold: float | dict[str, float] | None = None,
                 ner_entities: list | None = None) -> list[ClauseResult]:
        raw_probs = self._windowed_probs(text)

        if self.calibrators is not None:
            probs = calibrate(raw_probs, self.calibrators)
        else:
            probs = raw_probs

        results = []
        for i, label in enumerate(self.labels):
            conf = float(probs[i])
            thr = self._resolve_threshold(label, threshold)
            present = conf >= thr
            evidence = self._extract_evidence(text, label) if present else []
            results.append(ClauseResult(
                clause_type=label, present=present, confidence=conf, evidence_spans=evidence,
            ))

        results = apply_heuristics(text, results, ner_entities)
        # Re-apply `present` flag in case heuristics pushed confidence past threshold.
        results = [
            r if r.present == (r.confidence >= self._resolve_threshold(r.clause_type, threshold))
            else ClauseResult(
                r.clause_type,
                r.confidence >= self._resolve_threshold(r.clause_type, threshold),
                r.confidence, r.evidence_spans,
            )
            for r in results
        ]

        return sorted(results, key=lambda r: r.confidence, reverse=True)


_CACHED_CLASSIFIER: ClauseClassifier | None = None


def load_classifier(model_dir: str | None = None, labels_path: str | None = None,
                     use_calibration: bool = True) -> ClauseClassifier:
    global _CACHED_CLASSIFIER
    cfg = TrainingConfig()
    model_dir = model_dir or cfg.output_dir
    labels_path = labels_path or cfg.labels_path

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    calibrators = None
    if use_calibration:
        try:
            calibrators = joblib.load(f"{model_dir}/calibrators.pkl")
        except FileNotFoundError:
            logger.warning("No calibrators.pkl found in %s — using raw sigmoid probabilities", model_dir)

    default_thresholds = None
    try:
        with open(f"{model_dir}/thresholds.json", "r", encoding="utf-8") as f:
            default_thresholds = json.load(f)
        logger.info("Loaded per-label thresholds from %s/thresholds.json", model_dir)
    except FileNotFoundError:
        logger.info("No thresholds.json found in %s — using global 0.5 threshold", model_dir)

    _CACHED_CLASSIFIER = ClauseClassifier(model_dir, labels, calibrators, default_thresholds=default_thresholds)
    return _CACHED_CLASSIFIER


def classify_clauses(text: str, threshold: float | dict[str, float] | None = None,
                      ner_entities: list | None = None,
                      model_dir: str | None = None) -> list[ClauseResult]:
    """Convenience entry point. Loads (and caches) the classifier on first call.

    threshold=None (default) uses the per-label tuned thresholds from
    thresholds.json if present, otherwise falls back to a global 0.5.
    """
    global _CACHED_CLASSIFIER
    if _CACHED_CLASSIFIER is None:
        load_classifier(model_dir=model_dir)
    return _CACHED_CLASSIFIER.classify(text, threshold=threshold, ner_entities=ner_entities)
