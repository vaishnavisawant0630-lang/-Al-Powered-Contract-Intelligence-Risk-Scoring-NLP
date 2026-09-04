"""
Inference interface:
load a trained classifier and run it on raw contract text.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import joblib
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .calibrator import calibrate
from .config import TrainingConfig
from .heuristics import apply_heuristics

logger = logging.getLogger(__name__)


# ============================================================
# CLAUSE RESULT
# ============================================================

@dataclass(frozen=True)
class ClauseResult:
    """
    Result produced by the clause classifier.

    clause_text contains the actual text that was sent to
    the classifier.
    """

    clause_type: str
    present: bool
    confidence: float

    # Actual clause/evidence text
    clause_text: str = ""

    # Evidence spans detected for the clause
    evidence_spans: list[str] = field(
        default_factory=list
    )


# ============================================================
# CLAUSE CLASSIFIER
# ============================================================

class ClauseClassifier:

    def __init__(
        self,
        model_dir: str,
        labels: list[str],
        calibrators=None,
        max_length: int = 512,
        stride: int = 256,
        default_thresholds: dict[str, float] | None = None,
    ):
        self.model_dir = model_dir
        self.labels = labels
        self.calibrators = calibrators
        self.max_length = max_length
        self.stride = stride
        self.default_thresholds = default_thresholds

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            "Clause classifier device: %s",
            self.device,
        )

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_dir
            )
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_dir)
        )

        self.model.to(self.device)

        self.model.eval()

        logger.info(
            "Clause classifier loaded from %s",
            model_dir,
        )

    # ========================================================
    # WINDOWED PROBABILITIES
    # ========================================================

    def _windowed_probs(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Split long text into overlapping token windows.

        Returns the maximum probability for each clause
        label across all windows.
        """

        if not text or not text.strip():
            return np.zeros(
                len(self.labels),
                dtype=float,
            )

        enc = self.tokenizer(
            text,
            truncation=False,
            return_overflowing_tokens=True,
            max_length=self.max_length,
            stride=self.stride,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]

        attention_mask = enc[
            "attention_mask"
        ]

        all_probs = []

        batch_size = 8

        for i in range(
            0,
            input_ids.shape[0],
            batch_size,
        ):

            batch_ids = (
                input_ids[
                    i:i + batch_size
                ].to(self.device)
            )

            batch_mask = (
                attention_mask[
                    i:i + batch_size
                ].to(self.device)
            )

            with torch.no_grad():

                logits = self.model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                ).logits

            probs = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
            )

            all_probs.append(probs)

        if not all_probs:
            return np.zeros(
                len(self.labels),
                dtype=float,
            )

        stacked = np.concatenate(
            all_probs,
            axis=0,
        )

        # Maximum confidence across windows
        return stacked.max(axis=0)

    # ========================================================
    # EVIDENCE EXTRACTION
    # ========================================================

    def _extract_evidence(
        self,
        text: str,
        clause_type: str,
        window_chars: int = 200,
        max_spans: int = 3,
    ) -> list[str]:
        """
        Extract simple evidence spans.

        For short text, the entire text is returned.

        For long text, the first few overlapping windows
        are returned.
        """

        if not text or not text.strip():
            return []

        if len(text) <= window_chars:
            return [text.strip()]

        spans = []

        step = max(
            1,
            window_chars // 2,
        )

        for start in range(
            0,
            len(text),
            step,
        ):

            chunk = (
                text[
                    start:start + window_chars
                ]
                .strip()
            )

            if chunk:
                spans.append(chunk)

            if len(spans) >= max_spans:
                break

        return spans

    # ========================================================
    # THRESHOLD
    # ========================================================

    def _resolve_threshold(
        self,
        label: str,
        threshold: float | dict[str, float] | None,
    ) -> float:

        if isinstance(
            threshold,
            dict,
        ):
            return threshold.get(
                label,
                0.5,
            )

        if isinstance(
            threshold,
            (int, float),
        ):
            return float(threshold)

        if self.default_thresholds is not None:

            return self.default_thresholds.get(
                label,
                0.5,
            )

        return 0.5

    # ========================================================
    # CLASSIFY
    # ========================================================

    def classify(
        self,
        text: str,
        threshold: float | dict[str, float] | None = None,
        ner_entities: list | None = None,
    ) -> list[ClauseResult]:
        """
        Classify a piece of contract text.

        The supplied text is stored as clause_text whenever
        the clause is detected.
        """

        if not text or not text.strip():
            return []

        # ----------------------------------------------------
        # Model probabilities
        # ----------------------------------------------------

        raw_probs = self._windowed_probs(
            text
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        if self.calibrators is not None:

            probs = calibrate(
                raw_probs,
                self.calibrators,
            )

        else:

            probs = raw_probs

        # ----------------------------------------------------
        # Create results
        # ----------------------------------------------------

        results = []

        for i, label in enumerate(
            self.labels
        ):

            conf = float(
                probs[i]
            )

            thr = self._resolve_threshold(
                label,
                threshold,
            )

            present = conf >= thr

            # ------------------------------------------------
            # Actual text
            # ------------------------------------------------

            clause_text = (
                text.strip()
                if present
                else ""
            )

            # ------------------------------------------------
            # Evidence
            # ------------------------------------------------

            evidence = (
                self._extract_evidence(
                    text,
                    label,
                )
                if present
                else []
            )

            results.append(
                ClauseResult(
                    clause_type=label,
                    present=present,
                    confidence=conf,
                    clause_text=clause_text,
                    evidence_spans=evidence,
                )
            )

        # ----------------------------------------------------
        # Apply heuristics
        # ----------------------------------------------------

        results = apply_heuristics(
            text,
            results,
            ner_entities,
        )

        # ----------------------------------------------------
        # Re-apply present flag
        # ----------------------------------------------------

        fixed_results = []

        for r in results:

            resolved_threshold = (
                self._resolve_threshold(
                    r.clause_type,
                    threshold,
                )
            )

            expected_present = (
                r.confidence
                >= resolved_threshold
            )

            if (
                r.present
                == expected_present
            ):

                fixed_results.append(r)

            else:

                fixed_results.append(
                    ClauseResult(
                        clause_type=r.clause_type,
                        present=expected_present,
                        confidence=r.confidence,

                        # Preserve actual text
                        clause_text=(
                            r.clause_text
                            if r.clause_text
                            else (
                                text.strip()
                                if expected_present
                                else ""
                            )
                        ),

                        evidence_spans=(
                            r.evidence_spans
                        ),
                    )
                )

        results = fixed_results

        # ----------------------------------------------------
        # Sort highest confidence first
        # ----------------------------------------------------

        return sorted(
            results,
            key=lambda r: r.confidence,
            reverse=True,
        )


# ============================================================
# CACHED CLASSIFIER
# ============================================================

_CACHED_CLASSIFIER: ClauseClassifier | None = None


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_classifier(
    model_dir: str | None = None,
    labels_path: str | None = None,
    use_calibration: bool = True,
) -> ClauseClassifier:

    global _CACHED_CLASSIFIER

    cfg = TrainingConfig()

    model_dir = (
        model_dir
        or cfg.output_dir
    )

    labels_path = (
        labels_path
        or cfg.labels_path
    )

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    with open(
        labels_path,
        "r",
        encoding="utf-8",
    ) as f:

        labels = json.load(f)

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calibrators = None

    if use_calibration:

        try:

            calibrators = joblib.load(
                f"{model_dir}/calibrators.pkl"
            )

            logger.info(
                "Loaded calibrators from %s",
                model_dir,
            )

        except FileNotFoundError:

            logger.warning(
                "No calibrators.pkl found in %s "
                "— using raw sigmoid probabilities",
                model_dir,
            )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    default_thresholds = None

    try:

        with open(
            f"{model_dir}/thresholds.json",
            "r",
            encoding="utf-8",
        ) as f:

            default_thresholds = json.load(f)

        logger.info(
            "Loaded per-label thresholds from "
            "%s/thresholds.json",
            model_dir,
        )

    except FileNotFoundError:

        logger.info(
            "No thresholds.json found in %s "
            "— using global 0.5 threshold",
            model_dir,
        )

    # --------------------------------------------------------
    # Create classifier
    # --------------------------------------------------------

    _CACHED_CLASSIFIER = ClauseClassifier(
        model_dir=model_dir,
        labels=labels,
        calibrators=calibrators,
        default_thresholds=default_thresholds,
    )

    return _CACHED_CLASSIFIER


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def classify_clauses(
    text: str,
    threshold: float | dict[str, float] | None = None,
    ner_entities: list | None = None,
    model_dir: str | None = None,
) -> list[ClauseResult]:
    """
    Convenience entry point.

    Loads and caches the classifier on first call.

    threshold=None uses per-label tuned thresholds from
    thresholds.json if available, otherwise 0.5.
    """

    global _CACHED_CLASSIFIER

    if _CACHED_CLASSIFIER is None:

        load_classifier(
            model_dir=model_dir
        )

    return _CACHED_CLASSIFIER.classify(
        text,
        threshold=threshold,
        ner_entities=ner_entities,
    )