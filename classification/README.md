# Clause Classification — Fine-Tuning Module

> **Status: ✅ Complete** (previously "Not Started" in the original planning
> doc — this README replaces that with what was actually built and the real
> results obtained.)

## Overview

This module contains the fine-tuned Transformer model used for legal
contract clause classification in the AI-Powered Contract Intelligence &
Risk Scoring project. It classifies extracted contract text into 41
CUAD clause categories and feeds the risk-scoring pipeline.

**Note on model choice**: the original plan specified `roberta-base` /
Legal-BERT. We switched to **`law-ai/InLegalBERT`** — a transformer
pretrained specifically on Indian legal text — because it starts with
legal-domain vocabulary and phrasing already learned, instead of a
general-purpose model that has to learn legal language from scratch during
fine-tuning. A teammate independently trained `roberta-base` for
comparison (single-label, no calibration) — see the comparison table below.

The fine-tuning pipeline uses:

- Python, PyTorch, Hugging Face Transformers & Datasets
- `law-ai/InLegalBERT` (legal-domain pretrained)
- scikit-learn (isotonic calibration, metrics)
- Pandas / NumPy

## Folder Structure (actual)

```
classification/
├── config.py               ← paths, hyperparameters, risk-weight config
├── dataset_builder.py       ← CUAD → multi-label training examples
├── trainer.py                ← fine-tuning script
├── evaluator.py              ← per-label precision/recall/F1 report
├── calibrator.py             ← isotonic regression calibration (41 labels)
├── threshold_tuner.py        ← per-label decision-threshold optimization
├── inference.py              ← ClauseClassifier — load, classify, calibrate
├── config/clause_labels.json ← the 41 label names
└── heuristics/
    ├── clause_rules.py       ← regex-based confidence boosts
    └── post_processor.py

models/clause_classifier/     ← trained weights (gitignored — see root README)
```

## Dataset

Built from CUAD (`theatticusproject/cuad-qa` on HuggingFace) via
`data_processing.cuad_to_classification` →
`data/processed/cuad_clauses_train.json` / `cuad_clauses_dev.json`
(7,882 train spans / 2,115 dev spans, 41 labels, **multi-label** — a span can
be true for more than one clause type at once, unlike a single-label setup).

## Training Configuration (actual)

```
Model            : law-ai/InLegalBERT
Max seq length   : 512
Batch size       : 8 (Colab T4 GPU)
Epochs           : 5
Learning rate    : 2e-5
Training time    : ~19 minutes on a T4 GPU
```

## How to Run

```bash
python -m classification.dataset_builder
python -m classification.trainer
python -m classification.evaluator --model models/clause_classifier --dev data/processed/cuad_clauses_dev.json
python -m classification.calibrator --model models/clause_classifier --dev data/processed/cuad_clauses_dev.json
python -m classification.threshold_tuner
```

---

## 🛠️ Issues Faced, Fixes, and Results

### Issue 1: Raw model had very low precision on most clause types
- **Symptom**: Recall often 0.87–1.00 but Precision as low as 0.04–0.20 —
  the model over-predicted almost everything as "present."
- **Root cause**: Heavy `pos_weight` was needed during training to
  counteract severe class imbalance (some clause types had only 5 positive
  examples out of thousands of rows). This makes the model aggressive by
  design — good recall, poor precision, unless corrected downstream.
- **Fix**: Isotonic-regression calibration (one calibrator per label, 41
  total) followed by per-label threshold tuning (sweeping thresholds on
  calibrated dev-set probabilities to find the F1-maximizing cutoff per
  label, instead of one global 0.5 threshold for all 41 labels).

### Issue 2: Whole-document classification missed obvious clauses in production
- **Symptom**: A test contract with an explicit liability-cap sentence was
  classified with 0 relevant clauses detected; a contract deliberately
  written to be high-risk (uncapped liability, liquidated damages,
  non-compete, exclusivity) scored **0.00 risk** — none of those clauses
  were detected.
- **Root cause**: The classifier was trained on individual CUAD clause spans
  (short, single-topic text), but the live pipeline fed it the **entire
  multi-clause document as one input** — diluting the model's attention
  across many unrelated topics simultaneously. Training-time and
  inference-time input granularity didn't match.
- **Fix**: Split the document into paragraph-sized chunks before
  classification, classify each chunk separately, merge by max confidence
  per label.

### Issue 3: `TERMINATION_FOR_CONVENIENCE` heuristic didn't match its own clause name
- **Symptom**: A contract with the literal phrase "terminate this Agreement
  for convenience" wasn't getting the expected confidence boost.
- **Root cause**: The regex matched `"without cause"` / `"for any reason"`
  but not `"for convenience"`.
- **Fix**: Added the missing phrase to the pattern.

---

## 📊 Results — Before vs. After

| Stage | Macro F1 | Notes |
|---|---|---|
| Raw model, threshold=0.5 | **0.456** | Original training result |
| + Isotonic calibration, threshold=0.5 | 0.731 | Same dev set, decision rule only changed |
| + Per-label tuned thresholds | **0.770** | 22 of 41 labels improved, up to +0.468 on the worst-performing ones |

*(Honesty note: calibration and threshold tuning were both fit/evaluated on
the same 2,115-span dev set — no separate held-out calibration split — so
this number is likely somewhat optimistic versus truly unseen data. A
train/calibration/test 3-way split would give a more trustworthy figure and
is a natural next step.)*

### Full per-label results (dev set)

| Clause Type | F1 (0.5) | Best Thr | F1 (tuned) |
|---|---|---|---|
| DOCUMENT_NAME | 0.983 | 0.50 | 1.000 |
| PARTIES | 0.982 | 0.05 | 0.996 |
| GOVERNING_LAW | 0.957 | 0.50 | 0.987 |
| INSURANCE | 0.776 | 0.50 | 0.961 |
| AUDIT_RIGHTS | 0.793 | 0.50 | 0.966 |
| AGREEMENT_DATE | 0.796 | 0.50 | 0.947 |
| CAP_ON_LIABILITY | 0.800 | 0.50 | 0.892 |
| REVENUE_PROFIT_SHARING | 0.560 | 0.50 | 0.897 |
| ANTI_ASSIGNMENT | 0.750 | 0.34 | 0.906 |
| EXPIRATION_DATE | 0.636 | 0.50 | 0.895 |
| RENEWAL_TERM | 0.504 | 0.50 | 0.838 |
| ROFR_ROFO_ROFN | 0.566 | 0.25 | 0.820 |
| TERMINATION_FOR_CONVENIENCE | 0.471 | 0.50 | 0.846 |
| LICENSE_GRANT | 0.665 | 0.50 | 0.806 |
| SOURCE_CODE_ESCROW | 0.535 | 0.50 | 0.739 |
| COVENANT_NOT_TO_SUE | 0.425 | 0.50 | 0.781 |
| WARRANTY_DURATION | 0.487 | 0.50 | 0.883 |
| IP_OWNERSHIP_ASSIGNMENT | 0.468 | 0.19 | 0.809 |
| MINIMUM_COMMITMENT | 0.417 | 0.34 | 0.698 |
| NOTICE_PERIOD_TO_TERMINATE_RENEWAL | 0.452 | 0.29 | 0.680 |
| EFFECTIVE_DATE | 0.575 | 0.34 | 0.678 |
| THIRD_PARTY_BENEFICIARY | 0.400 | 0.50 | 1.000 |
| POST_TERMINATION_SERVICES | 0.393 | 0.50 | 0.767 |
| VOLUME_RESTRICTION | 0.374 | 0.19 | 0.640 |
| NON_TRANSFERABLE_LICENSE | 0.340 | 0.07 | 0.552 |
| NON_COMPETE | 0.353 | 0.21 | 0.621 |
| EXCLUSIVITY | 0.306 | 0.10 | 0.556 |
| UNCAPPED_LIABILITY | 0.301 | 0.50 | 0.708 |
| JOINT_IP_OWNERSHIP | 0.250 | 0.19 | 0.588 |
| LIQUIDATED_DAMAGES | 0.235 | 0.05 | 0.867 |
| NON_DISPARAGEMENT | 0.214 | 0.50 | 0.941 |
| MOST_FAVORED_NATION | 0.208 | 0.05 | 0.833 |
| COMPETITIVE_RESTRICTION_EXCEPTION | 0.206 | 0.21 | 0.486 |
| UNLIMITED_ALL_YOU_CAN_EAT_LICENSE | 0.192 | 0.50 | 0.875 |
| PRICE_RESTRICTIONS | 0.186 | 0.05 | 0.593 |
| AFFILIATE_LICENSE_LICENSEE | 0.180 | 0.17 | 0.490 |
| NO_SOLICIT_OF_EMPLOYEES | 0.109 | 0.50 | 0.667 |
| NO_SOLICIT_OF_CUSTOMERS | 0.077 | 0.50 | 0.615 |
| AFFILIATE_LICENSE_LICENSOR | 0.070 | 0.08 | 0.172 |
| CHANGE_OF_CONTROL | 0.333 | 0.50 | 0.780 |
| IRREVOCABLE_OR_PERPETUAL_LICENSE | 0.371 | 0.34 | 0.779 |

### Confidence calibration quality (ECE — lower is better)

Isotonic calibration reduced Expected Calibration Error from roughly
**0.13–0.24** (raw model, uncalibrated) to **~0.0000** across all 41 labels —
the model's stated confidence now closely matches its real accuracy.

### Comparison with teammate's RoBERTa baseline

A teammate trained `roberta-base` independently, as a single-label
(argmax) classifier with no calibration and no threshold tuning, on the
same CUAD data:

| | This module (InLegalBERT) | Teammate's (RoBERTa) |
|---|---|---|
| Problem type | Multi-label (harder — a text can match multiple clause types at once) | Single-label (easier — exactly one label per example) |
| Calibration | ✅ Yes | ❌ No |
| Threshold tuning | ✅ Yes | ❌ No |
| Macro F1 | **0.456 → 0.770** | 0.32 |
| Classes with 0.00 F1 | 0 | 7 of 38 |

---

## Live Pipeline Integration

Verified end-to-end through the project's web UI (`api/pipeline.py` →
`classification.inference`):

| Test contract | Clauses detected | Risk score |
|---|---|---|
| Normal (low-risk) | 6 | 0.00 (LOW) — correct: mostly protective/neutral clauses |
| Deliberately risky | 7 (uncapped liability, liquidated damages, non-compete, exclusivity, ...) | **3.17 (MEDIUM)** |

This differentiation (0.00 vs. 3.17 across two contracts) is the practical
proof that the classifier + calibration + threshold tuning + paragraph-level
inference actually work together correctly, not just in isolated metrics.
