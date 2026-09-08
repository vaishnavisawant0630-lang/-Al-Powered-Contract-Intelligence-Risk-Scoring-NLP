# Phase 02 — Transformer Fine-Tuning & Clause Classification

> **Platform**: AI-Powered Contract Intelligence & Risk Scoring
> **Week**: 2 | **Status**: 🔜 Pending Phase 1 Completion
> **Prerequisite**: Phase 1 complete — `cuad_clauses_train.json` and `cuad_clauses_dev.json` available

---

## Goal

- Fine-tune a **legal-domain transformer** (InLegalBERT / LegalRoBERTa) for **41-way clause classification**
- Evaluate rigorously with **per-clause precision/recall/F1** on the CUAD dev set
- Add **confidence calibration** (isotonic regression per label) for downstream risk scoring
- Add **post-processing heuristics** (regex + NER-assisted rules) to boost precision on high-value clauses

---

## Context

Phase 1 produced two JSON files consumed here:

```json
// Each record in cuad_clauses_train.json / cuad_clauses_dev.json
{
  "contract_name": "N-1_4.pdf",
  "clause_type":   "GOVERNING_LAW",
  "text_span":     "This Agreement shall be governed by the laws of California.",
  "label":          1
}
```

**Task**: Multi-label binary classification — for each of the 41 CUAD clause types,
predict whether the clause is **present** in a given `text_span`.

---

## Tech Stack

### Additions to Phase 1

| Library | Version | Purpose |
|---|---|---|
| **transformers** | HuggingFace latest | `AutoTokenizer`, `AutoModelForSequenceClassification` |
| **datasets** | HuggingFace latest | Batched tokenisation + DataLoader prep |
| **torch** | latest | PyTorch training loop + mixed precision via `torch.amp` |
| **evaluate** | HuggingFace latest | Precision, recall, F1 computation |
| **accelerate** | HuggingFace latest | Multi-GPU / CPU-offload support |
| **optuna** | latest | Hyperparameter search (LR, batch size, warmup) |
| **scipy** | latest | Calibration: isotonic regression / Platt scaling |
| **joblib** | latest | Serialise 41 calibration regressors to `.pkl` |
| **wandb** | optional | Training metrics logging (skippable via `try/except`) |

### Carried Over from Phase 1

`spaCy` (for NER-assisted heuristics), `structlog`, `pydantic-settings`, `python-dotenv`, `tqdm`

---

## Project Structure

```
contract-intelligence/
│
├── classification/
│   ├── __init__.py                    # Exports: load_classifier, classify_clauses
│   ├── config.py                      # TrainingConfig dataclass (all hyperparameters)
│   ├── dataset_builder.py             # JSON → HuggingFace Dataset + tokenisation + pos_weight
│   ├── trainer.py                     # Full fine-tuning loop via HuggingFace Trainer API
│   ├── evaluator.py                   # Per-clause P/R/F1 + macro/micro averages + metrics.json
│   ├── calibrator.py                  # Isotonic regression per label: raw logits → calibrated probs
│   ├── inference.py                   # load_model(), classify_clauses(text) → list[ClauseResult]
│   │
│   ├── heuristics/
│   │   ├── __init__.py                # Exports: apply_heuristics
│   │   ├── post_processor.py          # Orchestrates all rules in order → updated ClauseResult list
│   │   └── clause_rules.py            # Individual heuristic functions per clause type
│   │
│   └── config/
│       └── clause_labels.json         # Ordered list of all 41 CUAD clause names (index = label ID)
│
├── models/
│   └── clause_classifier/             # Saved fine-tuned model + tokenizer + calibrator artifacts
│       ├── config.json                # HuggingFace model config
│       ├── pytorch_model.bin          # Fine-tuned weights (or model.safetensors)
│       ├── tokenizer_config.json
│       ├── calibrators.pkl            # 41 IsotonicRegression objects (joblib)
│       └── training_meta.json         # Training stats, final macro F1, hardware info
│
├── notebooks/
│   └── error_analysis.ipynb           # Dev set predictions → surfaces highest-error clause types
│
└── tests/
    ├── test_dataset_builder.py        # label_vector shape, pos_weight computation
    ├── test_trainer_smoke.py          # 20 samples, 2 steps — assert no crash + loss decreases
    ├── test_calibrator.py             # Synthetic probs → assert output in [0,1] and monotone
    ├── test_heuristics.py             # Per-rule: trigger text → boosted, non-trigger → unchanged
    └── test_classifier_inference.py   # Saved model → short snippet → list[ClauseResult] valid
```

---

## `clause_labels.json` — All 41 CUAD Clause Types (Index Order)

```json
[
  "Document Name",
  "Parties",
  "Agreement Date",
  "Effective Date",
  "Expiration Date",
  "Renewal Term",
  "Notice Period To Terminate Renewal",
  "Governing Law",
  "Most Favored Nation",
  "Non-Compete",
  "Exclusivity",
  "No-Solicit Of Customers",
  "No-Solicit Of Employees",
  "Non-Disparagement",
  "Termination For Convenience",
  "ROFR/ROFO/ROFN",
  "Change Of Control",
  "Anti-Assignment",
  "Revenue/Profit Sharing",
  "Price Restrictions",
  "Minimum Commitment",
  "Volume Restriction",
  "IP Ownership Assignment",
  "Joint IP Ownership",
  "License Grant",
  "Non-Transferable License",
  "Affiliate License-Licensor",
  "Affiliate License-Licensee",
  "Unlimited/All-You-Can-Eat-License",
  "Irrevocable Or Perpetual License",
  "Source Code Escrow",
  "Post-Termination Services",
  "Audit Rights",
  "Uncapped Liability",
  "Cap On Liability",
  "Liquidated Damages",
  "Warranty Duration",
  "Insurance",
  "Covenant Not To Sue",
  "Third Party Beneficiary",
  "Limitation Of Liability"
]
```

> **Index 0 = label ID 0** in the 41-dim binary label vector.

---

## Detailed Requirements

### 1 · Model Selection & Dataset

**Files**: `classification/config.py`, `classification/dataset_builder.py`

#### 1.1 — `config.py` — `TrainingConfig` Dataclass

```python
@dataclass
class TrainingConfig:
    # Model
    model_name: str = "law-ai/InLegalBERT"
    # Fallback chain: InLegalBERT → lexlms/legal-roberta-large → roberta-base
    fallback_models: list[str] = field(default_factory=lambda: [
        "lexlms/legal-roberta-large",
        "roberta-base",
    ])
    num_labels: int = 41
    max_length: int = 512

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4      # effective batch = 32
    learning_rate: float = 2e-5
    num_train_epochs: int = 5
    warmup_ratio: float = 0.06
    fp16: bool = True                          # auto-disabled if no CUDA

    # Paths
    train_path: str = "data/processed/cuad_clauses_train.json"
    dev_path: str   = "data/processed/cuad_clauses_dev.json"
    output_dir: str = "models/clause_classifier"
    labels_path: str = "classification/config/clause_labels.json"
```

#### 1.2 — `dataset_builder.py` — Dataset + Tokenisation

**Input**: `cuad_clauses_train.json` / `cuad_clauses_dev.json`

**Build pipeline**:

```
JSON records
    │
    ▼ Group by contract_name + text_span
    │  (multiple records per span — one per clause type)
    │
    ▼ Build 41-dim binary label vector per span
    │  label_vector[i] = 1 if any record with clause_type=labels[i] has label=1
    │
    ▼ HuggingFace Dataset:
    │  columns: text (str), label_vector (List[int], len=41)
    │
    ▼ Tokenise:
    │  AutoTokenizer(model_name, truncation=True, padding="max_length", max_length=512)
    │
    ▼ Compute pos_weight per label:
       pos_weight[i] = (N - P_i) / P_i
       where N = total samples, P_i = positive samples for label i
       (used in BCEWithLogitsLoss to address class imbalance)
```

**Output**:

| Column | Type | Description |
|---|---|---|
| `input_ids` | `Tensor[512]` | Tokenised text |
| `attention_mask` | `Tensor[512]` | Padding mask |
| `labels` | `Tensor[41]` | Binary label vector (float32 for BCE) |
| `pos_weight` | `Tensor[41]` | Imbalance correction weights |

---

### 2 · Training Loop

**File**: `classification/trainer.py`

#### 2.1 — HuggingFace `TrainingArguments`

```python
TrainingArguments(
    output_dir                  = config.output_dir,
    evaluation_strategy         = "epoch",
    save_strategy               = "best",
    metric_for_best_model       = "eval_macro_f1",
    load_best_model_at_end      = True,
    per_device_train_batch_size = config.batch_size,         # 8
    gradient_accumulation_steps = config.gradient_accumulation_steps,  # 4
    learning_rate               = config.learning_rate,      # 2e-5
    num_train_epochs            = config.num_train_epochs,   # 5
    warmup_ratio                = config.warmup_ratio,       # 0.06
    fp16                        = config.fp16 and torch.cuda.is_available(),
    dataloader_num_workers      = 4,
    report_to                   = "wandb",  # wrapped in try/except
)
```

#### 2.2 — Custom Loss: `BCEWithLogitsLoss` + per-label `pos_weight`

```python
class WeightedTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
```

#### 2.3 — Per-Epoch Logging

Log at end of each epoch:
- **Macro F1** across all 41 labels
- **Per-label F1** for the 5 priority clause types:

| Priority Clause | Reason |
|---|---|
| Termination for Convenience | High legal risk |
| Governing Law | Jurisdiction-critical |
| Auto-Renewal (Renewal Term) | Common liability trap |
| Limitation of Liability | Cap on damages |
| Indemnification | Cost exposure |

---

### 3 · Evaluation

**File**: `classification/evaluator.py`

#### 3.1 — Metrics Computed

For each of the 41 clause labels:

| Metric | Description |
|---|---|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | Harmonic mean of P and R |
| Support | Number of positive examples in dev set |

**Aggregates**:
- **Micro-average F1** — pooled TP/FP/FN across all labels
- **Macro-average F1** — unweighted mean of per-label F1

#### 3.2 — Output

**Printed table** (sorted by F1 ascending — worst first):

```
Clause Type                    Precision   Recall    F1    Support
────────────────────────────────────────────────────────────────────
Source Code Escrow               0.412      0.341   0.373      29
Joint IP Ownership               0.523      0.401   0.454      47
...
Governing Law                    0.912      0.934   0.923     284
────────────────────────────────────────────────────────────────────
MICRO AVG                        0.821      0.798   0.809    3421
MACRO AVG                        0.756      0.731   0.743      —
```

**Saved file**: `models/clause_classifier/metrics.json`

```json
{
  "macro_f1": 0.743,
  "micro_f1": 0.809,
  "per_label": {
    "Governing Law": {"precision": 0.912, "recall": 0.934, "f1": 0.923, "support": 284},
    ...
  }
}
```

---

### 4 · Confidence Calibration

**File**: `classification/calibrator.py`

#### 4.1 — Why Calibrate?

Raw sigmoid outputs from the transformer are **overconfident or underconfident** on individual
clause types. The risk scorer (Phase 4) needs well-calibrated probabilities (e.g., a 0.85
confidence should mean the clause is present 85% of the time).

#### 4.2 — Method: Isotonic Regression (per label)

```python
# After training, collect dev set predictions
dev_probs   = sigmoid(dev_logits)   # shape: (N_dev, 41)
dev_labels  = ground_truth_labels   # shape: (N_dev, 41)

# For each of the 41 labels independently:
for i in range(41):
    calibrator_i = IsotonicRegression(out_of_bounds="clip")
    calibrator_i.fit(dev_probs[:, i], dev_labels[:, i])
    calibrators.append(calibrator_i)

# Save all 41 calibrators
joblib.dump(calibrators, "models/clause_classifier/calibrators.pkl")
```

#### 4.3 — Calibration API

```python
def calibrate(raw_probs: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    raw_probs : np.ndarray, shape (N, 41) or (41,)
        Raw sigmoid probabilities from the model.

    Returns
    -------
    calibrated_probs : np.ndarray, same shape
        Isotonic-regression-calibrated probabilities in [0, 1].
        Guaranteed monotone (higher raw prob → higher calibrated prob per label).
    """
```

#### 4.4 — Calibration Quality Check

After fitting, print a **reliability diagram** (expected calibration error per label):
- ECE < 0.05 = excellent
- ECE 0.05–0.10 = acceptable
- ECE > 0.10 = flag for review

---

### 5 · Post-Processing Heuristics

**Files**: `classification/heuristics/clause_rules.py`, `classification/heuristics/post_processor.py`

Applied **after** model prediction + calibration, **in the order listed**.

#### 5.1 — Heuristic Rules

| # | Clause Type | Trigger Condition | Action | Rationale |
|---|---|---|---|---|
| 1 | **Governing Law** | `calibrated_prob >= 0.40` AND text matches known jurisdiction regex | Boost to `max(prob, 0.85)` | Jurisdiction keywords are near-deterministic |
| 2 | **Auto-Renewal** (Renewal Term) | `calibrated_prob >= 0.35` AND text matches renewal trigger regex | Boost to `max(prob, 0.80)` | Renewal language is highly formulaic |
| 3 | **Expiration Date** | `calibrated_prob >= 0.30` AND spaCy NER found a `DATE` entity in same span | Boost to `max(prob, 0.75)` | Grounding on Phase 1 NER output |
| 4 | **Termination for Convenience** | Text contains `"either party may terminate"` within 100 chars of `"without cause"` OR `"for any reason"` | Force `confidence >= 0.90` | Explicit legal formula; model can miss it |
| 5 | **Limitation of Liability** | Text contains a **currency pattern** AND `"shall not exceed"` | Force `confidence >= 0.88` | Liability cap language is structurally rigid |

#### 5.2 — Known Jurisdiction Regex (Rule 1)

```python
JURISDICTION_RE = re.compile(
    r"\b(Delaware|New York|California|England and Wales|"
    r"Texas|Illinois|Washington|Singapore|Hong Kong)\b",
    re.IGNORECASE,
)
```

#### 5.3 — Auto-Renewal Trigger Regex (Rule 2)

```python
RENEWAL_RE = re.compile(
    r"\b(automatically renew|auto.renew|unless terminated|"
    r"evergreen|successive term|rolling renewal)\b",
    re.IGNORECASE,
)
```

#### 5.4 — Currency Pattern (Rule 5)

```python
CURRENCY_RE = re.compile(
    r"(\$[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|thousand))?|"
    r"USD\s?[\d,]+|EUR\s?[\d,]+)",
    re.IGNORECASE,
)
```

#### 5.5 — `post_processor.py` API

```python
def apply_heuristics(
    text: str,
    clause_results: list[ClauseResult],
    ner_entities: list[Entity],          # from Phase 1 NERModel.extract_entities()
) -> list[ClauseResult]:
    """
    Apply all heuristic rules in order.
    Returns updated ClauseResult list with adjusted confidence values.
    Logs every boost with rule name, original_prob, new_prob at DEBUG level.
    """
```

---

### 6 · Inference Interface

**File**: `classification/inference.py`

#### 6.1 — `ClauseResult` Data Structure

```python
@dataclass(frozen=True)
class ClauseResult:
    clause_type:    str         # One of the 41 CUAD labels
    present:        bool        # True if confidence >= threshold (default 0.5)
    confidence:     float       # Calibrated + heuristic-adjusted probability [0, 1]
    evidence_spans: list[str]   # Up to 3 text snippets supporting the prediction
```

#### 6.2 — `classify_clauses()` Algorithm

```python
def classify_clauses(
    text: str,
    threshold: float = 0.5,
) -> list[ClauseResult]:
```

**Steps**:

```
Input text
    │
    ▼ Split into overlapping 512-token windows (stride=256)
    │  (if text fits in 512 tokens: single window)
    │
    ▼ For each window:
    │   tokenise → model → sigmoid(logits) → raw_probs[41]
    │
    ▼ Aggregate windows:
    │   per-label confidence = max(raw_probs[label]) across all windows
    │
    ▼ calibrate(aggregated_probs) → calibrated_probs[41]
    │
    ▼ NERModel.extract_entities(full text) → ner_entities
    │
    ▼ apply_heuristics(text, initial_results, ner_entities) → final_results
    │
    ▼ Extract evidence_spans:
    │   For each predicted-present clause:
    │     sliding window overlap with CUAD training answer patterns → top 3 snippets
    │
    ▼ Build list[ClauseResult] sorted by confidence descending
    │
    Return
```

#### 6.3 — Evidence Span Extraction

For each clause predicted as present, extract up to **3 evidence snippets**:

1. Find the 200-char window in the input text that has maximum lexical overlap
   with the typical answer patterns for that clause type (derived from CUAD training data)
2. Return the raw text of those windows as `evidence_spans`

This is a **simple keyword/overlap heuristic** — Phase 3 will replace it with
attention-weighted extraction.

---

### 7 · Tests

**Files**: `tests/test_dataset_builder.py`, `tests/test_trainer_smoke.py`,
`tests/test_calibrator.py`, `tests/test_heuristics.py`, `tests/test_classifier_inference.py`

#### 7.1 — `test_dataset_builder.py`

```
ASSERT: label_vector shape == (41,)
ASSERT: pos_weight tensor shape == (41,)
ASSERT: pos_weight[i] > 1 for all labels (imbalance correction)
ASSERT: tokenised input_ids shape == (512,)
ASSERT: no NaN in label_vector or pos_weight
```

#### 7.2 — `test_trainer_smoke.py`

```
SETUP:  20 synthetic samples, 2 training steps
ASSERT: no exception raised
ASSERT: loss step 2 <= loss step 1 (loss decreased or stayed same)
ASSERT: model weights changed from initial (gradients flowed)
```

#### 7.3 — `test_calibrator.py`

```
SETUP:  synthetic raw_probs in [0,1], shape (100, 41)
ASSERT: calibrated_probs shape == (100, 41)
ASSERT: all values in [0, 1]
ASSERT: monotone per label — if raw_probs[a, i] < raw_probs[b, i]
        then calibrated_probs[a, i] <= calibrated_probs[b, i]
```

#### 7.4 — `test_heuristics.py`

For each of the 5 rules, provide:
- A **trigger text** that should activate the rule → assert confidence boosted
- A **non-trigger text** that should not activate → assert confidence unchanged

```python
# Example for Rule 1 — Governing Law
trigger_text     = "This Agreement shall be governed by the laws of Delaware."
non_trigger_text = "The parties agree to cooperate in good faith."
```

#### 7.5 — `test_classifier_inference.py`

```
SETUP:  load saved model from models/clause_classifier/
ASSERT: classify_clauses(snippet) returns list[ClauseResult]
ASSERT: len(result) == 41
ASSERT: all confidence values in [0, 1]
ASSERT: all present values are bool
ASSERT: evidence_spans is list (may be empty)
SKIP:   if models/clause_classifier/ does not exist
```

---

## How to Run (Quick Reference)

```bash
# Prerequisites: Phase 1 complete, data/processed/*.json exists

# 1. Install Phase 2 dependencies
pip install -r requirements-phase2.txt

# 2. Build dataset
python -m classification.dataset_builder

# 3. Fine-tune model (~6–12 hours on CPU, ~1–2 hours on single GPU)
python -m classification.trainer

# 4. Evaluate
python -m classification.evaluator \
  --model models/clause_classifier \
  --dev   data/processed/cuad_clauses_dev.json

# 5. Fit calibrators
python -m classification.calibrator \
  --model models/clause_classifier \
  --dev   data/processed/cuad_clauses_dev.json

# 6. Run tests
pytest tests/ -m "not integration" -v
```

---

## Expected Output Files

| File | Format | Description |
|---|---|---|
| `models/clause_classifier/pytorch_model.bin` | Binary | Fine-tuned transformer weights |
| `models/clause_classifier/calibrators.pkl` | joblib | 41 isotonic regression calibrators |
| `models/clause_classifier/metrics.json` | JSON | Per-label + macro/micro F1 scores |
| `models/clause_classifier/training_meta.json` | JSON | Hyperparameters, hardware, training time |
| `classification/config/clause_labels.json` | JSON | Ordered list of all 41 clause type names |

---

## Model Card

| Field | Value |
|---|---|
| **Architecture** | `law-ai/InLegalBERT` (BERT-base trained on legal corpora) |
| **Fallback** | `lexlms/legal-roberta-large` → `roberta-base` |
| **Task** | Multi-label binary classification (41 labels) |
| **Dataset** | CUAD — 510 contracts, ~22,450 Q&A pairs |
| **Loss** | `BCEWithLogitsLoss` with per-label `pos_weight` |
| **Effective Batch** | 32 (8 per device × 4 gradient accumulation steps) |
| **Learning Rate** | 2e-5 with linear warmup (6% of steps) |
| **Epochs** | 5 |
| **Hardware Estimate** | CPU: 6–12 hrs · Single GPU (T4): 1–2 hrs · A100: ~30 min |
| **Best Checkpoint** | Saved by `eval_macro_f1` |

---

## Expected Training Performance

### Dev Set Macro F1 Target

| Model | Expected Macro F1 |
|---|---|
| `roberta-base` (fallback) | 0.71 – 0.76 |
| `law-ai/InLegalBERT` | 0.76 – 0.82 |
| `lexlms/legal-roberta-large` | 0.78 – 0.85 |

### Top-5 Clause Types by F1 (Expected)

| Clause Type | Expected F1 | Why Easy |
|---|---|---|
| Governing Law | 0.90 – 0.94 | Distinctive jurisdiction keywords |
| Agreement Date | 0.88 – 0.93 | Structured date patterns |
| Parties | 0.87 – 0.92 | Named entity patterns |
| Expiration Date | 0.85 – 0.91 | Date + expiry language |
| Termination for Convenience | 0.83 – 0.90 | Formulaic legal phrase |

### Bottom-5 Clause Types by F1 (Expected)

| Clause Type | Expected F1 | Why Hard |
|---|---|---|
| Source Code Escrow | 0.35 – 0.50 | Rare, technical, few training examples |
| Joint IP Ownership | 0.40 – 0.55 | Ambiguous boundary with License Grant |
| ROFR/ROFO/ROFN | 0.42 – 0.57 | Multiple similar clause types confused |
| Covenant Not To Sue | 0.45 – 0.58 | Short spans, sparse signal |
| Uncapped Liability | 0.48 – 0.62 | Requires understanding negation |

---

## Heuristic Rule Explanations

### Rule 1 — Governing Law Boost
**Why**: Jurisdiction keywords (`Delaware`, `California`, etc.) are near-deterministic signals.
A transformer sometimes misses them due to tokenisation of jurisdiction names.
The regex is a cheap, high-precision override that eliminates false negatives when the model
has already assigned moderate probability (≥ 0.40).

### Rule 2 — Auto-Renewal Boost
**Why**: Renewal clauses use a small set of highly formulaic trigger phrases (`automatically renew`,
`evergreen`). When present, these phrases are virtually always clause-present. The model can
underweight them if the surrounding context is unusual. The regex adds a reliable hard floor.

### Rule 3 — Expiration Date NER Grounding
**Why**: This rule cross-validates the transformer with Phase 1's spaCy NER. If the NER model
independently found a `DATE` entity in the same text span AND the classifier is already somewhat
confident (≥ 0.30), the agreement between two independent systems justifies a confidence boost.
This also validates Phase 1 NER output in a downstream task.

### Rule 4 — Termination for Convenience Force
**Why**: The phrase `"either party may terminate"` + `"without cause"/"for any reason"` is the
legal definition of termination for convenience. If both are present within 100 characters of each
other, the clause is definitionally present. The model may score it lower due to surrounding
neutral language; the force override reflects domain knowledge that overrides statistical uncertainty.

### Rule 5 — Limitation of Liability Force
**Why**: A liability cap clause structurally requires (a) a monetary amount and (b) a cap phrase.
When both a currency pattern (`$X million`) and `"shall not exceed"` are present, the semantic
content is unambiguous. Forcing confidence ≥ 0.88 prevents the model from under-scoring a high-risk
legal provision that auditors specifically look for.

---

## Design Decisions (Locked for Phase 2)

| Decision | Choice | Rationale |
|---|---|---|
| Model selection order | InLegalBERT → LegalRoBERTa → roberta-base | Domain-specific models outperform general on legal text |
| Loss function | BCEWithLogitsLoss + per-label pos_weight | Severe class imbalance; raw BCE produces trivial all-zero predictions |
| Calibration method | Isotonic regression (per label) | Non-parametric; more flexible than Platt scaling for non-linear miscalibration |
| Heuristics placement | After calibration | Calibrated probabilities provide a stable threshold for rule triggers |
| Effective batch size | 32 (8 × 4 accumulation) | Matches BERT fine-tuning guidelines; feasible on consumer GPU (8GB VRAM) |
| Evidence extraction | Sliding-window lexical overlap | Simple and deterministic; Phase 3 replaces with attention weights |
| wandb | Optional (try/except) | Not everyone has W&B account; CI must not fail without it |
