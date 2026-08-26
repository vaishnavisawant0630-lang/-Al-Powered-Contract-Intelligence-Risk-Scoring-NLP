MODEL TRANING & FINE TUNING - MEMBER 5
YASH-KATHIRIYA
> **Phase 1**: Data Parsing & Baseline Modeling — ✅ Complete
> **Phase 2**: Legal NLP, Clause Classification & Risk Scoring — ✅ Complete
> **Phase 3**: Vector Search, API & Web UI — ✅ Complete
> **Phase 4**: Docker, Deployment & Load Testing — ⏳ In Progress

Fine-Tuning – Clause Classification
 main

Overview

 YASH-KATHIRIYA
## 🚦 Current Status — Last Updated: 2026-08-24

### ✅ Phases 1–3 Complete and Verified End-to-End

Upload a contract → OCR/text extraction → NER → clause classification →
risk scoring → semantic search — all working through a live web UI, not
just isolated scripts. See the Engineering Journal below for the real
issues hit along the way and how they were fixed.

AI-Powered Contract Intelligence & Risk Scoring

Phase 1: Data Parsing & Baseline Modeling

Phase 2: Legal NLP, Clause Classification & Risk Scoring
 main

This folder contains the fine-tuned Transformer model used for legal contract clause classification in the AI-Powered Contract Intelligence & Risk Scoring project.

The model is trained on the processed CUAD-based clause dataset and is designed to classify extracted contract clauses into predefined legal clause categories.

YASH-KATHIRIYA
| Phase | Focus | Status |
|---|---|---|
| 1 | CUAD dataset processing · OCR/PDF/DOCX/TXT ingestion · spaCy NER baseline | ✅ Complete |
| 2 | InLegalBERT clause classification · calibration · threshold tuning | ✅ Complete |
| 3 | FAISS semantic search · FastAPI backend · web UI | ✅ Complete |
| 4 | Docker · load testing · logging · deployment docs | ⏳ In progress |

### Phase 1 — Data Parsing & Baseline Modeling

| # | Task | Files | Status |
|---|---|---|---|
| 1 | CUAD Dataset Processing | `cuad_loader.py`, `span_validator.py`, `cuad_to_ner.py`, `cuad_to_classification.py`, `dataset_stats.py` | ✅ |
| 2 | OCR & Ingestion Pipeline | `pdf_extractor.py`, `ocr_extractor.py`, `docx_extractor.py`, `text_extractor.py`, `document_router.py`, `text_cleaner.py` | ✅ |
| 3 | spaCy NER Baseline | `base_config.cfg`, `train.py`, `evaluate.py`, `inference.py` | ✅ |
| 4 | Test Suite | full `tests/` directory | ✅ 91 passed / 0 failed |
| 5 | Model Training | `models/ner_baseline/model-best/` | ✅ |

### Phase 2 — Legal NLP, Clause Classification & Risk Scoring

| # | Task | Files | Status |
|---|---|---|---|
| 1 | Dataset Preparation (multi-label) | `classification/dataset_builder.py` | ✅ |
| 2 | Legal Transformer Fine-Tuning | `classification/trainer.py` (law-ai/InLegalBERT) | ✅ |
| 3 | Clause Classification & Evaluation | `classification/evaluator.py` | ✅ |
| 4 | Confidence Calibration | `classification/calibrator.py` (41 isotonic regressors) | ✅ |
| 5 | Per-Label Threshold Tuning | `classification/threshold_tuner.py` | ✅ |
| 6 | Heuristic Rules | `classification/heuristics/` | ✅ |
| 7 | Risk Scoring | `api/pipeline.py`, `api/config.py` | ✅ (weighted-sum scoring, integrated into the live pipeline — not a separate module) |

See [`classification/README.md`](classification/README.md) for the full
issue/fix/result journal specific to this phase.

### Phase 3 — Vector Search, API & Web UI

| # | Task | Files | Status |
|---|---|---|---|
| 1 | Sentence embeddings + FAISS index | `embeddings/embedder.py`, `embeddings/vector_store.py` | ✅ |
| 2 | FastAPI app (upload, background processing, status) | `api/main.py`, `api/pipeline.py`, `api/routers/contracts.py` | ✅ |
| 3 | Semantic + clause search endpoints | `api/routers/search.py` | ✅ |
| 4 | Risk report endpoint | `api/routers/risk.py` | ✅ |
| 5 | Web UI (upload, contracts list, search) | `api/static/index.html` | ✅ |

---

## 📊 Actual Results

### NER (Phase 1) — Dev set, 62 documents, 1438 entities

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| LAW_JURISDICTION | 0.804 | 0.719 | **0.759** | 57 |
| DATE | 0.482 | 0.512 | **0.496** | 129 |
| ORG | 0.455 | 0.269 | **0.338** | 316 |
| DURATION | 0.462 | 0.125 | **0.197** | 48 |
| IP_CLAUSE | 0.238 | 0.147 | **0.182** | 197 |
| CLAUSE | 0.337 | 0.107 | **0.162** | 515 |
| MONEY | 0.174 | 0.068 | **0.098** | 176 |
| **MICRO AVG** | **0.396** | **0.204** | **0.270** | **1438** |

### Clause Classification (Phase 2) — Dev set, 2115 spans, 41 labels

| Stage | Macro F1 |
|---|---|
| Raw model (threshold=0.5) | 0.456 |
| + Isotonic calibration | 0.731 |
| + Per-label tuned thresholds | **0.770** |

Full per-label table, issue journal, and the RoBERTa-baseline comparison are
in [`classification/README.md`](classification/README.md).

### Live Pipeline (Phase 3) — Verified via web UI

| Test contract | Clauses detected | Risk score |
|---|---|---|
| Normal (low-risk) | 6 | 0.00 (LOW) |
| Deliberately risky | 7 | **3.17 (MEDIUM)** |

### Test Suite

```
pytest tests/ -v
91 passed, 0 failed
```
The fine-tuning pipeline uses:

Python

Hugging Face Transformers

PyTorch

RoBERTa

Hugging Face Datasets

Scikit-learn

Pandas

NumPy

🚦 Current Status — Last Updated: 2026-08-24

✅ Currently At: Phase 1 Completed

Phase 1 — Data Parsing & Baseline Modeling is fully completed.
 main

CUAD dataset processing, OCR and document ingestion, spaCy NER baseline, model training, evaluation, and testing have been completed.

 YASH-KATHIRIYA
## 🛠️ Engineering Journal — Issues Faced, Fixes Applied, Results

This is the part of the project that shows the actual debugging and
decision-making, not just the final code. Kept here because commit
messages alone don't capture *why* something was fixed a certain way.

### Phase 1

**Issue: CUAD dataset loaded 0 samples.**
Root cause: `theatticusproject/cuad` on HuggingFace is a plain-text dump,
not the QA-format dataset the loader expected. Fix: switched to the correct
identifier, `theatticusproject/cuad-qa`. Result: 84,325 rows correctly
parsed instead of 0.

**Issue: `scripts/prepare_data.sh` printed "complete" but produced nothing.**
Root cause: the script was an unimplemented stub (`# TODO:` comments only).
Fix: ran the underlying Python modules directly.

**Issue: NER training crashed with exit code -9 (out of memory).**
Root cause: unbounded document length + `en_core_web_lg` warm-start vectors
spiked memory past the CPU runtime's RAM limit. Fix: capped
`corpora.*.max_length`.

**Issue: training-reported Dev F1 (0.463) didn't match an independent
evaluation on the same file (0.158).**
Root cause: the `max_length` cap above was set low enough (2000) that
spaCy's corpus reader *silently skipped* (not truncated) long documents from
both train and dev — including during training's own scoring pass, which
made the in-training number look better than the model actually was on the
full dataset. Fix: raised `max_length` to 6000 to stop dropping documents,
and treated the independent, full-dataset evaluation as the source of truth
from then on — not the number printed mid-training.

**Issue: `.txt` file upload crashed with
`'NoneType' object has no attribute 'raw_text'`.**
Root cause: `ingestion/text_extractor.py` was an unimplemented stub left
over from the original scaffold. Fix: implemented the encoding-fallback
text reader.

### Phase 2

See [`classification/README.md`](classification/README.md) for the full
detail — summary: raw-model precision was very low due to necessary class-
imbalance weighting during training; fixed with isotonic calibration + per-
label threshold tuning (Macro F1 0.456 → 0.770). A separate, larger issue —
whole-document classification missing clauses that were clearly present —
turned out to be a training/inference granularity mismatch (model trained
on short single-clause spans, but fed whole multi-clause documents at
inference time); fixed by classifying per-paragraph and merging results.
This was the single highest-impact fix in the project: it's the difference
between a demo contract always scoring 0 risk regardless of content, and one
that actually separates a safe contract (0.00, LOW) from a deliberately
risky one (3.17, MEDIUM).

### Phase 3

**Issue: server wouldn't start — `ModuleNotFoundError: No module named
'aiosqlite'`.**
Root cause: the virtual environment wasn't activated before running
`uvicorn`. Fix: `.venv\Scripts\activate` first.

**Issue: OCR tests failing — `poppler`/`tesseract` not found on Windows.**
Root cause: these are OS-level binaries, not Python packages — they don't
come from `pip install`. Fix: installed both separately and added them to
the Windows `PATH`.

---

## Known Limitations (stated honestly, not hidden)

- **NER Micro F1 (0.270)** is below the original 0.78–0.86 target. Root
  cause is a bounded CPU training budget, not a data or architecture
  problem — loss was still improving when training stopped. NER is a
  supporting signal in this system; clause classification (Phase 2) is the
  primary driver of the risk score.
- **Classification Macro F1 (0.770, tuned)** was calibrated and threshold-
  tuned on the same dev set used to report it — no separate held-out
  calibration split — so real-world performance on unseen contracts is
  likely somewhat lower than 0.770, though clearly better than the raw
  0.456. A 3-way train/calibration/test split would give a more trustworthy
  number.
- **models/ and data/processed/ are gitignored** (trained weights + data are
  ~450MB combined, over GitHub's practical limits) — they must be
  regenerated locally via the Quick Start steps below.
  
Next Step → Start Phase 2: Legal NLP, Clause Classification & Risk Scoring

Folder Structure
main

Fine-Tuning/
│
├── train.py
├── evaluate_metrics.py
├── README.md
│
└── clause_classifier/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── vocab.json
    └── merges.txt

Note: The trained model.safetensors file may be stored separately because of GitHub file-size/storage considerations.

 YASH-KATHIRIYA
```
contract-intelligence/
├── core/                    ← types, config, logging, exceptions
├── ingestion/               ← PDF, OCR, DOCX, TXT → clean text
├── data_processing/         ← CUAD → NER + classification training data
├── ner/                     ← spaCy NER: train, evaluate, inference
├── classification/          ← InLegalBERT fine-tuning, calibration,
│                              threshold tuning, heuristics, inference
├── embeddings/              ← sentence embeddings + FAISS index
├── api/                     ← FastAPI app: upload, pipeline, search,
│                              risk scoring, static web UI
├── models/                  ← trained weights (gitignored)
├── data/                    ← raw + processed data (gitignored)
├── tests/                   ← 91 passing tests
└── scripts/                 ← shell orchestration scripts
```

**Dependency rule**: `core ← ingestion, data_processing, ner, classification ← api, tests`

1. Model Training

Purpose

The training pipeline fine-tunes a pretrained Transformer model for legal clause classification.

The model learns to map contract clause text to the appropriate clause category.

Training Flow

CUAD / Processed Dataset
          ↓
Data Cleaning
          ↓
Train / Validation Split
          ↓
Tokenization
          ↓
RoBERTa Fine-Tuning
          ↓
Validation
          ↓
Trained Clause Classifier

Dataset

The project uses a processed clause dataset derived from the CUAD (Contract Understanding Atticus Dataset).

Expected dataset location:

data/processed/clause_dataset.csv

The dataset should contain at least:

text
label_id

where:

text = contract clause text

label_id = numerical clause category

Model

The baseline Transformer model used for fine-tuning is:

roberta-base

The model is trained as a sequence-classification model.

Example Configuration

Maximum sequence length : 512
Batch size              : 8
Epochs                  : 2
Learning rate           : 2e-5

These values can be modified in the training script depending on available hardware and dataset size.

2. Training the Model

Run the training script from the project root:

python train.py

The training pipeline performs the following operations:

Loads the processed clause dataset.

Detects the text and label columns.

Splits the dataset into training and validation sets.

Converts the data into a Hugging Face Dataset.

Tokenizes clause text using the RoBERTa tokenizer.

Fine-tunes the Transformer model.

Evaluates the model on the validation dataset.

Saves the trained model and tokenizer.

The trained model is saved to:

models/clause_classifier/

or the output directory configured in the training script.

#

Task

Files

Status

1

CUAD Dataset Processing

cuad_loader.py, span_validator.py, cuad_to_ner.py, cuad_to_classification.py, dataset_stats.py

✅ Completed

2

OCR & Ingestion Pipeline

pdf_extractor.py, ocr_extractor.py, docx_extractor.py, document_router.py, text_cleaner.py

✅ Completed

3

spaCy NER Baseline

base_config.cfg, train.py, evaluate.py, inference.py

✅ Completed

4

Test Suite

test_pdf_extractor.py, test_ocr_extractor.py, test_cuad_to_ner.py, test_ner_inference.py

✅ 65 passed / 12 skipped / 0 failed

5

Model Training

models/ner_baseline/model-best/
 main

✅ Completed

📦 What Has Been Delivered — Phase 1

23,063 KB of training data → data/processed/cuad_ner_train.spacy

4,147 KB of development data → data/processed/cuad_ner_dev.spacy

510 CUAD contracts processed

41 CUAD clause types mapped to 7 NER labels

Full ingestion pipeline:

Digital PDF extraction

Scanned PDF OCR

DOCX extraction

TXT processing

Automatic document routing

Text cleaning and normalization

Span validation and conflict resolution

spaCy NER configuration

NER model training

NER evaluation and inference

Trained model saved at:
models/ner_baseline/model-best/

65 unit + integration tests passing

12 tests skipped

0 tests failed

🧠 Phase 1 NER Labels

Label

Description

ORG

Contract parties / organizations

DATE

Contract-related dates

MONEY

Monetary amounts

LAW_JURISDICTION

Governing law / jurisdiction

DURATION

Contract duration / notice periods

IP_CLAUSE

Intellectual property related information

Other mapped legal entities

Additional legal contract information

⚙️ Phase 1 Key Design Decisions

All 41 CUAD clause types mapped to 7 NER labels

70% / 15% / 15% train/validation/test split

Automatic scanned-PDF detection

PDF text extraction first, OCR fallback when required

Span conflict resolution keeps the longer valid span

Structured logging for discarded/conflicting spans

No overlapping entities

en_core_web_lg used for spaCy warm-start

CPU-compatible training

GPU support available

Protocol interfaces used for loose coupling

Shared types and configuration handled through core/

🧪 Phase 1 Testing

pytest tests/ -v
Current Result
65 passed
12 skipped
0 failed
🏆 Phase 1 Status
Requirement	Status
CUAD dataset processing	✅
NER dataset generation	✅
PDF extraction	✅
OCR pipeline	✅
DOCX extraction	✅
TXT processing	✅
Text cleaning	✅
NER configuration	✅
NER model training	✅
NER evaluation	✅
NER inference	✅
Test suite	✅

Phase 1: ✅ 100% COMPLETED

🟡 PHASE 2 — Legal NLP, Clause Classification & Risk Scoring
Objective

Improve the Phase 1 baseline using a legal-domain transformer model and develop a clause classification and contract risk scoring system.

Status: ⬜ Not Started

Next Development Phase

📋 Phase 2 Task Tracker
#	Task	Files / Module	Description	Status
1	Dataset Preparation	data_processing/	Prepare CUAD data for clause classification	⬜
2	Label Mapping	label_mapping.py	Map CUAD questions to clause categories	⬜
3	Dataset Split	dataset_split.py	Split data into 70% train / 15% validation / 15% test	⬜
4	Tokenization	tokenizer.py	Tokenize legal contract text	⬜
5	Legal Transformer Setup	models/	Configure RoBERTa / Legal-BERT	⬜
6	Fine-Tuning	train_classifier.py	Fine-tune transformer on legal clauses	⬜
7	Clause Classification	classifier.py	Identify and classify contract clauses	⬜
8	Risk Detection	risk_detector.py	Detect potentially risky clauses	⬜
9	Risk Scoring	risk_scoring.py	Generate numerical contract risk score	⬜
10	Model Evaluation	evaluate.py	Calculate Accuracy, Precision, Recall and F1	⬜
11	Error Analysis	error_analysis.py	Analyze incorrect predictions	⬜
12	Model Saving	models/legal_classifier/	Save best trained model	⬜
13	Testing	tests/	Add classification and risk-scoring tests	⬜
🔄 Phase 2 Workflow
CUAD Dataset
      │
      ▼
Clause Extraction
      │
      ▼
Dataset Cleaning
      │
      ▼
70% Train ──────────┐
15% Validation ─────┤
15% Test ───────────┘
      │
      ▼
Tokenizer
      │
      ▼
RoBERTa / Legal-BERT
      │
      ▼
Fine-Tuning
      │
      ▼
Clause Classification
      │
      ▼
Risk Detection
      │
      ▼
Risk Scoring
      │
      ▼
Model Evaluation
🧠 Phase 2 Main Components
1. Legal Transformer

A legal-domain transformer model such as RoBERTa / Legal-BERT will be fine-tuned on contract data to understand legal language and clause context.

2. Clause Classification

The model will classify important contract clauses.

Clause Category	Description
Termination	Conditions for ending the agreement
Confidentiality	Protection of confidential information
Indemnification	Responsibility for losses or claims
Intellectual Property	Ownership and usage rights
Governing Law	Applicable law and jurisdiction
Limitation of Liability	Limits on financial/legal liability
Payment	Payment terms and obligations
Renewal	Contract renewal conditions
Assignment	Transfer of contractual rights
Non-Compete	Restrictions on competing activities
⚠️ Phase 2 Risk Detection

The system will analyze classified clauses and identify potentially risky contractual conditions.

Example
Contract Clause:

"The agreement may be terminated by either party
with 30 days written notice."

Expected analysis:

Clause Type: Termination

Duration: 30 Days

Risk Level: Medium

Risk Score: 55 / 100

Reason:
Short termination notice period.
📊 Phase 2 Risk Scoring

The system will generate a numerical risk score between 0 and 100.

Score	Risk Level	Meaning
0–20	🟢 Low	Low contractual risk
21–40	🟢 Low-Medium	Minor concerns
41–60	🟡 Medium	Requires review
61–80	🟠 High	Significant risk
81–100	🔴 Critical	Immediate review recommended

The final scoring thresholds and weights will be determined after evaluating the Phase 2 model and defining the project's risk methodology.

📈 Phase 2 Evaluation

The classification model will be evaluated using:

Metric	Purpose
Accuracy	Overall correct predictions
Precision	Correct positive predictions
Recall	Ability to find relevant clauses
F1 Score	Balance between precision and recall
Confusion Matrix	Class-level error analysis
Project Target
Metric	Target
Accuracy	≥ 80%
Precision	≥ 75%
Recall	≥ 75%
F1 Score	≥ 75%

These are project targets, not guaranteed results.

🎯 Phase 2 Deliverables
CUAD classification dataset
70/15/15 train/validation/test split
Legal-domain transformer model
Fine-tuned clause classification model
Clause classification pipeline
Risk detection module
Risk scoring module
Evaluation metrics
Confusion matrix
Error analysis report
Unit and integration tests
Trained model saved at:
models/legal_classifier/
🔗 Phase 1 → Phase 2 Connection
PHASE 1
Data Processing
      │
      ├── CUAD Dataset
      ├── Clean Contract Text
      ├── NER Annotations
      └── Baseline NER Model
              │
              ▼
PHASE 2
      │
      ├── Classification Dataset
      ├── Legal Transformer
      ├── Clause Classification
      ├── Risk Detection
      └── Risk Scoring
              │
              ▼
       Phase 2 Trained Model
🗺️ Phase 1 & Phase 2 Roadmap
Phase	Week	Focus	Main Deliverable	Status
Phase 1	Week 1	CUAD · OCR · Ingestion · NER · Testing · Training	Baseline NER System	✅ COMPLETED
Phase 2	Week 2	Legal Transformer · Clause Classification · Risk Scoring	Legal NLP + Risk System	⬜ NOT STARTED
🏗️ Project Architecture — Phase 1 & Phase 2
contract-intelligence/
│
├── core/
│   ├── types.py
│   ├── config.py
│   ├── logging.py
│   └── exceptions.py
│
├── ingestion/
│   ├── pdf_extractor.py
│   ├── ocr_extractor.py
│   ├── docx_extractor.py
│   ├── document_router.py
│   └── text_cleaner.py
│
├── data_processing/
│   ├── cuad_loader.py
│   ├── cuad_to_ner.py
│   ├── cuad_to_classification.py
│   ├── span_validator.py
│   ├── dataset_stats.py
│   └── dataset_split.py
│
├── ner/
│   ├── base_config.cfg
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── classification/
│   ├── tokenizer.py
│   ├── train_classifier.py
│   ├── classifier.py
│   ├── evaluate.py
│   └── error_analysis.py
│
├── risk/
│   ├── risk_detector.py
│   └── risk_scoring.py
│
├── models/
│   ├── ner_baseline/
│   │   └── model-best/
│   └── legal_classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│   ├── test_pdf_extractor.py
│   ├── test_ocr_extractor.py
│   ├── test_cuad_to_ner.py
│   ├── test_ner_inference.py
│   └── ...
│
├── scripts/
│
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
🔗 Dependency Rule
core
  ↓
ingestion
  ↓
data_processing
  ↓
ner
  ↓
classification
  ↓
risk

Dependency Rule: core provides shared types, configuration, logging, and exceptions. Modules should use defined interfaces and avoid circular dependencies.

⚙️ Environment Variables

See .env.example for complete configuration options.

GPU_ID=-1

MODEL_PATH=models/ner_baseline/model-best

CLASSIFIER_MODEL_PATH=models/legal_classifier

Security: Never commit .env, API keys, passwords, database credentials, or other secrets to GitHub.

💻 System Requirements
Dependency	Version	Installation
Python	3.11+	pyenv or system
spaCy	3.8+	pip install spacy
Tesseract	5.x	apt install tesseract-ocr / Windows installer
Poppler	Recent version	apt install poppler-utils
PyTorch	Compatible version	Project dependencies
GPU	Optional	CUDA-compatible GPU
🚀 Quick Start — Phase 1
# 1. Clone repository
git clone <repo>
cd contract-intelligence
 YASH-KATHIRIYA
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt -r requirements-phase2.txt

# 2. spaCy base model (warm-start vectors)
python -m spacy download en_core_web_lg

# 3. Tesseract + Poppler (OS-level, for OCR)
#    Windows: see UB-Mannheim/tesseract and oschwartz10612/poppler-windows releases
#    Mac:     brew install tesseract poppler
#    Linux:   apt install tesseract-ocr poppler-utils

# 4. Process CUAD data
python -m data_processing.cuad_to_ner
python -m data_processing.cuad_to_classification
python -m data_processing.dataset_stats

# 5. Train NER model


# 2. Create virtual environment
python -m venv .venv

# 3. Model Files

A trained Transformer model normally contains files such as:

```text
clause_classifier/
│
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.json
└── merges.txt

Important Files

File

Purpose

model.safetensors

Contains the trained model weights

config.json

Model architecture and configuration

tokenizer.json

Tokenizer configuration and vocabulary information

tokenizer_config.json

Tokenizer settings

special_tokens_map.json

Special-token configuration

vocab.json

RoBERTa vocabulary

merges.txt

RoBERTa BPE merge rules

The tokenizer files should be kept together with the model configuration. The trained model.safetensors file is required to run the trained classifier.

4. Model Evaluation

Purpose

The evaluation pipeline measures the performance of the trained clause classifier on validation/test data.

Run:

python evaluate_metrics.py

The evaluation script loads:

data/processed/clause_dataset.csv

and the trained model from:

models/clause_classifier/

Evaluation Metrics

The following metrics are calculated:

Accuracy

Measures the percentage of correctly classified clauses.

Accuracy = Correct Predictions / Total Predictions

Precision

Measures how many predicted instances of a class are actually correct.

Precision = TP / (TP + FP)

Recall

Measures how many actual instances of a class were correctly identified.

Recall = TP / (TP + FN)

F1-Score

The F1-score combines precision and recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

3. Activate on Windows

.venv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt
pip install -r requirements-dev.txt

5. Configure environment

copy .env.example .env

6. Install spaCy base model

python -m spacy download en_core_web_lg

7. Process CUAD data

python -m data_processing.cuad_loader

8. Train NER model
 main
python -m ner.train
python -m ner.evaluate

 YASH-KATHIRIYA
# 6. Train clause classifier (GPU strongly recommended)
python -m classification.dataset_builder
python -m classification.trainer
python -m classification.evaluator --model models/clause_classifier --dev data/processed/cuad_clauses_dev.json
python -m classification.calibrator --model models/clause_classifier --dev data/processed/cuad_clauses_dev.json
python -m classification.threshold_tuner

# 7. Test
pytest tests/ -v

# 8. Run
uvicorn api.main:app --reload
# open http://127.0.0.1:8000/
```

9. Evaluate NER model

python -m ner.evaluate --model models/ner_baseline/model-best

10. Run tests

pytest tests/ -v
🚀 Quick Start — Phase 2

1. Prepare classification dataset

python -m data_processing.cuad_to_classification

2. Create 70/15/15 dataset split

python -m data_processing.dataset_split

5. Classification Report

The evaluation pipeline generates a classification report containing:

precision
recall
f1-score
support

for each clause category.

Example:

              precision    recall    f1-score    support

Class 0          0.XX       0.XX       0.XX        XXX
Class 1          0.XX       0.XX       0.XX        XXX
Class 2          0.XX       0.XX       0.XX        XXX

accuracy                              0.XX        XXX
macro avg         0.XX       0.XX       0.XX        XXX
weighted avg      0.XX       0.XX       0.XX        XXX

The actual values depend on the trained model and evaluation dataset.
 main

3. Train legal classification model

python -m classification.train_classifier

 YASH-KATHIRIYA
| Dependency | Version | Installation |
|---|---|---|
| Python | 3.10+ | pyenv or system |
| spaCy | 3.8+ | `pip install spacy` |
| Tesseract | 5.x | `apt install tesseract-ocr` / `brew install tesseract` / Windows installer |
| Poppler | any recent | `apt install poppler-utils` / `brew install poppler` |
| GPU | recommended for Phase 2 | Colab T4 or local CUDA (~19 min fine-tune vs. hours on CPU) |

4. Evaluate classifier
 main

python -m classification.evaluate

 YASH-KATHIRIYA
## End-to-End Workflow

```
CONTRACT
   │
   ▼
Document Ingestion (PDF / DOCX / OCR / TXT)
   │
   ▼
Clean Text
   │
   ├──────────────► NER (Phase 1)  → ORG, DATE, MONEY, LAW_JURISDICTION, ...
   │
   ▼
Paragraph Splitting
   │
   ▼
Clause Classification (Phase 2, InLegalBERT)
   │
   ▼
Calibration + Threshold Tuning + Heuristics
   │
   ▼
Risk Scoring
   │
   ▼
FAISS Indexing (Phase 3) ──► Semantic / Clause Search
   │
   ▼
Contract Risk Report (Web UI)
```
5. Run error analysis

python -m classification.error_analysis
 main

< SUJALJETHWA10

6. Evaluation Output

 YASH-KATHIRIYA
See [`.env.example`](.env.example) for full configuration documentation.
**Never commit `.env`, API keys, or credentials to GitHub.**
Evaluation results are stored in the metrics directory.
 main

metrics/
├── metrics.json
└── classification_report.txt

metrics.json

 YASH-KATHIRIYA
```bash
pytest tests/ -v
pytest tests/ --cov=core --cov=ingestion --cov=data_processing --cov=ner --cov=classification --cov=api
```

Stores numerical evaluation results in JSON format.

Example:

{
    "accuracy": 0.00,
    "precision": 0.00,
    "recall": 0.00,
    "f1_score": 0.00
}

classification_report.txt

Contains the detailed classification report for each clause category.

7. Running the Complete Fine-Tuning Pipeline

From the project root:

Step 1 – Train

python train.py

Step 2 – Evaluate

python evaluate_metrics.py

Step 3 – Check Results

metrics/metrics.json
metrics/classification_report.txt

8. Requirements

Install the required dependencies:

pip install torch
pip install transformers
pip install datasets
pip install pandas
pip install numpy
pip install scikit-learn

Or install the project's requirements file:

pip install -r requirements.txt

9. Using the Trained Model

Once the trained model and tokenizer are available:

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/clause_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

A clause can then be tokenized and passed to the classifier for prediction.

text = "The agreement shall remain effective for a period of two years."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

outputs = model(**inputs)

prediction = outputs.logits.argmax(dim=-1).item()

print("Predicted class:", prediction)

10. Important Notes

The tokenizer used during inference should match the tokenizer used during training.

The trained model.safetensors file is required for actual model inference.

The model should be loaded from the same model directory structure used during training.

Keep the label-to-class mapping consistent between training and inference.

Do not commit API keys, passwords, .env files, or other secrets.

If the trained model is too large for normal GitHub storage, keep the model weights in dedicated model storage or use Git LFS.

11. Project Role

This fine-tuning component is responsible for:

Contract Text
      ↓
Clause Classification
      ↓
Clause Category
      ↓
Risk Scoring Pipeline

The predicted clause category can subsequently be used by the project's risk scoring and downstream NLP pipeline.

Summary

The Fine-Tuning module provides the Transformer-based clause classification component of the project.

It supports:

CUAD-based clause classification

RoBERTa fine-tuning

Automated tokenization

Model evaluation

Accuracy measurement

Precision/Recall/F1 evaluation

Classification reports

Saved model and tokenizer artifacts

This module forms the NLP classification layer of the AI-Powered Contract Intelligence & Risk Scoring system.

6. Run risk detection

python -m risk.risk_detector

7. Calculate risk score

python -m risk.risk_scoring

8. Run complete test suite

pytest tests/ -v
🧪 Testing Strategy
Phase 1
pytest tests/ -v

Current result:

65 passed
12 skipped
0 failed
Phase 2

Testing will cover:

Dataset
↓
Tokenizer
↓
Classification
↓
Risk Detection
↓
Risk Scoring
📊 Overall Project Status
Phase	Completion	Status
Phase 1 — Data Parsing & Baseline Modeling	100%	✅ Completed
Phase 2 — Legal NLP & Risk Scoring	0%	⬜ Not Started
📌 Current Next Step
✅ PHASE 1 COMPLETED
│
▼
🚀 START PHASE 2
│
├── Prepare classification dataset
├── Create 70/15/15 split
├── Setup RoBERTa / Legal-BERT
├── Fine-tune model
├── Build clause classifier
├── Build risk detector
├── Implement risk scoring
├── Evaluate model
└── Run tests
🎯 Phase 1 + Phase 2 Final Goal
CONTRACT
│
▼
Document Ingestion
│
┌────────┴────────┐
▼                 ▼
PDF/DOCX            OCR
│                 │
└────────┬────────┘
▼
Clean Text
│
▼
PHASE 1
│
▼
NER MODEL
│
┌────────────┼────────────┐
▼            ▼            ▼
ORG         DATE         MONEY
│            │            │
└────────────┼────────────┘
▼
PHASE 2
│
▼
Legal Transformer
│
▼
Clause Classification
│
▼
Risk Detection
│
▼
Risk Scoring
│
▼
Contract Risk Report

Current Milestone: Phase 1 — ✅ Completed

Next Milestone: Phase 2 — ⬜ Legal NLP, Clause Classification & Risk Scoring
 main
