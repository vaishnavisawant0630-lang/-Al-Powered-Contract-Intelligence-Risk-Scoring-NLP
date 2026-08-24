# AI-Powered Contract Intelligence & Risk Scoring

> **Phase 1**: Data Parsing & Baseline Modeling
>
> **Phase 2**: Legal NLP, Clause Classification & Risk Scoring

---

## 🚦 Current Status — Last Updated: 2026-08-24

### ✅ Currently At: Phase 1 Completed

> **Phase 1 — Data Parsing & Baseline Modeling is fully completed.**
>
> CUAD dataset processing, OCR and document ingestion, spaCy NER baseline, model training, evaluation, and testing have been completed.
>
> **Next Step → Start Phase 2: Legal NLP, Clause Classification & Risk Scoring**

---

## ✅ Progress Tracker

### Phase 1 — Data Parsing & Baseline Modeling

| # | Task | Files | Status |
|---|---|---|---|
| 1 | **CUAD Dataset Processing** | `cuad_loader.py`, `span_validator.py`, `cuad_to_ner.py`, `cuad_to_classification.py`, `dataset_stats.py` | ✅ Completed |
| 2 | **OCR & Ingestion Pipeline** | `pdf_extractor.py`, `ocr_extractor.py`, `docx_extractor.py`, `document_router.py`, `text_cleaner.py` | ✅ Completed |
| 3 | **spaCy NER Baseline** | `base_config.cfg`, `train.py`, `evaluate.py`, `inference.py` | ✅ Completed |
| 4 | **Test Suite** | `test_pdf_extractor.py`, `test_ocr_extractor.py`, `test_cuad_to_ner.py`, `test_ner_inference.py` | ✅ 65 passed / 12 skipped / 0 failed |
| 5 | **Model Training** | `models/ner_baseline/model-best/` | ✅ Completed |

---

## 📦 What Has Been Delivered — Phase 1

- **23,063 KB** of training data → `data/processed/cuad_ner_train.spacy`
- **4,147 KB** of development data → `data/processed/cuad_ner_dev.spacy`
- **510 CUAD contracts** processed
- **41 CUAD clause types** mapped to **7 NER labels**
- Full ingestion pipeline:
  - Digital PDF extraction
  - Scanned PDF OCR
  - DOCX extraction
  - TXT processing
- Automatic document routing
- Text cleaning and normalization
- Span validation and conflict resolution
- spaCy NER configuration
- NER model training
- NER evaluation and inference
- Trained model saved at:
  `models/ner_baseline/model-best/`
- **65 unit + integration tests passing**
- **12 tests skipped**
- **0 tests failed**

---

## 🧠 Phase 1 NER Labels

| Label | Description |
|---|---|
| `ORG` | Contract parties / organizations |
| `DATE` | Contract-related dates |
| `MONEY` | Monetary amounts |
| `LAW_JURISDICTION` | Governing law / jurisdiction |
| `DURATION` | Contract duration / notice periods |
| `IP_CLAUSE` | Intellectual property related information |
| Other mapped legal entities | Additional legal contract information |

---

## ⚙️ Phase 1 Key Design Decisions

- **All 41 CUAD clause types** mapped to 7 NER labels
- **70% / 15% / 15%** train/validation/test split
- Automatic scanned-PDF detection
- PDF text extraction first, OCR fallback when required
- Span conflict resolution keeps the longer valid span
- Structured logging for discarded/conflicting spans
- No overlapping entities
- `en_core_web_lg` used for spaCy warm-start
- CPU-compatible training
- GPU support available
- Protocol interfaces used for loose coupling
- Shared types and configuration handled through `core/`

---

## 🧪 Phase 1 Testing

```bash
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

# 2. Create virtual environment
python -m venv .venv

# 3. Activate on Windows
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Configure environment
copy .env.example .env

# 6. Install spaCy base model
python -m spacy download en_core_web_lg

# 7. Process CUAD data
python -m data_processing.cuad_loader

# 8. Train NER model
python -m ner.train

# 9. Evaluate NER model
python -m ner.evaluate --model models/ner_baseline/model-best

# 10. Run tests
pytest tests/ -v
🚀 Quick Start — Phase 2
# 1. Prepare classification dataset
python -m data_processing.cuad_to_classification

# 2. Create 70/15/15 dataset split
python -m data_processing.dataset_split

# 3. Train legal classification model
python -m classification.train_classifier

# 4. Evaluate classifier
python -m classification.evaluate

# 5. Run error analysis
python -m classification.error_analysis

# 6. Run risk detection
python -m risk.risk_detector

# 7. Calculate risk score
python -m risk.risk_scoring

# 8. Run complete test suite
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
