# AI-Powered Contract Intelligence & Risk Scoring

> **Phase 1**: Data Parsing & Baseline Modeling

---

## 🚦 Current Status — Last Updated: 2026-08-09

### ⏳ Currently At: Model Training (Phase 1 · Step 5 of 5)

> **Next Step → Run full NER training to completion and evaluate model quality**
>
> Command: `python -m ner.train` (est. 45–90 min on CPU, ~20 min on GPU)
> Then: `python -m ner.evaluate` to get per-label F1 scores

---

## ✅ Progress Tracker

### Phase 1 — Data Parsing & Baseline Modeling

| # | Task | Files | Status |
|---|---|---|---|
| 1 | **CUAD Dataset Processing** | `cuad_loader.py`, `span_validator.py`, `cuad_to_ner.py`, `cuad_to_classification.py`, `dataset_stats.py` | ✅ Complete |
| 2 | **OCR & Ingestion Pipeline** | `pdf_extractor.py`, `ocr_extractor.py`, `docx_extractor.py`, `document_router.py`, `text_cleaner.py` | ✅ Complete |
| 3 | **spaCy NER Baseline** | `base_config.cfg`, `train.py`, `evaluate.py`, `inference.py` | ✅ Complete |
| 4 | **Test Suite** | `test_pdf_extractor.py`, `test_ocr_extractor.py`, `test_cuad_to_ner.py`, `test_ner_inference.py` | ✅ 65 passed / 12 skipped / 0 failed |
| 5 | **Model Training** | `models/ner_baseline/model-best/` | ⏳ **In Progress — needs full run** |

### What Has Been Delivered

- **23,063 KB** of training data → `data/processed/cuad_ner_train.spacy`
- **4,147 KB** of dev data → `data/processed/cuad_ner_dev.spacy`
- 510 CUAD contracts processed, 41 clause types → 7 NER labels
- Full ingestion pipeline: PDF (digital + scanned OCR) + DOCX + TXT
- spaCy NER config validated (`patience=1600`, `max_steps=2000`, `en_core_web_lg` warm-start)
- 65 unit + integration tests passing; 12 skipped (need Tesseract + trained model)

### Training Attempt Log

| Attempt | Outcome | Fix Applied |
|---|---|---|
| Attempt 1 | `TypeError: 'module' object is not callable` | Switched to `subprocess python -m spacy train` |
| Attempt 2 | Config validation error — conflicting `learn_rate` | Removed duplicate scalar `learn_rate` from optimizer block |
| Attempt 3 | `patience=5` → stopped after 5 steps, F1=0.00 | Fixed: `patience=1600` (spaCy patience is in **steps**, not evaluations) |
| Attempt 4 | ⏳ Stopped manually by user (was running correctly) | — |

### Next Command to Resume

```bash
# Resume training (est. 45–90 min CPU)
python -m ner.train

# Then evaluate
python -m ner.evaluate

# Then re-run tests (8 skipped inference tests will now pass)
python -m pytest tests/ -v
```

### Expected Results After Training

| Entity | Expected F1 |
|---|---|
| ORG (parties) | 0.88 – 0.93 |
| DATE | 0.82 – 0.89 |
| LAW_JURISDICTION | 0.75 – 0.85 |
| MONEY | 0.79 – 0.87 |
| DURATION | 0.65 – 0.78 |
| IP_CLAUSE | 0.70 – 0.82 |
| **Overall Micro F1** | **0.78 – 0.86** |

---

## 🗺️ Full Phase Roadmap

| Phase | Week | Focus | Status |
|---|---|---|---|
| **1 (current)** | 1 | CUAD processing · OCR pipeline · spaCy NER baseline | ⏳ 90% done |
| 2 | 2 | RoBERTa-legal fine-tuning · clause classification | ⬜ Not started |
| 3 | 3 | Vector search (Pinecone/Milvus) · FastAPI + Celery | ⬜ Not started |
| 4 | 4 | Docker · AWS EC2 · Frontend highlights UI | ⬜ Not started |

---

## Project Architecture

```
contract-intelligence/
├── core/                    ← Foundation: types, config, logging, exceptions
├── ingestion/               ← Document → clean text (PDF, OCR, DOCX, TXT)
├── data_processing/         ← CUAD dataset → spaCy training artifacts
├── ner/                     ← NER model training, evaluation, and inference
├── models/                  ← Saved model artifacts (gitignored)
├── data/                    ← Raw + processed training data (gitignored)
├── tests/                   ← Pytest test suite (65 pass / 12 skip / 0 fail)
└── scripts/                 ← Shell scripts for pipeline orchestration
```

**Dependency rule**: `core ← ingestion, data_processing, ner ← tests`
No sibling package imports another sibling. All shared types flow through `core/`.

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo>
cd contract-intelligence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# 2. Configure
cp .env.example .env

# 3. Install spaCy base model (required for warm-start vectors)
python -m spacy download en_core_web_lg

# 4. Process CUAD data (already done — outputs in data/processed/)
python -m data_processing.cuad_loader

# 5. Train NER model  ← NEXT STEP
python -m ner.train

# 6. Evaluate
python -m ner.evaluate --model models/ner_baseline/model-best

# 7. Run tests
pytest tests/ -v
```

---

## System Requirements

| Dependency | Version | Installation |
|---|---|---|
| Python | 3.11+ | pyenv or system |
| spaCy | 3.8+ | `pip install spacy` |
| Tesseract | 5.x | `apt install tesseract-ocr` / `brew install tesseract` |
| poppler | any | `apt install poppler-utils` / `brew install poppler` |
| GPU | optional | Set `GPU_ID=0` in `.env` (~4× faster training) |

---

## Key Design Decisions (Phase 1)

- **All 41 CUAD clause types** mapped to 7 NER labels (future-proofed for Phase 2)
- **Auto-detect scanned PDFs**: pdfminer first → OCR fallback if char density < 50/page
- **Span conflict resolution**: keep longer span, log every discard as structured audit record
- **CPU-only training**: `GPU_ID=-1` (upgrade in Phase 2 with transformer models)
- **Protocol interfaces**: `BaseExtractor`, `BaseConverter`, `BaseNERModel` — loose coupling across phases
- **No overlapping entities**: interval sweep deduplication in both training data and inference

---

## Environment Variables

See [`.env.example`](.env.example) for full documentation of all configuration options.

---

## Running Tests

```bash
# All tests (65 pass, 12 skip until Tesseract + trained model available)
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov=ingestion --cov=data_processing --cov=ner
```
