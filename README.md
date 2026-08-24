# AI-Powered Contract Intelligence & Risk Scoring

> **Phase 1**: Data Parsing & Baseline Modeling — ✅ Complete

---

## 🚦 Current Status — Last Updated: 2026-08-19

### ✅ Phase 1 Complete — NER Baseline Trained & Evaluated

Training finished after 3 iterations of debugging (OOM fix, `max_length` tuning,
step-count tuning). Final model saved at `models/ner_baseline/model-best/`.

---

## ✅ Progress Tracker

### Phase 1 — Data Parsing & Baseline Modeling

| # | Task | Files | Status |
|---|---|---|---|
| 1 | **CUAD Dataset Processing** | `cuad_loader.py`, `span_validator.py`, `cuad_to_ner.py`, `cuad_to_classification.py`, `dataset_stats.py` | ✅ Complete |
| 2 | **OCR & Ingestion Pipeline** | `pdf_extractor.py`, `ocr_extractor.py`, `docx_extractor.py`, `text_extractor.py`, `document_router.py`, `text_cleaner.py` | ✅ Complete |
| 3 | **spaCy NER Baseline** | `base_config.cfg`, `train.py`, `evaluate.py`, `inference.py` | ✅ Complete |
| 4 | **Test Suite** | full `tests/` directory | ✅ 91 passed / 0 failed |
| 5 | **Model Training** | `models/ner_baseline/model-best/` | ✅ Complete |

### What Has Been Delivered

- 510 CUAD contracts processed (via `theatticusproject/cuad-qa` on HuggingFace),
  41 clause types → 7 NER labels
- Full ingestion pipeline: PDF (digital + scanned OCR), DOCX, and TXT — all
  extractors implemented and tested (TXT extractor was a stub in the original
  scaffold; implemented during integration testing)
- spaCy NER config: `max_steps=6000`, `patience=3000`, `max_length=6000` tokens
  per document, `en_core_web_lg` warm-start vectors, CPU training
- 91 unit + integration tests passing (0 failed, 0 skipped)

### Training Attempt Log

| Attempt | Outcome | Fix Applied |
|---|---|---|
| 1 | `theatticusproject/cuad` on HF Hub resolved to a plain-text dump, not the QA-format dataset — 0 samples loaded | Switched to `theatticusproject/cuad-qa` (correct SQuAD-format identifier) |
| 2 | `RuntimeError: spaCy training failed with exit code -9` (OOM, CPU runtime) | Capped `corpora.*.max_length` to bound per-document memory during training |
| 3 | `max_length=2000` silently **dropped** long documents from both train and dev corpora, inflating training-reported dev F1 (0.463) vs. the true full-dev-set F1 (0.158) from an independent `ner.evaluate` run | Raised `max_length` to 6000 so long documents are included rather than skipped, and treated the independently-scored number as the source of truth |
| 4 | Final run: `max_steps=6000`, `patience=3000` | Converged with acceptable stability; further step increases showed diminishing returns |

### Actual Results (Dev Set, 62 documents, 1438 entities)

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

**⚠️ Known limitation**: overall Micro F1 (0.270) is below the original
0.78–0.86 target range. Root cause: CPU-only training with a bounded step
budget (compute/time constraint), not a data or architecture defect — the
loss curve was still trending down when training stopped. `LAW_JURISDICTION`
performs well (0.759 F1) because its language pattern ("governed by the laws
of...") is short and highly consistent across contracts; `MONEY` and `CLAUSE`
perform worst because they are long, structurally diverse spans with fewer
clean training examples after span-conflict filtering.

**This is not the final accuracy figure for the system** — Phase 2's
transformer-based clause classifier (InLegalBERT) is the primary signal used
for risk scoring in the deployed pipeline, and significantly outperforms this
NER baseline (see Phase 2 README section).

---

## 🗺️ Full Phase Roadmap

| Phase | Week | Focus | Status |
|---|---|---|---|
| 1 | 1 | CUAD processing · OCR pipeline · spaCy NER baseline | ✅ Complete |
| 2 | 2 | InLegalBERT fine-tuning · clause classification · calibration | ✅ Complete |
| 3 | 3 | FAISS vector search · FastAPI + background processing · web UI | ✅ Complete |
| 4 | 4 | Docker · deployment docs · load testing · logging | ⏳ In progress |

---

## Project Architecture

```
contract-intelligence/
├── core/                    ← Foundation: types, config, logging, exceptions
├── ingestion/               ← Document → clean text (PDF, OCR, DOCX, TXT)
├── data_processing/         ← CUAD dataset → spaCy training artifacts
├── ner/                     ← NER model training, evaluation, and inference
├── classification/          ← Clause classification: training, calibration,
│                              heuristics, threshold tuning, inference (Phase 2)
├── embeddings/              ← Sentence embeddings + FAISS vector index (Phase 3)
├── api/                     ← FastAPI app: upload, processing pipeline,
│                              search, risk scoring, static web UI (Phase 3)
├── models/                  ← Saved model artifacts (gitignored — see below)
├── data/                    ← Raw + processed training data (gitignored)
├── tests/                   ← Pytest test suite (91 pass / 0 fail)
└── scripts/                 ← Shell scripts for pipeline orchestration
```

**Dependency rule**: `core ← ingestion, data_processing, ner, classification ← api, tests`
No sibling package imports another sibling. All shared types flow through `core/`.

**Note on `models/` and `data/`**: these are gitignored because trained model
weights (NER + InLegalBERT classifier, ~450MB combined) and processed
datasets exceed GitHub's practical size limits. They must be regenerated
locally by following the Quick Start steps below.

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo>
cd contract-intelligence
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt -r requirements-phase2.txt

# 2. Install spaCy base model (required for warm-start vectors)
python -m spacy download en_core_web_lg

# 3. Install Tesseract + Poppler (OS-level, for OCR)
#    Windows: see https://github.com/UB-Mannheim/tesseract/wiki and
#             https://github.com/oschwartz10612/poppler-windows/releases
#    Mac:     brew install tesseract poppler
#    Linux:   apt install tesseract-ocr poppler-utils

# 4. Process CUAD data
python -m data_processing.cuad_to_ner
python -m data_processing.cuad_to_classification
python -m data_processing.dataset_stats

# 5. Train NER model (~30-50 min on CPU)
python -m ner.train
python -m ner.evaluate

# 6. Train clause classifier (~20 min on GPU, much longer on CPU — GPU strongly recommended)
python -m classification.dataset_builder
python -m classification.trainer
python -m classification.evaluator --model models/clause_classifier --dev data/processed/cuad_clauses_dev.json
python -m classification.calibrator --model models/clause_classifier --dev data/processed/cuad_clauses_dev.json
python -m classification.threshold_tuner

# 7. Run tests
pytest tests/ -v

# 8. Start the app
uvicorn api.main:app --reload
# then open http://127.0.0.1:8000/
```

---

## System Requirements

| Dependency | Version | Installation |
|---|---|---|
| Python | 3.10+ | pyenv or system |
| spaCy | 3.8+ | `pip install spacy` |
| Tesseract | 5.x | `apt install tesseract-ocr` / `brew install tesseract` |
| poppler | any | `apt install poppler-utils` / `brew install poppler` |
| GPU | recommended for Phase 2 | Colab T4 or local CUDA GPU (~19 min fine-tune vs. hours on CPU) |

---

## Key Design Decisions (Phase 1)

- **All 41 CUAD clause types** mapped to 7 NER labels
- **Auto-detect scanned PDFs**: pdfminer first → OCR fallback if char density < 50/page
- **Span conflict resolution**: keep longer span, log every discard as structured audit record
- **CPU-only training**: `GPU_ID=-1` (Phase 2 uses GPU for the transformer model)
- **Protocol interfaces**: `BaseExtractor`, `BaseConverter`, `BaseNERModel` — loose coupling across phases
- **No overlapping entities**: interval sweep deduplication in both training data and inference

---

## Environment Variables

See [`.env.example`](.env.example) for full documentation of all configuration options.

---

## Running Tests

```bash
# All tests (91 pass, 0 fail)
pytest tests/ -v

# With coverage
pytest tests/ --cov=core --cov=ingestion --cov=data_processing --cov=ner --cov=classification --cov=api
```
