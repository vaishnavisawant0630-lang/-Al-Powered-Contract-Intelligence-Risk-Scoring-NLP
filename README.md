# AI-Powered Contract Intelligence & Risk Scoring

> **Phase 1**: Data Parsing & Baseline Modeling — ✅ Complete
> **Phase 2**: Legal NLP, Clause Classification & Risk Scoring — ✅ Complete
> **Phase 3**: Vector Search, API & Web UI — ✅ Complete
> **Phase 4**: Docker, Deployment & Load Testing — ⏳ In Progress

---

## 🚦 Current Status — Last Updated: 2026-08-24

### ✅ Phases 1–3 Complete and Verified End-to-End

Upload a contract → OCR/text extraction → NER → clause classification →
risk scoring → semantic search — all working through a live web UI, not
just isolated scripts. See the Engineering Journal below for the real
issues hit along the way and how they were fixed.

---

## ✅ Progress Tracker

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

---

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

---

## Project Architecture

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

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone <repo>
cd contract-intelligence
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
python -m ner.train
python -m ner.evaluate

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

---

## System Requirements

| Dependency | Version | Installation |
|---|---|---|
| Python | 3.10+ | pyenv or system |
| spaCy | 3.8+ | `pip install spacy` |
| Tesseract | 5.x | `apt install tesseract-ocr` / `brew install tesseract` / Windows installer |
| Poppler | any recent | `apt install poppler-utils` / `brew install poppler` |
| GPU | recommended for Phase 2 | Colab T4 or local CUDA (~19 min fine-tune vs. hours on CPU) |

---

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

---

## Environment Variables

See [`.env.example`](.env.example) for full configuration documentation.
**Never commit `.env`, API keys, or credentials to GitHub.**

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=core --cov=ingestion --cov=data_processing --cov=ner --cov=classification --cov=api
```
