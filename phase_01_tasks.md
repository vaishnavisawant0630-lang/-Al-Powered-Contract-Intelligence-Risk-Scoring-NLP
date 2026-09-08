# Phase 01 — Data Parsing & Baseline Modeling

> **Platform**: AI-Powered Contract Intelligence & Risk Scoring
> **Week**: 1 | **Status**: 🚧 In Progress

---

## Goal

Establish the full data foundation:

- Parse and tokenize the **CUAD dataset** into training-ready format
- Build a robust **OCR pipeline** that handles real-world contract PDFs
- Train a **baseline spaCy NER model** that can extract core legal entities
  (organizations, dates, monetary values, jurisdictions) from raw contract text

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Runtime |
| **spaCy** | 3.x | Baseline NER + pipeline backbone |
| **en_core_web_lg** | latest | Pre-trained model for warm-start |
| **datasets** | HuggingFace | Loading the CUAD dataset |
| **pdf2image** | latest | Render PDF pages to PIL images (OCR) |
| **pytesseract** | latest | OCR for scanned PDFs |
| **pdfminer.six** | latest | Direct text extraction from digital PDFs |
| **python-docx** | latest | Word document ingestion |
| **Pillow** | latest | Image preprocessing for OCR quality |
| **scikit-learn** | latest | Train/val split, evaluation metrics |
| **tqdm** | latest | Progress bars for data processing |
| **structlog** | latest | Structured JSON logging |
| **python-dotenv** | latest | Load `.env` config into environment |

---

## Project Structure

```
contract-intelligence/
│
├── data/
│   ├── raw/                           # Unzipped CUAD dataset lands here (NOT committed)
│   └── processed/
│       ├── cuad_ner_train.spacy       # spaCy binary training format (DocBin)
│       ├── cuad_ner_dev.spacy         # spaCy binary dev format
│       ├── cuad_clauses_train.json    # Clause classification format (used in Phase 2)
│       └── cuad_clauses_dev.json
│
├── ingestion/
│   ├── __init__.py                    # Exports: DocumentRouter, TextCleaner
│   ├── base.py                        # BaseExtractor Protocol
│   ├── pdf_extractor.py               # Extracts text from digital PDFs via pdfminer.six
│   ├── ocr_extractor.py               # OCR pipeline for scanned PDFs (pdf2image + tesseract)
│   ├── docx_extractor.py              # Extracts text from .docx files
│   ├── document_router.py             # Routes a file to correct extractor by type/content
│   └── text_cleaner.py                # Post-extraction cleanup (ligatures, whitespace, encoding)
│
├── data_processing/
│   ├── __init__.py                    # Exports: load_cuad, build_ner_corpus, build_clause_corpus
│   ├── base.py                        # BaseConverter Protocol
│   ├── cuad_loader.py                 # Loads CUAD via HuggingFace Datasets, inspects schema
│   ├── cuad_to_ner.py                 # Converts CUAD Q&A annotations → spaCy NER training spans
│   ├── cuad_to_classification.py      # Converts CUAD Q&A annotations → clause classification JSON
│   ├── span_validator.py              # Validates/fixes overlapping or misaligned annotation spans
│   └── dataset_stats.py               # Prints distribution of entity types and clause categories
│
├── ner/
│   ├── __init__.py                    # Exports: NERModel, load_model
│   ├── base.py                        # BaseNERModel Protocol
│   ├── config/
│   │   └── base_config.cfg            # spaCy training config (CPU, batch=32, max_steps=2000)
│   ├── train.py                       # Trains the spaCy NER model → models/ner_baseline/
│   ├── evaluate.py                    # Precision/recall/F1 per entity type on dev set
│   └── inference.py                   # load_model(), extract_entities(text) → list[Entity]
│
├── core/
│   ├── __init__.py                    # Exports: get_settings, get_logger
│   ├── config.py                      # Pydantic Settings — all env vars validated here
│   ├── logging.py                     # structlog setup (JSON prod / console dev)
│   ├── exceptions.py                  # Typed exception hierarchy (12 classes)
│   └── types.py                       # Shared dataclasses, enums, EntityLabel (41 CUAD + 4 core)
│
├── models/                            # Saved model artifacts (gitignored except config)
│   └── ner_baseline/                  # Output of ner/train.py
│       ├── model-best/                # Best checkpoint by dev F1
│       ├── model-last/                # Last epoch checkpoint
│       └── training_meta.json         # Training stats + final F1
│
├── tests/
│   ├── conftest.py                    # Shared fixtures (sample PDFs, CUAD stubs, mock model)
│   ├── fixtures/
│   │   ├── minimal_digital.pdf        # 1-page digital PDF fixture
│   │   ├── scanned_contract.pdf       # 1-page scanned (image-only) PDF fixture
│   │   └── sample_contract.docx       # .docx with paragraphs + table
│   ├── ingestion/
│   │   ├── test_pdf_extractor.py      # pdfminer extraction + char density tests
│   │   └── test_ocr_extractor.py      # OCR pipeline + preprocessing tests
│   ├── data_processing/
│   │   ├── test_cuad_to_ner.py        # Label mapping, span alignment, DocBin output
│   │   └── test_span_validator.py     # Bounds check, overlap resolution, audit trail
│   └── ner/
│       └── test_ner_inference.py      # load_model, extract_entities, batch_extract tests
│
├── scripts/
│   ├── download_cuad.sh               # Downloads CUAD from HuggingFace Hub → data/raw/
│   ├── prepare_data.sh                # Runs full data pipeline end-to-end
│   └── train_ner.sh                   # Calls ner/train.py with env-controlled args
│
├── docker-compose.yml                 # Phase 1: base service only; Phase 3 scaffolds commented
├── Dockerfile                         # Python 3.11 + Tesseract + poppler base image
├── requirements.txt                   # Pinned production dependencies
├── requirements-dev.txt               # pytest, ruff, mypy, pre-commit
├── pyproject.toml                     # Tool config: ruff, mypy, pytest, coverage
├── .env.example                       # All required env vars documented with defaults
├── .gitignore
└── README.md
```

---

## Detailed Requirements

### 1 · CUAD Dataset Processing

**Files**: `cuad_loader.py`, `cuad_to_ner.py`, `cuad_to_classification.py`, `span_validator.py`, `dataset_stats.py`

#### 1.1 — Loading

```python
# HuggingFace dataset identifier
load_dataset("theatticusproject/cuad")
```

**CUAD Schema** (per row):

| Field | Type | Description |
|---|---|---|
| `contract_name` | `str` | Filename of the source contract |
| `full_text` | `str` | Complete contract text |
| `question` | `str` | One of 41 clause-type question templates |
| `answers` | `dict` | `{text: [str], answer_start: [int]}` — empty if clause absent |

#### 1.2 — Entity Mapping (`cuad_to_ner.py`)

Extract and map the following CUAD questions to NER entity labels:

| CUAD Question | NER Label |
|---|---|
| Parties | `ORG` |
| Governing Law | `LAW_JURISDICTION` |
| Effective Date / Expiration Date | `DATE` |
| Contract Value / Minimum Commitment | `MONEY` |
| Notice Period to Terminate Renewal | `DURATION` |
| Jurisdiction | `JURISDICTION` |
| *(remaining 35 clause types)* | *clause-specific labels* |

**Output per contract**:

```python
(text, {"entities": [(start, end, label), ...]})
# Written as spaCy DocBin → cuad_ner_train.spacy / cuad_ner_dev.spacy
```

#### 1.3 — Span Validation (`span_validator.py`)

Run a clean-up pass on every extracted span list:

1. **Bounds check** — remove spans where `end > len(full_text)` or `start < 0`
2. **Overlap resolution** — when two spans overlap, keep the **longer span**
   (spaCy does not allow overlapping entities in NER training data)
3. **Token alignment** — shift span boundaries by ±1 character to align to
   whitespace token boundaries if off-by-one

Every discarded span is recorded as a `SpanConflict` audit record.

#### 1.4 — Clause Classification (`cuad_to_classification.py`)

For each of the **41 CUAD question types**, produce records:

```json
{
  "contract_name": "N-1_4.pdf",
  "clause_type": "GOVERNING_LAW",
  "text_span": "This Agreement shall be governed by the laws of California.",
  "label": 1
}
```

- `label: 1` if an answer span exists, `label: 0` if the clause is absent
- Split: **80% train** → `cuad_clauses_train.json`, **20% dev** → `cuad_clauses_dev.json`
- Format: **JSON Lines** (one record per line)

#### 1.5 — Dataset Statistics (`dataset_stats.py`)

After processing, print a report showing:

- Entity type → count table
- Clause type → positive/negative ratio table
- Total samples, conflicts resolved, avg text length

---

### 2 · OCR & Ingestion Pipeline

**Files**: `pdf_extractor.py`, `ocr_extractor.py`, `docx_extractor.py`, `document_router.py`, `text_cleaner.py`

#### 2.1 — Digital PDF (`pdf_extractor.py`)

- Use **pdfminer.six** to extract text page-by-page
- Flag any page with **< 50 characters** of extracted text as likely scanned
- Return flagged page numbers so the router can hand them to `OcrExtractor`

#### 2.2 — Scanned PDF OCR (`ocr_extractor.py`)

**Pipeline per page**:

```
PDF page
  │
  ▼ pdf2image.convert_from_path(dpi=300)
PIL.Image (RGB, 300 DPI)
  │
  ▼ Preprocessing:
  │   1. Convert to greyscale
  │   2. Adaptive thresholding (cv2 or PIL)
  │   3. Deskew via pytesseract OSD
  │   4. Denoise
  │
  ▼ pytesseract.image_to_string(config="--oem 3 --psm 6")
raw OCR text
  │
  ▼ Reassemble with: "\n\n--- PAGE {n} ---\n\n"
full document text
```

**Tesseract config**: `--oem 3` (LSTM + legacy), `--psm 6` (uniform block of text)

#### 2.3 — DOCX (`docx_extractor.py`)

- Use **python-docx** to extract:
  - Paragraph text (in order)
  - Table cell text (row by row, pipe-delimited)
- Preserve reading order

#### 2.4 — Document Router (`document_router.py`)

Given a file path (or bytes + filename):

1. Detect filetype via **magic bytes** or file extension
2. Route to the correct extractor
3. For PDFs: run `PdfExtractor` first → fall back to `OcrExtractor` for
   any flagged low-character-density pages

#### 2.5 — Text Cleaner (`text_cleaner.py`)

Post-extraction normalisation:

| Step | Description |
|---|---|
| Ligature fix | `ﬁ→fi`, `ﬂ→fl`, `ﬃ→ffi`, etc. |
| Whitespace | Collapse excess newlines and spaces |
| Header/footer removal | Heuristic: lines < 60 chars appearing on every page |
| Metadata | Return `{pages, word_count, extraction_method: "pdfminer"\|"ocr"\|"docx"}` |

---

### 3 · spaCy NER Baseline

**Files**: `ner/config/base_config.cfg`, `train.py`, `evaluate.py`, `inference.py`

#### 3.1 — Training Config (`base_config.cfg`)

```ini
[nlp]
pipeline = ["tok2vec", "ner"]

[training]
max_steps = 2000
eval_frequency = 200
batch_size = 32

[system]
gpu_id = -1     # CPU only (Phase 1)
                # set to 0 if GPU available
```

- GPU: use if `torch.cuda.is_available()`, else CPU
- Warm-start from `en_core_web_lg` word vectors

#### 3.2 — Training (`train.py`)

1. Load `cuad_ner_train.spacy` + `cuad_ner_dev.spacy`
2. Run `spacy train` (delegates to spaCy CLI API)
3. Save **best model by NER F1** to `models/ner_baseline/model-best/`

#### 3.3 — Evaluation (`evaluate.py`)

Load saved model → run on dev set → print per-entity table:

```
Label               Precision  Recall     F1      Support
──────────────────────────────────────────────────────────
ORG                   0.912     0.887    0.899      412
LAW_JURISDICTION      0.871     0.903    0.887      284
DATE                  0.834     0.761    0.796      201
MONEY                 0.798     0.743    0.770      156
DURATION              0.756     0.701    0.728       89
──────────────────────────────────────────────────────────
MICRO AVG             0.856     0.831    0.843     3421
```

#### 3.4 — Inference (`inference.py`)

```python
def load_model(model_path: str) -> nlp:
    """Load and cache spaCy pipeline. Thread-safe singleton."""

def extract_entities(text: str) -> list[Entity]:
    """
    Returns:
        list of Entity(text, label, start_char, end_char, confidence)

    Notes:
        - confidence: from token-level IOB scores if available, else 1.0
        - Deduplicates overlapping spans by score (higher score wins)
        - Long texts auto-chunked at sentence boundaries
    """
```

---

### 4 · Tests

**Files**: `tests/ingestion/test_pdf_extractor.py`, `test_ocr_extractor.py`,
`tests/data_processing/test_cuad_to_ner.py`, `tests/ner/test_ner_inference.py`

#### 4.1 — `test_pdf_extractor.py`

- Create a minimal **in-memory PDF with reportlab**
- Assert text is extracted correctly
- Assert char density is computed and > 50 for digital PDF

#### 4.2 — `test_ocr_extractor.py`

- Convert a simple test image with **known text** to PDF
- Assert OCR output matches expected string (within Levenshtein threshold)
- Skip automatically if Tesseract binary not installed

#### 4.3 — `test_cuad_to_ner.py`

Run on **10 sample CUAD rows**:

- Assert **no overlapping spans** in output
- Assert entity labels **match expected mapping** table
- Assert output is a **valid spaCy DocBin** (can be loaded with `DocBin().from_disk()`)

#### 4.4 — `test_ner_inference.py`

- Load trained model (or tiny mock model)
- Feed a 2-sentence contract snippet
- Assert **at least one ORG** and **one DATE** entity returned

---

## How to Run (Quick Reference)

```bash
# 1. Download CUAD dataset
bash scripts/download_cuad.sh

# 2. Prepare training data
bash scripts/prepare_data.sh
# Output: data/processed/cuad_ner_train.spacy + cuad_ner_dev.spacy
#          data/processed/cuad_clauses_train.json + cuad_clauses_dev.json

# 3. Train NER baseline (~2–4 hours on CPU)
bash scripts/train_ner.sh

# 4. Evaluate
python -m ner.evaluate \
  --model models/ner_baseline \
  --dev   data/processed/cuad_ner_dev.spacy

# 5. Run tests
pytest tests/ -v --cov
```

---

## Expected Output Files

| File | Format | Description |
|---|---|---|
| `data/processed/cuad_ner_train.spacy` | spaCy DocBin | NER training corpus |
| `data/processed/cuad_ner_dev.spacy` | spaCy DocBin | NER dev/eval corpus |
| `data/processed/cuad_clauses_train.json` | JSON Lines | Clause classification train |
| `data/processed/cuad_clauses_dev.json` | JSON Lines | Clause classification dev |
| `models/ner_baseline/model-best/` | spaCy model dir | Best NER checkpoint |
| `models/ner_baseline/training_meta.json` | JSON | Training stats + final F1 |

---

## Expected Training Performance (Baseline)

> These are **reference targets** based on CUAD benchmarks.
> Actual numbers depend on the train/dev split and hardware.

| Metric | Expected Range |
|---|---|
| Micro F1 (all labels) | 0.78 – 0.86 |
| ORG F1 | 0.88 – 0.93 |
| DATE F1 | 0.76 – 0.85 |
| MONEY F1 | 0.72 – 0.82 |
| LAW_JURISDICTION F1 | 0.80 – 0.90 |
| Training time (CPU, 2000 steps) | 2 – 4 hours |

---

## Design Decisions (Locked for Phase 1)

| Decision | Choice | Rationale |
|---|---|---|
| CUAD label scope | All 41 clause types | Prevents rework in Phase 2 |
| OCR detection | Auto-detect (char density < 50/page) | No manual flags needed |
| Span conflict resolution | Keep longer span | Maximises clause context for NER |
| GPU | CPU only (`GPU_ID=-1`) | Phase 2 upgrades with transformer |
| Interfaces | `typing.Protocol` (not ABC) | Structural typing — zero forced inheritance |
| Dependency direction | `core <- ingestion, data_processing, ner` | No sibling cross-imports |
