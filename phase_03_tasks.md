# Phase 03 — Vector Search, FastAPI & Async Inference Pipeline

> **Platform**: AI-Powered Contract Intelligence & Risk Scoring
> **Week**: 3 | **Status**: 🔜 Pending Phase 2 Completion
> **Prerequisites**: `models/ner_baseline/` (Phase 1) · `models/clause_classifier/` (Phase 2)

---

## Goal

- Generate **document-level embeddings** using `BAAI/bge-large-en-v1.5` and populate a **vector database** (Pinecone / Milvus) for semantic search
- Build the full **FastAPI application** with contract upload, end-to-end processing, semantic search, and risk report endpoints
- Implement **asynchronous Celery + Redis** task queue so contract processing never blocks the HTTP layer
- Integrate all Phase 1 & 2 models into a single **pipeline orchestrator** (OCR → NER → classify → embed → risk score → persist)

---

## Context

Two trained model artifacts are available:

| Artifact | Location | Produces |
|---|---|---|
| spaCy NER baseline | `models/ner_baseline/` | `list[Entity]` |
| Transformer clause classifier | `models/clause_classifier/` | `list[ClauseResult]` |

The API must accept contract uploads (PDF / DOCX), process them end-to-end asynchronously, store results in PostgreSQL, and serve structured JSON results including a risk report.

---

## Tech Stack

### Additions to Phases 1 & 2

| Library | Version | Purpose |
|---|---|---|
| **FastAPI** | latest | Async HTTP API framework |
| **Uvicorn** | latest | ASGI server for FastAPI |
| **Celery** | latest | Distributed async task queue |
| **Redis** | 7.x (Docker) | Celery broker + result backend |
| **pinecone-client** | latest | Vector database (default) |
| **pymilvus** | latest | Vector database (self-hosted fallback) |
| **sentence-transformers** | latest | `BAAI/bge-large-en-v1.5` — document embeddings |
| **SQLAlchemy** (async) | latest | ORM for contract metadata + job status |
| **asyncpg** | latest | Async PostgreSQL driver |
| **PostgreSQL** | 15 (Docker) | Persistent contract metadata storage |
| **Alembic** | latest | Database schema migrations |
| **python-multipart** | latest | Multipart file upload handling |
| **boto3** | latest | S3 raw contract file storage |
| **slowapi** | latest | Rate limiting for API endpoints |

### Carried Over from Phases 1 & 2

`spaCy`, `transformers`, `scikit-learn`, `structlog`, `pydantic-settings`, `python-dotenv`, `tqdm`

---

## Project Structure

```
contract-intelligence/
│
├── api/
│   ├── __init__.py                    # Package marker
│   ├── main.py                        # FastAPI app factory (create_app())
│   ├── config.py                      # ApiSettings — extends Phase 1 Settings
│   ├── dependencies.py                # DB session, Celery app, API key auth dependency
│   ├── middleware.py                  # Request ID injection + structured access logging
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── contracts.py               # POST /v1/contracts/upload
│   │   │                              # GET  /v1/contracts/{id}
│   │   │                              # GET  /v1/contracts/{id}/status
│   │   │                              # GET  /v1/contracts/{id}/results
│   │   │                              # DELETE /v1/contracts/{id}
│   │   ├── search.py                  # POST /v1/search/semantic
│   │   │                              # POST /v1/search/clause
│   │   └── risk.py                    # GET  /v1/contracts/{id}/risk-report
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── contract.py                # ContractUploadResponse, ContractStatus, ContractResults
│   │   ├── search.py                  # SemanticSearchRequest/Response, ClauseSearchRequest/Response
│   │   └── risk.py                    # RiskReport, RiskFinding, RiskLevel enum
│   │
│   └── models/
│       ├── __init__.py
│       ├── contract.py                # SQLAlchemy Contract ORM model
│       └── job.py                     # SQLAlchemy ProcessingJob ORM model
│
├── worker/
│   ├── __init__.py
│   ├── celery_app.py                  # Celery instance (Redis broker + backend)
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── document_task.py           # process_contract.delay(contract_id) — main pipeline task
│   │   └── embedding_task.py          # generate_and_index_embeddings(contract_id)
│   └── pipeline/
│       ├── __init__.py
│       └── pipeline_orchestrator.py   # Orchestrates: extract → clean → NER → classify → embed → risk → persist
│
├── embedding/
│   ├── __init__.py                    # Exports: get_embedder, get_vector_store
│   ├── chunker.py                     # Splits text into overlapping ~512-token chunks + metadata
│   ├── embedder.py                    # BAAI/bge-large-en-v1.5 — generates (N, 1024) embeddings
│   └── vector_store.py                # VectorStoreBase Protocol + PineconeAdapter + MilvusAdapter
│
├── risk_scoring/
│   ├── __init__.py                    # Exports: compute_risk_report
│   ├── risk_engine.py                 # Aggregates NER + clause results → RiskReport
│   └── risk_rules.py                  # 8 risk rules per clause/entity pattern (see requirements)
│
├── storage/
│   ├── __init__.py                    # Exports: get_storage_client
│   └── s3_client.py                   # S3Adapter + LocalFilesystemAdapter (STORAGE_BACKEND=local)
│
├── db.py                              # Async SQLAlchemy engine + session factory
│
├── alembic/
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
│       └── 001_initial_schema.py      # Contract + ProcessingJob tables
│
├── tests/
│   ├── conftest.py                    # Shared fixtures: TestClient, mock Celery, mock Pinecone
│   ├── test_contracts_router.py       # Upload fixture PDF → assert 202 + job enqueued
│   ├── test_search_router.py          # Mock vector store query → assert response shape
│   ├── test_risk_engine.py            # Synthetic ClauseResults → assert correct RiskFindings
│   ├── test_pipeline_orchestrator.py  # Mock all Phase 1/2 models → assert stage order
│   └── test_vector_store.py           # Mock Pinecone client → assert upsert/query shapes
│
└── docker-compose.yml                 # Extended: postgres + redis + (optional) milvus services
```

---

## Detailed Requirements

### 1 · Database Models

**Files**: `api/models/contract.py`, `api/models/job.py`

#### 1.1 — `Contract` ORM Model

```python
class Contract(Base):
    __tablename__ = "contracts"

    id:                 UUID           # Primary key (auto-generated)
    filename:           str            # Original uploaded filename
    s3_key:             str            # Storage path (S3 key or local path)
    upload_time:        datetime       # UTC timestamp of upload
    status:             ContractStatus # Enum: PENDING | PROCESSING | DONE | FAILED
    page_count:         int | None     # Set after extraction
    word_count:         int | None     # Set after extraction
    extraction_method:  str | None     # "pdfminer" | "ocr" | "docx"
    entities:           dict | None    # JSON column — list[Entity] after NER
    clauses:            dict | None    # JSON column — list[ClauseResult] after classification
    risk_report:        dict | None    # JSON column — full RiskReport
    created_at:         datetime       # Auto-set on insert
    updated_at:         datetime       # Auto-set on update
```

#### 1.2 — `ProcessingJob` ORM Model

```python
class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id:               UUID           # Primary key
    contract_id:      UUID           # FK → contracts.id (cascade delete)
    celery_task_id:   str            # Celery task UUID for status polling
    stage:            JobStage       # Enum: OCR | NER | CLASSIFY | EMBED | RISK
    started_at:       datetime       # When this stage began
    completed_at:     datetime | None
    error_message:    str | None     # Set on failure
```

#### 1.3 — Status Enums

```python
class ContractStatus(str, Enum):
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    DONE       = "DONE"
    FAILED     = "FAILED"

class JobStage(str, Enum):
    OCR      = "OCR"
    NER      = "NER"
    CLASSIFY = "CLASSIFY"
    EMBED    = "EMBED"
    RISK     = "RISK"
```

---

### 2 · API Configuration

**File**: `api/config.py`

#### `ApiSettings` — extends Phase 1 `Settings`

```python
class ApiSettings(Settings):
    # API
    api_title:   str = "Contract Intelligence API"
    api_version: str = "1.0.0"
    valid_api_keys: list[str]         # from env: VALID_API_KEYS="key1,key2"

    # Database
    database_url: str                 # e.g. "postgresql+asyncpg://user:pass@localhost/contracts"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"

    # Storage
    storage_backend: str = "s3"       # "s3" | "local"
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    local_storage_path: str = "/tmp/contract-uploads"

    # Vector DB
    vector_db_backend: str = "pinecone"  # "pinecone" | "milvus"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "contract-intelligence"
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # Embedding
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_batch_size: int = 32     # GPU; set 8 for CPU

    # Risk scoring
    company_jurisdiction: str = "Delaware"  # Used in Rule 4

    # Rate limiting
    rate_limit_per_minute: int = 60
```

---

### 3 · Contract Upload & Processing Flow

**Files**: `api/routers/contracts.py`, `worker/tasks/document_task.py`, `worker/pipeline/pipeline_orchestrator.py`

#### 3.1 — `POST /v1/contracts/upload`

```
Request:  multipart/form-data
          file: UploadFile (PDF or DOCX, max 50 MB)

Response: 202 Accepted
          {
            "contract_id": "uuid-string",
            "status": "PENDING",
            "message": "Contract uploaded. Processing started."
          }
```

**Steps**:

1. Validate file type (`application/pdf` or `application/vnd.openxmlformats-officedocument...`)
2. Validate file size ≤ 50 MB
3. Upload to S3 (or local) via `storage_client.upload(file, key)`
4. Insert `Contract` record with `status=PENDING`
5. `process_contract.delay(str(contract.id))`
6. Return `202` with `contract_id`

#### 3.2 — All Contract Endpoints

| Method | Path | Description | Auth |
|---|---|---|---|
| `POST` | `/v1/contracts/upload` | Upload + enqueue processing | ✅ Required |
| `GET` | `/v1/contracts/{id}` | Full contract record | ✅ Required |
| `GET` | `/v1/contracts/{id}/status` | Status + current stage + progress % | ✅ Required |
| `GET` | `/v1/contracts/{id}/results` | Full results (NER + clauses) — 404 if not DONE | ✅ Required |
| `DELETE` | `/v1/contracts/{id}` | Delete contract + S3 file + vector embeddings | ✅ Required |
| `GET` | `/health` | `{"status": "ok", "version": "1.0.0"}` | ❌ No auth |

#### 3.3 — `process_contract` Celery Task

**File**: `worker/tasks/document_task.py`

```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_contract(self, contract_id: str) -> dict:
    """
    Full async processing pipeline for a single contract.
    Retries up to 3 times on transient failure (network, OOM).
    Updates Contract.status and ProcessingJob records at each stage.
    """
```

#### 3.4 — Pipeline Orchestrator

**File**: `worker/pipeline/pipeline_orchestrator.py`

```
Stage 1 — OCR       Download from S3 → DocumentRouter.route() → ExtractionResult
             │       Update: Contract.extraction_method, page_count, word_count
             │       Log:    "ocr_complete" with method, pages, chars
             ▼
Stage 2 — NER       TextCleaner.clean() → NERModel.extract_entities() → list[Entity]
             │       Update: Contract.entities (JSON)
             │       Log:    "ner_complete" with entity_count
             ▼
Stage 3 — CLASSIFY  classify_clauses(text) → list[ClauseResult]
             │       Update: Contract.clauses (JSON)
             │       Log:    "classification_complete" with present_clauses count
             ▼
Stage 4 — EMBED     chunker.chunk(text) → embedder.generate() → vector_store.upsert()
             │       Log:    "embedding_complete" with chunk_count, vector_count
             ▼
Stage 5 — RISK      risk_engine.compute_risk_report(entities, clauses) → RiskReport
             │       Update: Contract.risk_report (JSON), Contract.status = DONE
             │       Log:    "risk_complete" with risk_score, overall_level
```

**On any stage failure**:
- Update `ProcessingJob.error_message`
- Update `Contract.status = FAILED`
- Re-raise for Celery retry logic

#### 3.5 — Status Response

```json
GET /v1/contracts/{id}/status
{
  "contract_id": "uuid",
  "status": "PROCESSING",
  "current_stage": "CLASSIFY",
  "progress_pct": 60,
  "started_at": "2025-08-07T10:00:00Z",
  "elapsed_seconds": 42
}
```

**Progress % mapping**:

| Stage | Progress |
|---|---|
| PENDING | 0% |
| OCR | 20% |
| NER | 40% |
| CLASSIFY | 60% |
| EMBED | 80% |
| RISK / DONE | 100% |

---

### 4 · Chunking & Embedding

**Files**: `embedding/chunker.py`, `embedding/embedder.py`, `embedding/vector_store.py`

#### 4.1 — `chunker.py`

**Parameters**:

| Parameter | Value |
|---|---|
| `chunk_size` | 512 tokens |
| `stride` | 128 tokens (overlap = 384 tokens) |
| Metadata per chunk | `{contract_id, chunk_index, char_start, char_end, page_hint}` |

**Algorithm**:

```
1. Tokenise full text with bge tokenizer (no truncation)
2. Slide window of 512 tokens, step 128
3. Decode each window back to text
4. Record char_start / char_end by tracking token → char offsets
5. Attach page_hint = approximate page number (char_start / avg_chars_per_page)
```

**Output**:

```python
@dataclass
class TextChunk:
    text:         str
    contract_id:  str
    chunk_index:  int
    char_start:   int
    char_end:     int
    page_hint:    int
```

#### 4.2 — `embedder.py`

```python
# Module-level singleton — loaded once at worker startup
_MODEL: SentenceTransformer | None = None

def get_embedder() -> SentenceTransformer:
    """Thread-safe singleton loader for BAAI/bge-large-en-v1.5."""

def generate_embeddings(chunks: list[str]) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, 1024)

    Batching:
        batch_size = 32 (GPU) or 8 (CPU) from settings
        Uses sentence_transformers encode() with show_progress_bar=False
        Normalise embeddings (L2) — required for cosine similarity in Pinecone
    """
```

#### 4.3 — `vector_store.py`

**Protocol** (shared interface for both backends):

```python
class VectorStoreBase(Protocol):
    def upsert(self, vectors: list[VectorRecord]) -> None: ...
    def query(self, embedding: np.ndarray, top_k: int = 10,
              filter_metadata: dict | None = None) -> list[QueryResult]: ...
    def delete(self, contract_id: str) -> None: ...
```

**`VectorRecord`** dataclass:

```python
@dataclass
class VectorRecord:
    id:         str           # "{contract_id}::{chunk_index}"
    embedding:  np.ndarray    # shape (1024,)
    metadata:   dict          # See below
```

**Metadata stored per vector**:

```json
{
  "contract_id":          "uuid",
  "chunk_index":          3,
  "char_start":           1024,
  "char_end":             2048,
  "filename":             "acme_msa.pdf",
  "clause_types_present": ["GOVERNING_LAW", "LIMITATION_OF_LIABILITY"]
}
```

`clause_types_present` = list of clause types whose character spans overlap this chunk
(computed by comparing `ClauseResult` evidence spans against chunk `char_start/char_end`).

#### 4.4 — Pinecone Config

```python
# Index creation (idempotent — skip if already exists)
pinecone.create_index(
    name      = settings.pinecone_index_name,
    dimension = 1024,
    metric    = "cosine",
    spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

#### 4.5 — Milvus Fallback Config

```python
# Collection schema
fields = [
    FieldSchema("id",        DataType.VARCHAR, max_length=200, is_primary=True),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema("metadata",  DataType.JSON),
]
index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "nlist": 128}
```

---

### 5 · Semantic Search API

**File**: `api/routers/search.py`

#### 5.1 — `POST /v1/search/semantic`

```
Request:
{
  "query":        "governing law California indemnification",
  "top_k":        10,
  "contract_ids": ["uuid-1", "uuid-2"]   # optional filter
}

Response:
{
  "results": [
    {
      "contract_id":  "uuid-1",
      "filename":     "acme_msa.pdf",
      "chunk_text":   "This Agreement shall be governed by the laws of California...",
      "score":        0.923,
      "char_start":   4512,
      "char_end":     5024
    },
    ...
  ],
  "query_embedding_ms": 42,
  "vector_search_ms":   18
}
```

**Steps**:

1. `embed query → (1, 1024)` via `get_embedder()`
2. Build optional metadata filter: `{"contract_id": {"$in": contract_ids}}`
3. `vector_store.query(embedding, top_k, filter_metadata)`
4. Fetch `chunk_text` from PostgreSQL or reconstruct from S3 using `char_start/char_end`
5. Return results with timing metadata

#### 5.2 — `POST /v1/search/clause`

```
Request:
{
  "clause_type": "GOVERNING_LAW",
  "top_k":       10
}

Response:
{
  "results": [
    {
      "contract_id":   "uuid",
      "filename":      "acme_msa.pdf",
      "confidence":    0.923,
      "evidence_span": "This Agreement shall be governed by..."
    },
    ...
  ]
}
```

**Steps**:

1. Query vector store with `filter_metadata={"clause_types_present": {"$in": [clause_type]}}`
2. Rank by `ClauseResult.confidence` (from PostgreSQL `clauses` JSON column)
3. Return top-k ranked contracts

---

### 6 · Risk Scoring Engine

**Files**: `risk_scoring/risk_engine.py`, `risk_scoring/risk_rules.py`

#### 6.1 — `RiskFinding` & `RiskReport` Schemas

```python
class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass(frozen=True)
class RiskFinding:
    clause_type:     str
    risk_level:      RiskLevel
    description:     str        # One-sentence finding
    evidence:        str        # Relevant text snippet
    recommendation: str        # Actionable recommendation

@dataclass
class RiskReport:
    contract_id:       str
    overall_risk_level: RiskLevel
    risk_score:        int                  # 0–100
    findings:          list[RiskFinding]
    key_entities:      KeyEntities          # parties, dates, jurisdictions, monetary_values
    summary:           str                  # Template-generated human-readable summary
```

#### 6.2 — All 8 Risk Rules

| # | Severity | Trigger | Finding | Recommendation |
|---|---|---|---|---|
| 1 | **CRITICAL** | Auto-Renewal present **AND** Termination for Convenience absent | "Evergreen trap: contract auto-renews with no unilateral exit right" | "Negotiate a termination for convenience clause with 30-day notice" |
| 2 | **CRITICAL** | Limitation of Liability **absent** entirely | "No liability cap; unlimited financial exposure" | "Insert a liability cap clause capped at contract value or 12 months' fees" |
| 3 | **HIGH** | Uncapped Liability present + no monetary ceiling in NER `MONEY` entities | "Liability clause found but no defined monetary ceiling" | "Specify an explicit dollar cap in the limitation of liability clause" |
| 4 | **HIGH** | Governing Law jurisdiction ≠ `settings.company_jurisdiction` | "Unfavorable governing law: disputes resolved under {jurisdiction}" | "Negotiate governing law to {company_jurisdiction} or a neutral forum" |
| 5 | **HIGH** | IP Ownership Assignment present + assigned to counterparty | "IP created under this contract belongs to the counterparty" | "Negotiate for joint ownership or a license-back provision" |
| 6 | **MEDIUM** | Indemnification present **AND** Limitation of Liability present **AND** indemnification carved out | "Indemnification obligations may exceed the liability cap" | "Ensure indemnification is subject to the liability cap" |
| 7 | **MEDIUM** | Notice Period to Terminate < 30 days (extracted from `DURATION` entity) | "Short termination notice window ({N} days) creates operational risk" | "Negotiate a minimum 30-day notice period for termination" |
| 8 | **LOW** | Confidentiality clause present + no `DATE` entity for confidentiality term | "Confidentiality obligation has no defined expiry" | "Define a confidentiality term (typically 3–5 years post-termination)" |

#### 6.3 — Risk Score Calculation

```python
RISK_WEIGHTS = {
    RiskLevel.CRITICAL: 40,
    RiskLevel.HIGH:     20,
    RiskLevel.MEDIUM:   10,
    RiskLevel.LOW:       5,
}

risk_score = min(100, sum(RISK_WEIGHTS[f.risk_level] for f in findings))

overall_risk_level = (
    RiskLevel.CRITICAL if risk_score >= 80 else
    RiskLevel.HIGH     if risk_score >= 60 else
    RiskLevel.MEDIUM   if risk_score >= 40 else
    RiskLevel.LOW
)
```

#### 6.4 — Summary Template

```python
summary = (
    f"This contract presents a {overall_risk_level} risk (score {risk_score}/100). "
    f"{len(findings)} findings were identified: "
    f"{critical_count} critical, {high_count} high, "
    f"{medium_count} medium, {low_count} low."
)
```

---

### 7 · Risk Report API

**File**: `api/routers/risk.py`

#### `GET /v1/contracts/{id}/risk-report`

```json
{
  "contract_id":         "uuid",
  "overall_risk_level":  "HIGH",
  "risk_score":          65,
  "summary":             "This contract presents a HIGH risk (score 65/100). 4 findings were identified: 0 critical, 2 high, 1 medium, 1 low.",
  "findings": [
    {
      "clause_type":    "GOVERNING_LAW",
      "risk_level":     "HIGH",
      "description":    "Unfavorable governing law: disputes resolved under New York",
      "evidence":       "This Agreement shall be governed by the laws of the State of New York.",
      "recommendation": "Negotiate governing law to Delaware or a neutral forum"
    }
  ],
  "key_entities": {
    "parties":          ["Acme Corporation", "Beta Technologies LLC"],
    "dates":            ["January 15, 2024", "December 31, 2026"],
    "jurisdictions":    ["New York"],
    "monetary_values":  ["$500,000", "$50,000 per month"]
  }
}
```

---

### 8 · Authentication

**File**: `api/dependencies.py`

Simple API key authentication via `X-API-Key` header:

```python
async def require_api_key(
    x_api_key: str = Header(...),
    settings: ApiSettings = Depends(get_settings),
) -> str:
    """
    FastAPI dependency injected into all protected routes.
    Returns the key on success, raises HTTP 401 on failure.
    """
    if x_api_key not in settings.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
```

**Unprotected routes**: `GET /health` only.

---

### 9 · Middleware

**File**: `api/middleware.py`

Two middleware layers applied globally:

| Middleware | Behaviour |
|---|---|
| **Request ID** | Inject `X-Request-ID` UUID into every request; include in response headers |
| **Access Logging** | Structured `structlog` log per request: `method, path, status_code, duration_ms, request_id` |

```python
# Rate limiting via slowapi
limiter = Limiter(key_func=get_remote_address)

@app.get("/v1/contracts/upload")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def upload(...): ...
```

---

### 10 · Tests

**Files**: `tests/test_contracts_router.py`, `tests/test_search_router.py`,
`tests/test_risk_engine.py`, `tests/test_pipeline_orchestrator.py`, `tests/test_vector_store.py`

#### 10.1 — `test_contracts_router.py`

```
SETUP:   FastAPI TestClient, Celery task mocked, DB in-memory (SQLite)
ASSERT:  POST /v1/contracts/upload with fixture PDF → 202 Accepted
ASSERT:  response body contains "contract_id" and "status": "PENDING"
ASSERT:  Celery process_contract.delay called once with contract_id
ASSERT:  POST without X-API-Key → 401 Unauthorized
ASSERT:  POST with file > 50 MB → 422 Unprocessable Entity
```

#### 10.2 — `test_pipeline_orchestrator.py`

```
SETUP:   Mock DocumentRouter, NERModel, classify_clauses, embedder, risk_engine
ASSERT:  All 5 stages called in order (OCR → NER → CLASSIFY → EMBED → RISK)
ASSERT:  Contract.status == DONE after successful run
ASSERT:  Contract.status == FAILED if any stage raises an exception
ASSERT:  ProcessingJob records created for each stage with correct stage enum
ASSERT:  Failed stage records error_message correctly
```

#### 10.3 — `test_vector_store.py`

```
SETUP:   Mock Pinecone client (pinecone.Index)
ASSERT:  upsert() called with vectors of shape (N, 1024)
ASSERT:  Vector IDs follow "{contract_id}::{chunk_index}" format
ASSERT:  query() returns list of QueryResult with score and metadata
ASSERT:  delete() calls pinecone.delete with contract_id filter
```

#### 10.4 — `test_risk_engine.py`

```python
# Test Case 1 — Evergreen trap (Rule 1)
clauses = [
    ClauseResult("Renewal Term",              present=True,  confidence=0.92, ...),
    ClauseResult("Termination for Convenience", present=False, confidence=0.10, ...),
]
report = compute_risk_report(entities=[], clauses=clauses, contract_id="test")
ASSERT: any(f.risk_level == CRITICAL and "evergreen" in f.description.lower() for f in report.findings)

# Test Case 2 — No liability cap (Rule 2)
clauses = [ClauseResult("Limitation Of Liability", present=False, confidence=0.08, ...)]
ASSERT: any(f.risk_level == CRITICAL and "no liability cap" in f.description.lower() for f in report.findings)

# Test Case 3 — All clear
clauses = [all 41 clauses present with high confidence]
ASSERT: report.overall_risk_level == LOW and report.risk_score < 20
```

#### 10.5 — `test_search_router.py`

```
SETUP:   Mock vector_store.query returning 3 fake QueryResult objects
ASSERT:  POST /v1/search/semantic → 200 with len(results) == 3
ASSERT:  Each result has: contract_id, filename, chunk_text, score, char_start, char_end
ASSERT:  POST /v1/search/clause with unknown clause_type → 422
```

---

## `docker-compose.yml` — Full Updated Services

```yaml
services:
  # --- Phase 1 base (already exists) ---
  base:
    build: .
    volumes: [".:/app", "./data:/app/data", "./models:/app/models"]

  # --- Phase 3 additions ---
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB:       contracts
      POSTGRES_USER:     contract_user
      POSTGRES_PASSWORD: contract_pass
    ports: ["5432:5432"]
    volumes: ["postgres_data:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U contract_user"]
      interval: 10s

  redis:
    image: redis:7.2-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ports: ["8000:8000"]
    depends_on: [postgres, redis]
    env_file: .env

  worker:
    build: .
    command: celery -A worker.celery_app worker --loglevel=info --concurrency=2
    depends_on: [redis, postgres]
    env_file: .env
    volumes: ["./models:/app/models", "./data:/app/data"]

  # Optional — Milvus for self-hosted vector DB
  # milvus:
  #   image: milvusdb/milvus:v2.4.0
  #   ports: ["19530:19530"]

volumes:
  postgres_data:
```

---

## How to Run (Quick Reference)

```bash
# 1. Install Phase 3 dependencies
pip install -r requirements.txt

# 2. Start infrastructure (postgres + redis)
docker-compose up -d postgres redis

# 3. Run database migrations
alembic upgrade head

# 4. Start API server (development)
uvicorn api.main:app --reload --port 8000

# 5. Start Celery worker (separate terminal)
celery -A worker.celery_app worker --loglevel=info

# 6. Upload a contract (example curl)
curl -X POST http://localhost:8000/v1/contracts/upload \
  -H "X-API-Key: your-api-key" \
  -F "file=@tests/fixtures/minimal_digital.pdf"

# 7. Poll for status
curl http://localhost:8000/v1/contracts/{contract_id}/status \
  -H "X-API-Key: your-api-key"

# 8. Retrieve results
curl http://localhost:8000/v1/contracts/{contract_id}/results \
  -H "X-API-Key: your-api-key"

# 9. Get risk report
curl http://localhost:8000/v1/contracts/{contract_id}/risk-report \
  -H "X-API-Key: your-api-key"

# 10. Semantic search
curl -X POST http://localhost:8000/v1/search/semantic \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "governing law California", "top_k": 5}'

# 11. Run tests
pytest tests/ -v
```

---

## Environment Variables (Phase 3 Additions)

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | `postgresql+asyncpg://user:pass@host/db` |
| `REDIS_URL` | ✅ | `redis://localhost:6379/0` | Celery broker + result backend |
| `VALID_API_KEYS` | ✅ | — | Comma-separated API keys |
| `STORAGE_BACKEND` | ✅ | `s3` | `s3` or `local` |
| `S3_BUCKET` | ⚠️ S3 only | — | S3 bucket name for raw contracts |
| `S3_REGION` | ⚠️ S3 only | `us-east-1` | AWS region |
| `LOCAL_STORAGE_PATH` | ⚠️ local only | `/tmp/contract-uploads` | Local filesystem fallback |
| `VECTOR_DB_BACKEND` | ✅ | `pinecone` | `pinecone` or `milvus` |
| `PINECONE_API_KEY` | ⚠️ Pinecone only | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | ⚠️ Pinecone only | `contract-intelligence` | Pinecone index name |
| `MILVUS_HOST` | ⚠️ Milvus only | `localhost` | Milvus server host |
| `MILVUS_PORT` | ⚠️ Milvus only | `19530` | Milvus server port |
| `EMBEDDING_MODEL` | ❌ | `BAAI/bge-large-en-v1.5` | HuggingFace embedding model |
| `EMBEDDING_BATCH_SIZE` | ❌ | `32` | Embedding batch size (8 for CPU) |
| `COMPANY_JURISDICTION` | ❌ | `Delaware` | Used in Rule 4 (Governing Law risk) |
| `RATE_LIMIT_PER_MINUTE` | ❌ | `60` | API rate limit per IP per minute |

> **Local dev tip**: Set `STORAGE_BACKEND=local` to skip S3 entirely.
> Files are stored at `LOCAL_STORAGE_PATH`. No AWS credentials needed.

---

## Expected Output Files

| File | Format | Description |
|---|---|---|
| `api/main.py` | Python | FastAPI app with all routers registered |
| `worker/celery_app.py` | Python | Celery instance with Redis broker |
| `worker/tasks/document_task.py` | Python | `process_contract` task with retry logic |
| `worker/pipeline/pipeline_orchestrator.py` | Python | 5-stage pipeline with stage tracking |
| `embedding/chunker.py` | Python | Overlapping 512-token chunker |
| `embedding/embedder.py` | Python | bge-large singleton + batch encode |
| `embedding/vector_store.py` | Python | Pinecone + Milvus adapters |
| `risk_scoring/risk_engine.py` | Python | Risk aggregation + summary template |
| `risk_scoring/risk_rules.py` | Python | 8 rule functions + `RiskFinding` builders |
| `alembic/versions/001_initial_schema.py` | Python | Contract + ProcessingJob migration |
| `docker-compose.yml` | YAML | postgres + redis + api + worker services |

---

## Design Decisions (Locked for Phase 3)

| Decision | Choice | Rationale |
|---|---|---|
| Task queue | Celery + Redis | Battle-tested; Redis doubles as result backend with no extra service |
| Vector DB default | Pinecone | Serverless, no infra to manage; Milvus available for air-gapped deployments |
| Embedding model | `BAAI/bge-large-en-v1.5` | Top-ranked on MTEB legal retrieval; 1024-dim cosine-optimised |
| Chunk strategy | 512 tokens, stride 128 | 75% overlap ensures no entity is split across chunk boundaries |
| Auth method | API key (`X-API-Key`) | Simple for Phase 3; Phase 4 upgrades to JWT/OAuth |
| Storage abstraction | `StorageBase` Protocol | `STORAGE_BACKEND=local` switches to filesystem with zero code change |
| Risk score cap | 100 | Standardised 0–100 scale maps cleanly to dashboard gauges in Phase 4 |
| Summary generation | Template string | No LLM call in Phase 3; deterministic output; Phase 4 adds LLM summaries |
