# Phase 04 — Docker, AWS Deployment, Frontend & Production Hardening

> **Platform**: AI-Powered Contract Intelligence & Risk Scoring
> **Week**: 4 | **Status**: 🔜 Pending Phase 3 Completion
> **Prerequisites**: All Phase 1–3 artifacts complete + running `docker-compose` stack

---

## Goal

- **Containerise** every service with multi-stage Docker builds (no internet access at runtime)
- **Deploy** to AWS EC2 via ECR with production `docker-compose` + NGINX reverse proxy
- **Build a React frontend** with live contract highlights, risk report panel, and semantic search UI
- **Harden** the system with structured logging, Prometheus metrics, and Locust load testing
- **Document** the full platform in a comprehensive final README

---

## Context

All three backend phases are complete and produce:

| Phase | Artifact | Output |
|---|---|---|
| 1 | `models/ner_baseline/` | `list[Entity]` |
| 2 | `models/clause_classifier/` | `list[ClauseResult]` + `calibrators.pkl` |
| 3 | FastAPI + Celery + Pinecone | REST API + vector search + `RiskReport` |

Phase 4 wraps everything into a **production-deployable, user-facing product**.

---

## Tech Stack

### Additions to Phases 1–3

| Tool / Library | Purpose |
|---|---|
| **Docker** multi-stage | Containerise API, worker, frontend |
| **docker-compose.prod.yml** | Production service orchestration |
| **NGINX** | Reverse proxy, TLS, gzip, static frontend serving |
| **React 18 + TypeScript** | Frontend framework |
| **Vite** | Frontend build tool |
| **react-pdf** | PDF rendering with highlights overlay |
| **Tailwind CSS** | Utility-first styling (slate/red/amber/yellow/blue theme) |
| **react-dropzone** | Drag-and-drop file upload |
| **Locust** | Load testing (50 concurrent users, 3 user classes) |
| **prometheus-fastapi-instrumentator** | Prometheus metrics endpoint |
| **structlog + python-json-logger** | Consistent structured JSON logging |
| **AWS ECR** | Container image registry |
| **AWS EC2** | Deployment target (t3.xlarge recommended) |
| **AWS Secrets Manager** | Secrets injection at deploy time |
| **CloudWatch** | Container log aggregation |

---

## Project Structure

```
contract-intelligence/
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.api          # Multi-stage: builder (deps + models) → runtime (non-root)
│   │   ├── Dockerfile.worker       # Same base as api, CMD = celery worker
│   │   └── .dockerignore           # Exclude: data/, models/, .venv, __pycache__, tests/
│   │
│   ├── docker-compose.prod.yml     # api (×2), worker (×2), postgres, redis, nginx
│   │
│   ├── nginx/
│   │   └── nginx.conf              # Proxy /api/ → uvicorn, /static/ → frontend, gzip, TLS scaffold
│   │
│   └── ec2/
│       ├── deploy.sh               # ECR login → docker pull → docker-compose up -d
│       └── user_data.sh            # EC2 bootstrap: install docker, pull images, start services
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Router: / → UploadPage, /results/:id → ResultsPage, /search → SearchPage
│   │   │
│   │   ├── pages/
│   │   │   ├── UploadPage.tsx      # Drag-and-drop upload + StatusPoller stage stepper
│   │   │   ├── ResultsPage.tsx     # ContractViewer (left) + RiskReport + ClauseList (right)
│   │   │   └── SearchPage.tsx      # Semantic query box + SearchResults list
│   │   │
│   │   ├── components/
│   │   │   ├── ContractViewer.tsx  # react-pdf renderer + coloured highlight boxes (char offset → coords)
│   │   │   ├── RiskReport.tsx      # Risk gauge ring + collapsible findings sorted by severity
│   │   │   ├── ClauseList.tsx      # 41-row table: present/absent badge, confidence bar, evidence hover
│   │   │   ├── EntityPanel.tsx     # 4 sections: Parties, Dates, Money, Jurisdictions
│   │   │   ├── StatusPoller.tsx    # Polls /status every 3s, updates horizontal stage stepper
│   │   │   └── SearchResults.tsx   # Ranked chunk results with highlighted match text
│   │   │
│   │   ├── api/
│   │   │   └── client.ts           # Typed fetch wrappers for all backend routes
│   │   │
│   │   └── hooks/
│   │       ├── useContractResults.ts
│   │       └── useSearch.ts
│   │
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
│
├── monitoring/
│   └── prometheus.yml              # Scrape config for /metrics endpoint
│
└── tests/
    ├── load/
    │   └── locustfile.py           # 3 Locust user classes: Upload, Poll, Search
    └── e2e/
        └── test_upload_to_results.py  # Full pipeline E2E against running docker-compose stack
```

---

## Detailed Requirements

### 1 · Docker

**Files**: `infra/docker/Dockerfile.api`, `infra/docker/Dockerfile.worker`, `infra/docker/.dockerignore`

#### 1.1 — `Dockerfile.api` — Multi-Stage Build

```dockerfile
# ═══════════════════════════════════════════
# STAGE 1 — builder
# ═══════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps required for PDF processing + OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv in /venv for clean copy to runtime
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install Python deps (CPU-only torch — EC2 without GPU)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download spaCy model (no internet at runtime)
RUN python -m spacy download en_core_web_lg

# Pre-download sentence-transformers model to HuggingFace cache
RUN python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-large-en-v1.5')
print('bge-large downloaded.')
"

# Pre-download HuggingFace transformer model (roberta-base fallback)
RUN python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained('roberta-base')
AutoModelForSequenceClassification.from_pretrained('roberta-base')
print('roberta-base downloaded.')
"

# ═══════════════════════════════════════════
# STAGE 2 — runtime
# ═══════════════════════════════════════════
FROM python:3.11-slim AS runtime

# System runtime libs only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtualenv + HuggingFace model cache from builder
COPY --from=builder /venv /venv
COPY --from=builder /root/.cache /root/.cache

# Non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

WORKDIR /app
COPY --chown=appuser:appuser . .

ENV PATH="/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "warning"]
```

#### 1.2 — `Dockerfile.worker`

Identical to `Dockerfile.api` up to the final `CMD`:

```dockerfile
# All layers identical to Dockerfile.api (use COPY --from or shared base)
# Only difference:
CMD ["celery", "-A", "worker.celery_app", "worker", \
     "--loglevel=info", "--concurrency=2"]
```

> **Why concurrency=2?** Each worker process loads NER + transformer models (~2 GB RAM each).
> Two workers per container = ~4 GB; stays within the 3 GB worker memory limit.

#### 1.3 — `.dockerignore`

```dockerignore
# Development artifacts (never in image)
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Data (mounted as volumes in production)
data/raw/
data/processed/

# Models (mounted as volumes OR pre-baked in builder stage)
models/

# Frontend build artefacts (built separately)
frontend/node_modules/
frontend/dist/

# Secrets
.env
.env.prod

# Tests
tests/

# Infra docs
infra/ec2/
```

---

### 2 · `docker-compose.prod.yml`

**File**: `infra/docker-compose.prod.yml`

```yaml
version: "3.9"

services:

  postgres:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB:       ${POSTGRES_DB}
      POSTGRES_USER:     ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
    # No external port exposed — only accessible within docker network

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build:
      context: ..
      dockerfile: infra/docker/Dockerfile.api
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "1.0"
          memory: 1024M
    depends_on:
      postgres: { condition: service_healthy }
      redis:    { condition: service_healthy }
    env_file: .env.prod
    volumes:
      - model_cache:/root/.cache     # HuggingFace + spaCy cached models
      - contract_uploads:/tmp/contract-uploads
    logging:
      driver: awslogs
      options:
        awslogs-region:       ${AWS_REGION}
        awslogs-group:        /contract-intelligence/api
        awslogs-stream-prefix: api

  worker:
    build:
      context: ..
      dockerfile: infra/docker/Dockerfile.worker
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "2.0"
          memory: 3072M
    depends_on:
      postgres: { condition: service_healthy }
      redis:    { condition: service_healthy }
    env_file: .env.prod
    volumes:
      - model_cache:/root/.cache
      - contract_uploads:/tmp/contract-uploads
    logging:
      driver: awslogs
      options:
        awslogs-region:       ${AWS_REGION}
        awslogs-group:        /contract-intelligence/worker
        awslogs-stream-prefix: worker

  nginx:
    image: nginx:1.25-alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./frontend_dist:/var/www/html:ro     # Built frontend static files
      - /etc/letsencrypt:/etc/letsencrypt:ro # TLS certs (Let's Encrypt)
    depends_on:
      - api
    logging:
      driver: awslogs
      options:
        awslogs-region:       ${AWS_REGION}
        awslogs-group:        /contract-intelligence/nginx
        awslogs-stream-prefix: nginx

volumes:
  postgres_data:
  model_cache:
  contract_uploads:
```

---

### 3 · NGINX Configuration

**File**: `infra/nginx/nginx.conf`

```nginx
# nginx.conf
# ==========
# Reverse proxy for Contract Intelligence Platform
# - /api/    → FastAPI (uvicorn, port 8000)
# - /static/ → React build output
# - gzip     → JSON + text compression
# - 60MB     → max upload size (contracts)
# - TLS      → commented block for Let's Encrypt (certbot)

worker_processes auto;
events { worker_connections 1024; }

http {
    include       mime.types;
    default_type  application/octet-stream;
    sendfile      on;
    keepalive_timeout 65;

    # ── Gzip compression ──────────────────────────────────────────
    gzip on;
    gzip_types application/json text/plain text/css application/javascript;
    gzip_min_length 1024;
    gzip_comp_level 5;

    # ── Upstream: FastAPI API servers ─────────────────────────────
    upstream api_backend {
        # Docker Compose service name (load balanced across 2 replicas)
        server api:8000;
        keepalive 32;
    }

    server {
        listen 80;
        server_name _;

        # Max upload: 60MB (contracts)
        client_max_body_size 60M;

        # ── API routes ─────────────────────────────────────────────
        location /api/ {
            proxy_pass         http://api_backend/;
            proxy_set_header   Host              $host;
            proxy_set_header   X-Real-IP         $remote_addr;
            proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
            proxy_set_header   X-Forwarded-Proto $scheme;
            proxy_read_timeout 300s;   # Long timeout for file uploads
            proxy_send_timeout 300s;
        }

        # ── Health check (no auth) ──────────────────────────────
        location /health {
            proxy_pass http://api_backend/health;
        }

        # ── Prometheus metrics (internal only) ─────────────────
        location /metrics {
            proxy_pass http://api_backend/metrics;
            allow 10.0.0.0/8;   # VPC CIDR only
            deny  all;
        }

        # ── Frontend static files ───────────────────────────────
        location / {
            root  /var/www/html;
            index index.html;
            try_files $uri $uri/ /index.html;  # SPA routing fallback
        }
    }

    # ── TLS block (enable after certbot issues certificate) ─────
    # server {
    #     listen 443 ssl http2;
    #     server_name your-domain.com;
    #
    #     ssl_certificate     /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    #     ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    #     ssl_protocols       TLSv1.2 TLSv1.3;
    #     ssl_ciphers         HIGH:!aNULL:!MD5;
    #
    #     client_max_body_size 60M;
    #
    #     location /api/ {
    #         proxy_pass http://api_backend/;
    #         proxy_set_header X-Forwarded-Proto https;
    #     }
    #
    #     location / {
    #         root /var/www/html;
    #         try_files $uri /index.html;
    #     }
    # }
}
```

---

### 4 · EC2 Deployment

**Files**: `infra/ec2/deploy.sh`, `infra/ec2/user_data.sh`

#### 4.1 — `user_data.sh` — EC2 Bootstrap Script

```bash
#!/bin/bash
# user_data.sh
# ============
# EC2 user-data script. Runs once at first boot.
# Recommended instance: t3.xlarge (4 vCPU, 16 GB RAM)
# AMI: Amazon Linux 2023 or Ubuntu 22.04 LTS

set -euo pipefail

# Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user   # or ubuntu

# Install Docker Compose plugin
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
     -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Install AWS CLI (for Secrets Manager + ECR)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && ./aws/install
rm -rf awscliv2.zip aws/

# Create app directory
mkdir -p /opt/contract-intelligence
cd /opt/contract-intelligence

# Pull docker-compose.prod.yml from S3 (or embed inline)
# aws s3 cp s3://${CONFIG_BUCKET}/docker-compose.prod.yml .

# Run deploy script
bash /opt/contract-intelligence/deploy.sh
```

#### 4.2 — `deploy.sh` — CI/CD Deploy Script

```bash
#!/bin/bash
# deploy.sh
# =========
# Called by CI/CD pipeline (GitHub Actions, CodePipeline) OR manually.
# Never hard-codes secrets — all secrets from AWS Secrets Manager.
#
# USAGE
# -----
#   AWS_REGION=us-east-1 \
#   ECR_REGISTRY=123456789.dkr.ecr.us-east-1.amazonaws.com \
#   IMAGE_TAG=$(git rev-parse --short HEAD) \
#   bash infra/ec2/deploy.sh

set -euo pipefail

ECR_REGISTRY="${ECR_REGISTRY:?ECR_REGISTRY is required}"
AWS_REGION="${AWS_REGION:-us-east-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
APP_DIR="/opt/contract-intelligence"

echo "=== Contract Intelligence Deploy ==="
echo "Registry: $ECR_REGISTRY | Tag: $IMAGE_TAG | Region: $AWS_REGION"

# ── Step 1: Authenticate with ECR ────────────────────────────────
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# ── Step 2: Pull images from ECR ─────────────────────────────────
docker pull "$ECR_REGISTRY/contract-api:$IMAGE_TAG"
docker pull "$ECR_REGISTRY/contract-worker:$IMAGE_TAG"

# ── Step 3: Fetch secrets from AWS Secrets Manager ───────────────
# Pattern: fetch JSON secret → write to .env.prod
# SECRET_NAME should be set in the environment
echo "Fetching secrets from AWS Secrets Manager..."
aws secretsmanager get-secret-value \
    --region "$AWS_REGION" \
    --secret-id "contract-intelligence/prod" \
    --query "SecretString" \
    --output text \
  | python3 -c "
import json, sys
secrets = json.load(sys.stdin)
with open('$APP_DIR/.env.prod', 'w') as f:
    for k, v in secrets.items():
        f.write(f'{k}={v}\n')
print('Wrote .env.prod from Secrets Manager.')
"
chmod 600 "$APP_DIR/.env.prod"

# ── Step 4: One-time Pinecone index bootstrap (idempotent) ───────
# Only runs if index does not already exist
# docker run --rm --env-file "$APP_DIR/.env.prod" \
#   "$ECR_REGISTRY/contract-api:$IMAGE_TAG" \
#   python -c "
# from embedding.vector_store import PineconeAdapter
# PineconeAdapter().create_index_if_not_exists()
# "

# ── Step 5: Run database migrations ──────────────────────────────
docker run --rm --env-file "$APP_DIR/.env.prod" \
  "$ECR_REGISTRY/contract-api:$IMAGE_TAG" \
  alembic upgrade head

# ── Step 6: Start / restart services ────────────────────────────
cd "$APP_DIR"
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d --remove-orphans

echo ""
echo "=== Deploy complete ==="
echo "API health: curl http://localhost/health"
docker compose -f docker-compose.prod.yml ps
```

#### 4.3 — Recommended EC2 Instance Types

| Workload | Instance | vCPU | RAM | Notes |
|---|---|---|---|---|
| **Standard** | `t3.xlarge` | 4 | 16 GB | Default; handles 2 api + 2 worker replicas |
| **Heavy** | `c5.2xlarge` | 8 | 16 GB | Compute-optimised; better for CPU inference |
| **GPU (Phase 5)** | `g4dn.xlarge` | 4 | 16 GB + T4 | Enables GPU inference in Phase 5 |

---

### 5 · Structured Logging

**Applied throughout**: `api/main.py`, `api/middleware.py`, `worker/tasks/document_task.py`

#### 5.1 — Standard Log Fields (all services)

Every structured log event must include these fields:

| Field | Type | Example | Source |
|---|---|---|---|
| `event` | `str` | `"request_complete"` | Log call site |
| `level` | `str` | `"info"` | Logger |
| `timestamp` | `ISO8601` | `"2025-08-07T10:00:00Z"` | structlog processor |
| `service` | `str` | `"api"` / `"worker"` | App startup config |
| `request_id` | `str` | `"uuid"` | Middleware injection |
| `contract_id` | `str\|None` | `"uuid"` | Task context |

#### 5.2 — API Access Log Event

```json
{
  "event":       "request_complete",
  "service":     "api",
  "request_id":  "a3f2c1d4-...",
  "method":      "POST",
  "path":        "/v1/contracts/upload",
  "status_code": 202,
  "duration_ms": 143,
  "client_ip":   "10.0.1.42"
}
```

#### 5.3 — Pipeline Stage Log Events

```json
// OCR complete
{
  "event":           "stage_complete",
  "service":         "worker",
  "contract_id":     "uuid",
  "stage":           "OCR",
  "duration_ms":     2341,
  "extraction_method": "pdfminer",
  "page_count":      12,
  "char_count":      48320
}

// NER complete
{
  "event":        "stage_complete",
  "stage":        "NER",
  "duration_ms":  891,
  "entity_count": 23
}

// CLASSIFY complete
{
  "event":          "stage_complete",
  "stage":          "CLASSIFY",
  "duration_ms":    3412,
  "clauses_found":  7,
  "top_clause":     "GOVERNING_LAW",
  "top_confidence": 0.923
}

// RISK complete
{
  "event":              "stage_complete",
  "stage":              "RISK",
  "duration_ms":        45,
  "risk_score":         65,
  "overall_risk_level": "HIGH",
  "findings_count":     4
}
```

#### 5.4 — Logging Setup (`core/logging.py` — updated)

```python
import structlog

def configure_logging(service: str, log_format: str = "json") -> None:
    """
    Call once at application startup.
    log_format: "json" (production) | "console" (development)
    """
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_logger_name,
    ]
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(processors=processors)
```

**Invariant rule**: Every `structlog` call uses the same field names from §5.1.
No `print()` statements anywhere in the codebase.

---

### 6 · Prometheus Metrics

**File**: `api/main.py` (via `prometheus-fastapi-instrumentator`)

#### 6.1 — Metrics Defined

| Metric | Type | Labels | Description |
|---|---|---|---|
| `http_request_duration_seconds` | Histogram | `method, path, status` | Request latency |
| `contracts_uploaded_total` | Counter | — | Total contracts uploaded |
| `contracts_processed_total` | Counter | `status: success\|failure` | Processed contract outcomes |
| `celery_task_duration_seconds` | Histogram | `task_name` | Celery task execution time |
| `risk_score_distribution` | Histogram | — | Risk score distribution (buckets: 0,20,40,60,80,100) |

#### 6.2 — Setup in `api/main.py`

```python
from prometheus_fastapi_instrumentator import Instrumentator

def create_app() -> FastAPI:
    app = FastAPI(title=settings.api_title, version=settings.api_version)

    # Prometheus — expose /metrics
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/health", "/metrics"],
    ).instrument(app).expose(app)

    return app
```

#### 6.3 — `monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "contract-intelligence-api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
    # Note: /metrics is only accessible from within the Docker network
    # (NGINX blocks external access — see nginx.conf)
```

---

### 7 · Frontend — React + TypeScript

**Language**: TypeScript (strict mode)
**Styling**: Tailwind CSS — slate palette, `red/amber/yellow/blue` severity system
**Font**: Monospace (`font-mono`) for contract text excerpts; Inter for UI chrome

#### 7.1 — `api/client.ts` — Typed API Client

Full typed wrappers for every backend endpoint:

```typescript
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const API_KEY  = import.meta.env.VITE_API_KEY ?? "";

// Types (mirroring backend Pydantic schemas)
export interface UploadResponse    { contract_id: string; status: string; message: string; }
export interface ContractStatus    { contract_id: string; status: string; current_stage: string; progress_pct: number; elapsed_seconds: number; }
export interface Entity            { label: string; text: string; start_char: number; end_char: number; confidence: number; }
export interface ClauseResult      { clause_type: string; present: boolean; confidence: number; evidence_spans: string[]; }
export interface RiskFinding       { clause_type: string; risk_level: "LOW"|"MEDIUM"|"HIGH"|"CRITICAL"; description: string; evidence: string; recommendation: string; }
export interface RiskReport        { contract_id: string; overall_risk_level: string; risk_score: number; findings: RiskFinding[]; key_entities: KeyEntities; summary: string; }
export interface SemanticResult    { contract_id: string; filename: string; chunk_text: string; score: number; char_start: number; char_end: number; }
export interface ContractResults   { contract_id: string; entities: Entity[]; clauses: ClauseResult[]; risk_report: RiskReport; }

// API functions
export const uploadContract    = (file: File)                          => Promise<UploadResponse>
export const getContractStatus = (id: string)                          => Promise<ContractStatus>
export const getContractResults= (id: string)                          => Promise<ContractResults>
export const getRiskReport     = (id: string)                          => Promise<RiskReport>
export const semanticSearch    = (query: string, topK?: number)        => Promise<{ results: SemanticResult[] }>
export const clauseSearch      = (clauseType: string, topK?: number)   => Promise<{ results: any[] }>
```

#### 7.2 — `UploadPage.tsx`

```
┌──────────────────────────────────────────────────────────────────┐
│  Contract Intelligence                                           │
│  ─────────────────────────────────────────────────────────────── │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │         📄  Drag & drop your contract here               │   │
│  │              or click to browse                          │   │
│  │         Accepts: PDF, DOCX  ·  Max: 50 MB               │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  [After upload — StatusPoller mounts]                            │
│                                                                  │
│  ● OCR ──────── ● NER ──────── ● CLASSIFY ─── ● EMBED ── ● RISK │
│  ✓ Done         ✓ Done         🔄 Processing   ○ Pending  ○ Pend │
│                                                                  │
│  ██████████████████████░░░░░░░  60% — Classifying clauses...    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Behaviour**:
- `react-dropzone` accepts `.pdf`, `.docx`, max 50 MB (client-side check)
- On drop: call `uploadContract(file)` → store `contract_id` in state → navigate to results preview
- Mount `<StatusPoller contractId={id} />` which drives the stepper

#### 7.3 — `StatusPoller.tsx`

```typescript
// Polls GET /v1/contracts/{id}/status every 3000ms
// Updates parent via onStatusChange callback
// Stops polling when status === "DONE" or "FAILED"
// Uses setInterval + clearInterval on unmount (no memory leak)
```

**Stage stepper**: 5 stages mapped to progress %:

```
OCR(20%) → NER(40%) → CLASSIFY(60%) → EMBED(80%) → RISK(100%)
```

Each stage node: `✓` (complete, green), `🔄` (current, blue pulse), `○` (pending, grey)

#### 7.4 — `ResultsPage.tsx` — Layout

```
┌────────────────────────────────────────────────────────────────────┐
│  ← Back    acme_msa.pdf    Risk: HIGH ██████░░░░  65/100           │
├──────────────────────────────┬─────────────────────────────────────┤
│                              │  ⚠ RISK REPORT                      │
│   [ContractViewer]           │  ────────────────────────────────── │
│                              │  ▼ CRITICAL  Evergreen trap         │
│   PDF rendered page-by-page  │  ▼ HIGH      Unfavorable law (NY)  │
│   with coloured highlight    │  ▼ MEDIUM    Short notice period   │
│   overlays on spans          │  ▼ LOW       No confidentiality... │
│                              │                                     │
│   🟣 Parties                 │  [EntityPanel]                      │
│   🔴 CRITICAL risk clause    │  Parties: Acme Corp, Beta LLC       │
│   🟠 HIGH risk clause        │  Dates: Jan 15, 2024               │
│   🟡 MEDIUM risk clause      │  Money: $500,000                   │
│   🔵 LOW risk clause / date  │  Jurisdictions: New York           │
│                              │                                     │
├──────────────────────────────┴─────────────────────────────────────┤
│  ALL CLAUSES (41)                                                  │
│  ───────────────────────────────────────────────────────────────── │
│  Clause Type         Present  Confidence              Evidence     │
│  Governing Law       ✓         ████████░░  92%        [hover]     │
│  Limitation of Liab  ✗         ██░░░░░░░░  20%        —           │
│  ...                                                               │
└────────────────────────────────────────────────────────────────────┘
```

#### 7.5 — `ContractViewer.tsx` — Highlight Overlay Logic

**Challenge**: Map character offsets (`char_start`, `char_end`) to PDF page coordinates.

**Phase 4 approximation** (replaced with exact coords in Phase 5):

```typescript
// Approximate position from char offset
const CHARS_PER_LINE = 80;
const LINE_HEIGHT_PX  = 16;
const PAGE_HEIGHT_PX  = 842;    // A4 at 72 DPI
const CHARS_PER_PAGE  = 3000;   // Approximate

function charOffsetToPageCoords(charOffset: number, pageWidth: number): Rect {
  const approxPage  = Math.floor(charOffset / CHARS_PER_PAGE);
  const offsetInPage = charOffset % CHARS_PER_PAGE;
  const line        = Math.floor(offsetInPage / CHARS_PER_LINE);
  const col         = offsetInPage % CHARS_PER_LINE;
  return {
    page: approxPage + 1,
    x:    (col / CHARS_PER_LINE) * pageWidth,
    y:    line * LINE_HEIGHT_PX,
    w:    pageWidth * 0.6,       // Approximate span width
    h:    LINE_HEIGHT_PX,
  };
}
```

**Colour coding**:

| Annotation Type | Tailwind Colour | Hex |
|---|---|---|
| `CRITICAL` finding | `bg-red-400/40` | `#f87171` (40% opacity) |
| `HIGH` finding | `bg-amber-400/40` | `#fbbf24` |
| `MEDIUM` finding | `bg-yellow-300/40` | `#fde047` |
| `LOW` finding | `bg-blue-400/40` | `#60a5fa` |
| Entity (any) | `bg-purple-400/40` | `#c084fc` |

#### 7.6 — `RiskReport.tsx`

- **Gauge ring**: SVG circle with `stroke-dasharray` driven by `risk_score` (0–100)
  - Colour: `stroke-red-500` (≥80), `stroke-amber-500` (≥60), `stroke-yellow-400` (≥40), `stroke-blue-400` (<40)
- **Findings list**: Sorted CRITICAL → HIGH → MEDIUM → LOW
- Each finding: collapsible `<details>` element showing `evidence` text + `recommendation`

#### 7.7 — `SearchPage.tsx`

```
┌──────────────────────────────────────────────────────────┐
│  Semantic Search                                         │
│  ──────────────────────────────────────────────────────  │
│  [  governing law California indemnification    ] [→]    │
│                                                          │
│  Filter by contract: [All contracts         ▼]          │
│                                                          │
│  Results (3 matches):                                    │
│  ──────────────────────────────────────────────────────  │
│  ● acme_msa.pdf  (score: 0.923)                         │
│    "...This Agreement shall be governed by the laws of  │
│     California, and the parties submit to the exclusive  │
│     jurisdiction of..."                                  │
│    [View Contract →]                                     │
└──────────────────────────────────────────────────────────┘
```

Match text: the `chunk_text` rendered with the query terms **bold**
(client-side substring highlight using `String.prototype.split()` + `<mark>` tags).

#### 7.8 — `vite.config.ts`

```typescript
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/v1":     { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    outDir: "../infra/frontend_dist",  // Output to docker nginx volume
  },
});
```

---

### 8 · Load Testing

**File**: `tests/load/locustfile.py`

#### 8.1 — Three User Classes

```python
class UploadUser(HttpUser):
    """
    Simulates: client uploads a contract PDF.
    Weight 1 (1 in 6 users).
    Uses: tests/fixtures/minimal_digital.pdf (< 1 MB, fast)
    """
    weight    = 1
    wait_time = between(5, 15)

    @task
    def upload_contract(self):
        with open("tests/fixtures/minimal_digital.pdf", "rb") as f:
            self.client.post(
                "/v1/contracts/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                headers={"X-API-Key": LOAD_TEST_API_KEY},
            )


class PollUser(HttpUser):
    """
    Simulates: client polls for status after upload.
    Weight 3 (3 in 6 users — polling dominates real traffic).
    contract_ids injected from a shared fixture list.
    """
    weight    = 3
    wait_time = between(1, 5)

    @task
    def poll_status(self):
        contract_id = random.choice(FIXTURE_CONTRACT_IDS)
        self.client.get(
            f"/v1/contracts/{contract_id}/status",
            headers={"X-API-Key": LOAD_TEST_API_KEY},
            name="/v1/contracts/[id]/status",
        )


class SearchUser(HttpUser):
    """
    Simulates: client performs semantic searches.
    Weight 2 (2 in 6 users).
    """
    weight    = 2
    wait_time = between(3, 10)

    QUERIES = [
        "governing law California",
        "termination for convenience",
        "limitation of liability cap",
        "intellectual property assignment",
        "automatic renewal clause",
        "indemnification obligations",
        "confidentiality term expiry",
        "notice period to terminate",
    ]

    @task
    def semantic_search(self):
        self.client.post(
            "/v1/search/semantic",
            json={"query": random.choice(self.QUERIES), "top_k": 5},
            headers={"X-API-Key": LOAD_TEST_API_KEY},
        )
```

#### 8.2 — Run Command

```bash
locust -f tests/load/locustfile.py \
  --headless \
  --host http://localhost \
  -u 50 \
  -r 5 \
  --run-time 2m \
  --html tests/load/report.html
```

#### 8.3 — Target SLAs

| Endpoint | p50 target | p95 target | p99 target |
|---|---|---|---|
| `POST /v1/contracts/upload` | < 200ms | < 500ms | < 1000ms |
| `GET /v1/contracts/{id}/status` | < 30ms | < 100ms | < 200ms |
| `POST /v1/search/semantic` | < 200ms | < 600ms | < 1200ms |
| `GET /v1/contracts/{id}/risk-report` | < 50ms | < 150ms | < 300ms |

> Upload p95 target is 500ms because the Celery task is **async** —
> the HTTP response returns immediately after file receipt; actual processing
> happens in the background.

---

### 9 · E2E Test

**File**: `tests/e2e/test_upload_to_results.py`

**Target**: Running `docker-compose` stack (local or CI)

```python
"""
Full pipeline E2E test.

SETUP
-----
Requires docker-compose stack running:
    docker compose up -d

Set env vars:
    E2E_API_URL = "http://localhost:8000"  (or via .env.test)
    E2E_API_KEY = "test-key"

STEPS
-----
1. Upload tests/fixtures/minimal_digital.pdf
2. Poll GET /v1/contracts/{id}/status every 3s until status=DONE or 120s timeout
3. Assert results contain ≥ 1 entity
4. Assert ≥ 1 ClauseResult with confidence > 0.5
5. Assert risk_report present with overall_risk_level not None
6. Search "governing law" → assert uploaded contract appears in results (by contract_id)
"""

ASSERTIONS = [
    "status == DONE within 120 seconds",
    "len(results.entities) >= 1",
    "any(c.confidence > 0.5 for c in results.clauses)",
    "results.risk_report is not None",
    "results.risk_report.overall_risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']",
    "uploaded contract_id appears in semantic search results for 'governing law'",
]
```

---

### 10 · Final README Structure

**File**: `README.md` (comprehensive final version replacing Phase 1 README)

#### Required Sections

**1. System Architecture Diagram (ASCII)**

```
┌──────────┐   HTTPS    ┌────────┐    /api/    ┌─────────────┐
│  Browser  │──────────▶│ NGINX  │────────────▶│  FastAPI    │
│ (React)   │◀──────────│        │             │  (Uvicorn)  │
└──────────┘  HTML/JSON └────────┘             └──────┬──────┘
                                                      │ Celery task
                                                      ▼
┌──────────┐            ┌─────────┐          ┌──────────────┐
│ Pinecone │◀───────────│  Redis  │◀─────────│    Celery    │
│ (vectors)│            │(broker) │          │   Worker     │
└──────────┘            └─────────┘          └──────┬───────┘
                                                    │ reads/writes
              ┌──────────────┐                      ▼
              │  PostgreSQL  │◀───────────── ORM models
              │  (metadata)  │              (Contract, Job)
              └──────────────┘
```

**2. Component Responsibilities Table**

| Component | Technology | Responsibility |
|---|---|---|
| **API** | FastAPI + Uvicorn | REST endpoints, auth, file upload, result serving |
| **Worker** | Celery | Async pipeline: OCR → NER → classify → embed → risk |
| **PostgreSQL** | postgres:15 | Contract metadata, job status, results JSON |
| **Redis** | redis:7 | Celery task broker + result backend |
| **Pinecone** | Serverless | Vector storage and semantic search |
| **NGINX** | nginx:1.25 | Reverse proxy, TLS, static file serving, rate limiting |

**3. Local Dev Quickstart (3 commands)**

```bash
cp .env.example .env                    # 1. Configure
docker compose up -d                     # 2. Start all services
open http://localhost:3000               # 3. Visit frontend
```

**4. EC2 Deployment Checklist**

```
[ ] Launch t3.xlarge with Amazon Linux 2023
[ ] Attach IAM role: ECR pull + Secrets Manager read + CloudWatch logs
[ ] Run user_data.sh via EC2 launch template
[ ] Create Pinecone index (one-time): bash deploy.sh --bootstrap
[ ] Point domain DNS to EC2 Elastic IP
[ ] Run certbot for TLS: certbot --nginx -d your-domain.com
[ ] Enable CloudWatch log groups for api, worker, nginx
```

**5. All Environment Variables** (complete reference table — all phases)

**6. Model Card**

| Model | Architecture | Dev F1 | Training Data |
|---|---|---|---|
| NER baseline | spaCy en_core_web_lg | F1 0.843 (micro) | CUAD 22,450 Q&A |
| Clause classifier | InLegalBERT fine-tuned | Macro F1 ~0.79 | CUAD 41 clause types |
| Embedder | BAAI/bge-large-en-v1.5 | MTEB rank #4 | Pre-trained, no fine-tune |

**7. Load Test Results Table**

| Endpoint | p50 | p95 | p99 | RPS @ 50 users |
|---|---|---|---|---|
| POST /upload | — | — | — | *fill after run* |
| GET /status | — | — | — | *fill after run* |
| POST /search | — | — | — | *fill after run* |

**8. Known Limitations & Future Work**

| Limitation | Phase 5 Plan |
|---|---|
| CPU-only inference (slow) | GPU EC2 instance + CUDA builds |
| Char-offset → PDF coord approximation | Use pdfminer word-level bbox extraction |
| No LLM-generated summaries | GPT-4 / Claude via Phase 5 API |
| Single-region Pinecone | Multi-region replication |
| API key auth | JWT / OAuth2 with refresh tokens |
| No streaming upload | Chunked upload + S3 multipart |
| Avro schema registry | Replace JSON columns with schema-validated Avro |

---

## How to Run (Quick Reference)

```bash
# ── Local Development ────────────────────────────────────────────
cp .env.example .env
docker compose up -d postgres redis
alembic upgrade head
uvicorn api.main:app --reload --port 8000
celery -A worker.celery_app worker --loglevel=info

# ── Frontend ─────────────────────────────────────────────────────
cd frontend
npm install
npm run dev     # → http://localhost:3000

# ── Full Stack (Docker) ──────────────────────────────────────────
docker compose up -d
open http://localhost

# ── Production Deploy ────────────────────────────────────────────
docker build -f infra/docker/Dockerfile.api   -t contract-api:latest .
docker build -f infra/docker/Dockerfile.worker -t contract-worker:latest .
bash infra/ec2/deploy.sh

# ── Load Test ────────────────────────────────────────────────────
locust -f tests/load/locustfile.py --headless -u 50 -r 5 --run-time 2m

# ── E2E Test ─────────────────────────────────────────────────────
docker compose up -d
pytest tests/e2e/ -v
```

---

## Design Decisions (Locked for Phase 4)

| Decision | Choice | Rationale |
|---|---|---|
| Frontend framework | React 18 + TypeScript + Vite | Type-safety for API client; Vite for fast HMR in dev |
| Styling | Tailwind CSS (slate palette) | Consistent utility classes; no custom CSS |
| PDF highlights | Char-offset approximation | Exact bbox extraction is Phase 5; approximation is good enough for demo |
| Docker model caching | Builder stage pre-download | Runtime image has no internet — critical for EC2 air-gap mode |
| Secrets management | AWS Secrets Manager | No secrets in env files committed to repo; single source of truth |
| CloudWatch logging | `awslogs` Docker driver | Native AWS integration; no Logstash/Fluentd needed in Phase 4 |
| Multi-stage Docker | builder → runtime | Reduces runtime image size by ~3 GB (no build tools, git, pip cache) |
| NGINX rate limiting | slowapi in FastAPI | App-level rate limiting; NGINX handles connection-level throttling |
| E2E test timeout | 120 seconds | Full pipeline on CPU typically completes in 60–90 seconds |
