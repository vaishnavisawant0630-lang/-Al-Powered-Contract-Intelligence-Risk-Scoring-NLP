# AI-Powered Contract Intelligence & Risk Scoring

> **An end-to-end AI-powered system for automated contract analysis, clause classification, risk scoring, semantic search, and contract visualization.**

---

## 📌 Project Overview

**AI-Powered Contract Intelligence & Risk Scoring** is an end-to-end legal NLP system designed to automatically process contracts, identify important clauses, classify contract sections, detect potential risks, and provide an interactive interface for contract analysis.

The system combines **document ingestion, OCR, NLP, transformer-based classification, vector search, risk scoring, REST APIs, asynchronous processing, Docker deployment, and a web-based frontend**.

---

# 🚀 Project Status

## ✅ All Phases Completed

| Phase       | Week   | Main Focus                                            | Status     |
| ----------- | ------ | ----------------------------------------------------- | ---------- |
| **Phase 1** | Week 1 | CUAD Dataset Processing, OCR Pipeline & NER Baseline  | ✅ Complete |
| **Phase 2** | Week 2 | Legal Transformer Fine-Tuning & Clause Classification | ✅ Complete |
| **Phase 3** | Week 3 | Vector Search, Risk Scoring, FastAPI & Celery         | ✅ Complete |
| **Phase 4** | Week 4 | Docker, AWS Deployment & Frontend UI                  | ✅ Complete |

> **🎉 Project Status: Completed — All 4 Development Phases Successfully Implemented**

---

# 📊 Complete Phase-Wise Progress Tracker

| Phase       | Task                       | Description                                                       | Files / Components                                                                                       | Status     |
| ----------- | -------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------- |
| **Phase 1** | CUAD Dataset Processing    | Processed CUAD contracts and generated training datasets          | `cuad_loader.py`, `cuad_to_ner.py`, `cuad_to_classification.py`, `span_validator.py`, `dataset_stats.py` | ✅ Complete |
| **Phase 1** | OCR & Document Ingestion   | Extracted text from PDF, scanned PDF, DOCX and TXT files          | `pdf_extractor.py`, `ocr_extractor.py`, `docx_extractor.py`, `document_router.py`, `text_cleaner.py`     | ✅ Complete |
| **Phase 1** | NER Baseline               | Built and trained spaCy-based NER pipeline                        | `base_config.cfg`, `train.py`, `evaluate.py`, `inference.py`                                             | ✅ Complete |
| **Phase 1** | Testing                    | Added unit and integration tests for ingestion and NLP components | `tests/`                                                                                                 | ✅ Complete |
| **Phase 1** | NER Model Training         | Trained baseline NER model and evaluated entity extraction        | `models/ner_baseline/model-best/`                                                                        | ✅ Complete |
| **Phase 2** | Clause Dataset Preparation | Prepared CUAD clause classification datasets                      | `cuad_clauses_train.json`, `cuad_clauses_dev.json`                                                       | ✅ Complete |
| **Phase 2** | Legal Transformer          | Fine-tuned legal-domain transformer model                         | `InLegalBERT` / Legal RoBERTa                                                                            | ✅ Complete |
| **Phase 2** | Clause Classification      | Implemented 41-way contract clause classification                 | `clause_classifier/`                                                                                     | ✅ Complete |
| **Phase 2** | Model Calibration          | Added confidence calibration for classification predictions       | `calibrators.pkl`                                                                                        | ✅ Complete |
| **Phase 2** | Risk Detection             | Implemented heuristic and model-based risk identification         | `risk_scoring/`                                                                                          | ✅ Complete |
| **Phase 3** | Vector Database            | Added semantic contract search using vector embeddings            | Pinecone / Milvus                                                                                        | ✅ Complete |
| **Phase 3** | Embeddings                 | Generated vector representations of contract clauses              | Embedding pipeline                                                                                       | ✅ Complete |
| **Phase 3** | Semantic Search            | Implemented similarity-based clause and contract retrieval        | Vector search module                                                                                     | ✅ Complete |
| **Phase 3** | FastAPI Backend            | Developed REST API for contract processing and analysis           | `api/`, FastAPI                                                                                          | ✅ Complete |
| **Phase 3** | Celery Processing          | Added asynchronous document-processing tasks                      | Celery + Redis                                                                                           | ✅ Complete |
| **Phase 3** | Risk Scoring API           | Exposed automated risk analysis through API endpoints             | Risk API                                                                                                 | ✅ Complete |
| **Phase 4** | Dockerization              | Containerized backend and supporting services                     | `Dockerfile`, `docker-compose.yml`                                                                       | ✅ Complete |
| **Phase 4** | Cloud Deployment           | Deployed application infrastructure on AWS EC2                    | AWS EC2                                                                                                  | ✅ Complete |
| **Phase 4** | Frontend                   | Developed interactive contract analysis dashboard                 | Frontend UI                                                                                              | ✅ Complete |
| **Phase 4** | Clause Highlighting        | Added visual highlighting of detected clauses and risks           | Highlighting UI                                                                                          | ✅ Complete |
| **Phase 4** | End-to-End Integration     | Connected frontend, API, NLP models and database                  | Full system                                                                                              | ✅ Complete |
| **Phase 4** | Final Testing              | Performed system and integration testing                          | Test Suite                                                                                               | ✅ Complete |

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │      Frontend UI     │
                         │ Contract Dashboard   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │      REST API        │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
          │   Document   │  │     NER      │  │    Clause    │
          │   Ingestion  │  │    Model     │  │ Classifier   │
          └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │    Risk Scoring      │
                         │       Engine         │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  ▼                 ▼                 ▼
          ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
          │ Vector Store │  │ Celery/Redis │  │   Database   │
          │ Pinecone/    │  │ Async Tasks  │  │   Storage    │
          │ Milvus       │  │              │  │              │
          └──────────────┘  └──────────────┘  └──────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      AWS EC2         │
                         │    Deployment        │
                         └──────────────────────┘
```

---

# 📁 Project Structure

```text
contract-intelligence/
│
├── core/
│   ├── types/
│   ├── config/
│   ├── logging/
│   └── exceptions/
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
│   └── dataset_stats.py
│
├── ner/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── base_config.cfg
│
├── clause_classifier/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── risk_scoring/
│   ├── risk_engine.py
│   ├── rules.py
│   └── scoring.py
│
├── vector_search/
│   ├── embeddings.py
│   ├── indexer.py
│   └── search.py
│
├── api/
│   ├── main.py
│   ├── routes/
│   └── schemas/
│
├── workers/
│   ├── celery_app.py
│   └── tasks.py
│
├── frontend/
│   ├── components/
│   ├── pages/
│   └── services/
│
├── models/
│   ├── ner_baseline/
│   └── clause_classifier/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│
├── scripts/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

---

# 🔄 End-to-End Workflow

| Step | Component         | Function                                            |
| ---- | ----------------- | --------------------------------------------------- |
| 1    | Document Upload   | User uploads contract                               |
| 2    | Document Router   | Identifies PDF, scanned PDF, DOCX or TXT            |
| 3    | Text Extraction   | Extracts text from the document                     |
| 4    | OCR               | Processes scanned documents when required           |
| 5    | Text Cleaning     | Removes unnecessary formatting/noise                |
| 6    | NER Model         | Detects important entities and contract information |
| 7    | Clause Classifier | Classifies clauses into contract categories         |
| 8    | Embedding Model   | Converts clauses into vector representations        |
| 9    | Vector Database   | Stores and retrieves semantically similar clauses   |
| 10   | Risk Engine       | Identifies potential contractual risks              |
| 11   | Risk Score        | Assigns risk level/score                            |
| 12   | FastAPI           | Provides analysis through REST endpoints            |
| 13   | Celery            | Handles long-running processing asynchronously      |
| 14   | Frontend          | Displays contract insights and risk highlights      |
| 15   | Deployment        | Runs the complete system using Docker/AWS           |

---

# 🧠 Phase 1 — Data Parsing & Baseline Modeling

| Task                    | Work Completed                                                      | Status |
| ----------------------- | ------------------------------------------------------------------- | ------ |
| CUAD Dataset Processing | Processed CUAD dataset containing 510 contracts and 41 clause types | ✅      |
| NER Conversion          | Converted CUAD annotations into NER training format                 | ✅      |
| Classification Dataset  | Generated clause classification datasets                            | ✅      |
| Span Validation         | Added validation and conflict resolution                            | ✅      |
| Document Ingestion      | PDF, scanned PDF, DOCX and TXT support                              | ✅      |
| OCR                     | Automatic OCR fallback for scanned PDFs                             | ✅      |
| Text Cleaning           | Normalized extracted contract text                                  | ✅      |
| spaCy NER               | Implemented baseline NER model                                      | ✅      |
| Model Training          | Completed baseline NER training                                     | ✅      |
| Evaluation              | Evaluated trained model                                             | ✅      |
| Testing                 | Unit and integration tests completed                                | ✅      |

### Phase 1 Output

* 510 CUAD contracts processed
* 41 clause categories
* 7 NER labels
* PDF + OCR + DOCX + TXT ingestion
* spaCy NER baseline
* Automated testing pipeline

---

# 🤖 Phase 2 — Legal NLP & Clause Classification

| Task                   | Work Completed                                      | Status |
| ---------------------- | --------------------------------------------------- | ------ |
| Dataset Preparation    | Prepared clause classification datasets             | ✅      |
| Legal Transformer      | Integrated legal-domain transformer model           | ✅      |
| Fine-Tuning            | Fine-tuned transformer for contract clauses         | ✅      |
| 41-Way Classification  | Classified CUAD clause categories                   | ✅      |
| Model Evaluation       | Evaluated classification performance                | ✅      |
| Confidence Calibration | Added prediction calibration                        | ✅      |
| Risk-Oriented Clauses  | Added important risk clause detection               | ✅      |
| Inference Pipeline     | Integrated classification into application pipeline | ✅      |

### Target Clause Categories

The system supports classification across the **41 CUAD contract clause types**, including important clauses such as:

* Governing Law
* Limitation of Liability
* Termination
* Renewal
* Expiration
* Confidentiality
* Indemnification
* Intellectual Property
* Non-Compete
* Assignment
* Insurance
* Payment Terms
* Dispute Resolution

---

# 🔎 Phase 3 — Semantic Search & Backend

| Task                 | Work Completed                                     | Status |
| -------------------- | -------------------------------------------------- | ------ |
| Embedding Generation | Generated semantic embeddings for contract content | ✅      |
| Vector Database      | Integrated vector storage                          | ✅      |
| Semantic Search      | Implemented similarity-based retrieval             | ✅      |
| Contract Search      | Search across uploaded contracts                   | ✅      |
| Clause Search        | Retrieve similar clauses                           | ✅      |
| FastAPI              | Developed REST backend                             | ✅      |
| API Schemas          | Added request/response validation                  | ✅      |
| Celery               | Added background processing                        | ✅      |
| Redis                | Added task/message broker                          | ✅      |
| Risk API             | Integrated risk scoring with API                   | ✅      |
| Backend Integration  | Connected NLP pipeline with API                    | ✅      |

---

# 🖥️ Phase 4 — Deployment & Frontend

| Task               | Work Completed                         | Status |
| ------------------ | -------------------------------------- | ------ |
| Frontend Dashboard | Created contract analysis dashboard    | ✅      |
| Contract Upload    | Added document upload functionality    | ✅      |
| Clause Display     | Displayed detected clauses             | ✅      |
| Risk Highlighting  | Highlighted potentially risky sections | ✅      |
| Risk Score         | Displayed overall contract risk        | ✅      |
| Search Interface   | Added semantic search interface        | ✅      |
| API Integration    | Connected frontend with FastAPI        | ✅      |
| Docker             | Containerized application              | ✅      |
| Docker Compose     | Configured multi-service environment   | ✅      |
| AWS EC2            | Deployed application to cloud server   | ✅      |
| End-to-End Testing | Tested complete workflow               | ✅      |
| Final Integration  | Integrated all four phases             | ✅      |

---

# 🛡️ Risk Scoring

The system analyzes detected clauses and assigns a risk level based on predefined rules and model predictions.

| Risk Level         | Meaning                                                  |
| ------------------ | -------------------------------------------------------- |
| 🟢 **Low Risk**    | Clause has relatively low contractual risk               |
| 🟡 **Medium Risk** | Clause requires attention or review                      |
| 🔴 **High Risk**   | Clause contains potentially significant contractual risk |

### Risk Factors

The system can evaluate factors such as:

* Unlimited liability
* Broad indemnification
* Automatic renewal
* Short termination notice
* Restrictive governing law
* Missing important clauses
* Unusual contractual language
* High-risk obligations
* Ambiguous terms

---

# 🔍 Key Features

| Feature                     | Description                          |
| --------------------------- | ------------------------------------ |
| 📄 Multi-format Documents   | PDF, scanned PDF, DOCX and TXT       |
| 🔤 OCR                      | Extracts text from scanned contracts |
| 🧠 NER                      | Detects important entities           |
| 🤖 AI Clause Classification | Classifies contract clauses          |
| 📊 Risk Scoring             | Calculates contractual risk          |
| 🔎 Semantic Search          | Finds similar clauses/contracts      |
| ⚡ FastAPI                   | REST API backend                     |
| 🔄 Celery                   | Background document processing       |
| 🗄️ Vector Database         | Stores semantic embeddings           |
| 📈 Dashboard                | Interactive contract analysis        |
| 🎯 Clause Highlighting      | Highlights important/risky clauses   |
| 🐳 Docker                   | Containerized deployment             |
| ☁️ AWS                      | Cloud deployment                     |

---

# 🧪 Testing

The project includes unit, integration and end-to-end tests covering:

| Test Area             | Status   |
| --------------------- | -------- |
| PDF Extraction        | ✅ Passed |
| OCR Extraction        | ✅ Passed |
| DOCX Extraction       | ✅ Passed |
| Text Cleaning         | ✅ Passed |
| CUAD Processing       | ✅ Passed |
| NER Inference         | ✅ Passed |
| Clause Classification | ✅ Passed |
| Risk Scoring          | ✅ Passed |
| API Endpoints         | ✅ Passed |
| Vector Search         | ✅ Passed |
| Frontend Integration  | ✅ Passed |
| End-to-End Workflow   | ✅ Passed |

---

# 🛠️ Technology Stack

| Category             | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python 3.11+                |
| NLP                  | spaCy                       |
| Dataset              | CUAD                        |
| Legal NLP            | InLegalBERT / Legal RoBERTa |
| NER                  | spaCy NER                   |
| Backend              | FastAPI                     |
| Task Queue           | Celery                      |
| Message Broker       | Redis                       |
| Vector Search        | Pinecone / Milvus           |
| OCR                  | Tesseract                   |
| PDF Processing       | pdfminer / Poppler          |
| Document Processing  | python-docx                 |
| Testing              | Pytest                      |
| Containerization     | Docker                      |
| Cloud                | AWS EC2                     |
| Frontend             | Web-based dashboard         |
| Version Control      | Git & GitHub                |

---

# 📦 Installation

```bash
# Clone repository
git clone <repo>

# Enter project directory
cd contract-intelligence

# Create virtual environment
python -m venv .venv

# Activate environment

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env

# Install spaCy model
python -m spacy download en_core_web_lg
```

---

# ▶️ Running the Application

### Start Backend

```bash
uvicorn api.main:app --reload
```

### Start Celery Worker

```bash
celery -A workers.celery_app worker --loglevel=info
```

### Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### Run with Docker

```bash
docker-compose up --build
```

---

# 🧪 Run Tests

```bash
pytest tests/ -v
```

For coverage:

```bash
pytest tests/ --cov=core --cov=ingestion --cov=data_processing --cov=ner
```

---

# 🔐 Environment Configuration

Create a `.env` file using `.env.example`.

Example configuration:

```env
GPU_ID=-1

MODEL_NAME=law-ai/InLegalBERT

VECTOR_DB=pinecone

REDIS_URL=redis://localhost:6379

API_HOST=0.0.0.0
API_PORT=8000
```

> Do not commit API keys, passwords, model credentials, or other secrets to GitHub.

---

# 📈 Project Achievements

| Area             | Achievement                      |
| ---------------- | -------------------------------- |
| Dataset          | 510 CUAD contracts processed     |
| Clause Types     | 41 CUAD categories               |
| NER Labels       | 7 labels                         |
| Document Formats | PDF, OCR PDF, DOCX, TXT          |
| NLP              | NER + Transformer Classification |
| Search           | Semantic Vector Search           |
| Risk             | Automated Risk Scoring           |
| Backend          | FastAPI + Celery                 |
| Deployment       | Docker + AWS EC2                 |
| Frontend         | Interactive Analysis Dashboard   |
| Testing          | Unit + Integration + End-to-End  |
| Development      | 4 Phases Completed               |

---

# 🎯 Project Objectives

1. Automate contract document processing.
2. Extract important contractual information.
3. Identify and classify contract clauses.
4. Detect potentially risky contractual terms.
5. Provide automated risk scoring.
6. Enable semantic contract and clause search.
7. Reduce manual contract review effort.
8. Provide an easy-to-use contract analysis dashboard.
9. Support scalable API-based processing.
10. Deploy the complete system in a production-ready environment.

---

# 🌟 Advantages

* Reduces manual contract review time.
* Automates repetitive legal document analysis.
* Identifies important clauses quickly.
* Provides centralized risk information.
* Supports scanned and digital contracts.
* Enables semantic search.
* Provides visual risk and clause highlighting.
* Supports asynchronous processing for large documents.
* Scalable cloud-based architecture.
* Modular design allows future improvements.

---

# 🔮 Future Scope

| Future Enhancement       | Description                                         |
| ------------------------ | --------------------------------------------------- |
| Advanced LLM Integration | Use LLMs for contract summarization and explanation |
| Multi-language Support   | Analyze contracts in multiple languages             |
| Explainable AI           | Explain why a clause received a specific risk score |
| Contract Comparison      | Compare two or more contracts automatically         |
| Negotiation Assistant    | Suggest safer alternative clauses                   |
| Email Integration        | Analyze contracts received through email            |
| Advanced Analytics       | Organization-level contract risk dashboards         |
| Continuous Learning      | Improve models using reviewed contracts             |
| Mobile Application       | Provide contract analysis through mobile devices    |

---

# 👥 Development Phases

| Phase       | Focus            | Major Deliverables                                    | Status      |
| ----------- | ---------------- | ----------------------------------------------------- | ----------- |
| **Phase 1** | Data & Baseline  | CUAD, ingestion, OCR, NER, testing                    | ✅ Completed |
| **Phase 2** | AI Models        | Legal transformer, clause classification, calibration | ✅ Completed |
| **Phase 3** | Backend & Search | Vector search, FastAPI, Celery, risk API              | ✅ Completed |
| **Phase 4** | Deployment & UI  | Frontend, Docker, AWS, highlighting, integration      | ✅ Completed |

---

# 🏆 Final Project Status

```text
Phase 1  ████████████████████  100%
Phase 2  ████████████████████  100%
Phase 3  ████████████████████  100%
Phase 4  ████████████████████  100%

Overall Project Completion: 100% ✅
```

> **AI-Powered Contract Intelligence & Risk Scoring — Complete End-to-End System**

The project successfully integrates **AI/NLP, document processing, clause classification, risk analysis, semantic search, REST APIs, asynchronous processing, frontend visualization, containerization, and cloud deployment** into a unified contract intelligence platform.
