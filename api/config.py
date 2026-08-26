"""
api/config.py
==============
Phase 3 API settings. Deliberately lightweight (SQLite + local FAISS +
FastAPI BackgroundTasks) instead of the full Postgres/Redis/Celery/Pinecone/S3
stack described in phase_03_tasks.md — chosen so the whole pipeline runs on a
single local machine with no external services or paid accounts required.

Swap these for the "real" infra in Phase 4 (AWS EC2 deployment) if needed.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Storage ──────────────────────────────────────────────────────────────
UPLOAD_DIR   = ROOT / "data" / "uploads"
FAISS_DIR    = ROOT / "data" / "faiss_index"
DB_PATH      = ROOT / "data" / "contracts.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH.as_posix()}"

# ── Models ───────────────────────────────────────────────────────────────
NER_MODEL_PATH        = ROOT / "models" / "ner_baseline" / "model-best"
CLASSIFIER_MODEL_PATH = ROOT / "models" / "clause_classifier"
CLAUSE_LABELS_PATH    = ROOT / "classification" / "config" / "clause_labels.json"

# Small embedding model so this runs at reasonable speed on CPU.
# (Spec calls for BAAI/bge-large-en-v1.5 — swap in Phase 4 if a GPU host is
# available; bge-small trades a little retrieval quality for ~5x CPU speed.)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM         = 384

# ── Risk scoring ─────────────────────────────────────────────────────────
# Clause types that increase contract risk when present, with a weight.
# Purely heuristic — documented as such in the risk report.
HIGH_RISK_CLAUSES: dict[str, float] = {
    "UNCAPPED_LIABILITY":            3.0,
    "LIQUIDATED_DAMAGES":            2.0,
    "NON_COMPETE":                   1.5,
    "EXCLUSIVITY":                   1.5,
    "IRREVOCABLE_OR_PERPETUAL_LICENSE": 2.0,
    "MOST_FAVORED_NATION":           1.5,
    "CHANGE_OF_CONTROL":             1.0,
    "MINIMUM_COMMITMENT":            1.0,
    "VOLUME_RESTRICTION":            1.0,
    "PRICE_RESTRICTIONS":            1.0,
    "NON_DISPARAGEMENT":             0.5,
    "COVENANT_NOT_TO_SUE":           1.0,
    "UNLIMITED_ALL_YOU_CAN_EAT_LICENSE": 1.0,
    "JOINT_IP_OWNERSHIP":            1.0,
    "AFFILIATE_LICENSE_LICENSOR":    0.5,
}

# Clauses that reduce risk when present (protective terms).
PROTECTIVE_CLAUSES: dict[str, float] = {
    "CAP_ON_LIABILITY":              -1.5,
    "TERMINATION_FOR_CONVENIENCE":   -1.0,
    "INSURANCE":                     -0.5,
    "AUDIT_RIGHTS":                  -0.5,
}

RISK_THRESHOLDS = {
    "LOW":    3.0,   # score below this  -> LOW
    "MEDIUM": 7.0,   # score below this  -> MEDIUM, else HIGH
}

for _dir in (UPLOAD_DIR, FAISS_DIR, DB_PATH.parent):
    _dir.mkdir(parents=True, exist_ok=True)
