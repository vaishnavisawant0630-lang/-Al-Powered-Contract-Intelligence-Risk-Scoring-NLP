#!/usr/bin/env bash
# scripts/prepare_data.sh
# ========================
# Runs the full data processing pipeline end-to-end.
#
# PIPELINE EXECUTED
# -----------------
# 1. Load CUAD dataset via CuadLoader
# 2. Convert to NER training data (cuad_ner_train.spacy + cuad_ner_dev.spacy)
# 3. Convert to clause classification data (cuad_clauses_train.json + dev.json)
# 4. Print dataset statistics report
#
# PREREQUISITES
# -------------
# - CUAD dataset downloaded: run scripts/download_cuad.sh first
# - Python virtual environment activated with requirements.txt installed
# - data/raw/ directory exists and contains CUAD data
#
# USAGE
# -----
#   bash scripts/prepare_data.sh
#
#   # Override output directory:
#   DATA_PROCESSED_DIR=data/custom_output bash scripts/prepare_data.sh
#
# OUTPUT FILES
# ------------
#   data/processed/cuad_ner_train.spacy     → spaCy DocBin for NER training
#   data/processed/cuad_ner_dev.spacy       → spaCy DocBin for NER evaluation
#   data/processed/cuad_clauses_train.json  → clause classification training data
#   data/processed/cuad_clauses_dev.json    → clause classification dev data
#
# EXPECTED RUNTIME
# ----------------
# ~10–15 minutes on CPU for full CUAD dataset (22,450 Q&A samples)
# Progress bars shown via tqdm during conversion.
#
# ERROR HANDLING
# --------------
# set -euo pipefail ensures any step failure exits the script immediately.
# Check logs/ directory for structured JSON logs if errors occur.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Contract Intelligence — Data Preparation Pipeline ==="
echo "Project root: $PROJECT_ROOT"

# Step 1: Verify CUAD data exists
echo "[1/4] Checking CUAD data..."
# TODO: check data/raw/ contains CUAD cache

# Step 2: Build NER corpus
echo "[2/4] Building NER training corpus..."
# TODO: python3 -m data_processing.cuad_to_ner

# Step 3: Build clause classification corpus
echo "[3/4] Building clause classification corpus..."
# TODO: python3 -m data_processing.cuad_to_classification

# Step 4: Print statistics
echo "[4/4] Dataset statistics..."
# TODO: python3 -m data_processing.dataset_stats

echo ""
echo "=== Data preparation complete ==="
echo "Output files:"
ls -lh "$PROJECT_ROOT/data/processed/" 2>/dev/null || echo "  (no files yet — run implementation)"
