#!/usr/bin/env bash
# scripts/train_ner.sh
# =====================
# Trains the spaCy NER baseline model.
#
# PREREQUISITES
# -------------
# 1. Data prepared: run scripts/prepare_data.sh first
# 2. en_core_web_lg installed: python -m spacy download en_core_web_lg
# 3. Virtual environment active with requirements.txt installed
#
# USAGE
# -----
#   # Default (CPU, 20 epochs, reads from .env):
#   bash scripts/train_ner.sh
#
#   # Override epochs:
#   NER_TRAIN_EPOCHS=10 bash scripts/train_ner.sh
#
# OUTPUT
# ------
#   models/ner_baseline/model-best/     ← best checkpoint (highest dev F1)
#   models/ner_baseline/model-last/     ← last epoch checkpoint
#   models/ner_baseline/training_meta.json
#
# EXPECTED RUNTIME
# ----------------
# CPU (en_core_web_lg): ~2–4 hours for 20 epochs on full CUAD
# For quick iteration: set NER_TRAIN_EPOCHS=3 (produces a rough baseline)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== NER Model Training ==="
echo "Config: $PROJECT_ROOT/ner/config/base_config.cfg"
echo "Output: $PROJECT_ROOT/models/ner_baseline"

mkdir -p "$PROJECT_ROOT/models/ner_baseline"

# TODO (implementation):
# python3 -m ner.train \
#   --config ner/config/base_config.cfg \
#   --output models/ner_baseline

echo "=== Training complete. Run evaluate: ==="
echo "  python -m ner.evaluate --model models/ner_baseline --dev data/processed/cuad_ner_dev.spacy"
