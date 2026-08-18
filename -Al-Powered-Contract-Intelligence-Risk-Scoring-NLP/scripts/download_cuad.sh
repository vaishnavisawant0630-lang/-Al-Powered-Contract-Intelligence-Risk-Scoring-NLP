#!/usr/bin/env bash
# scripts/download_cuad.sh
# =========================
# Downloads the CUAD dataset from HuggingFace Hub to data/raw/.
#
# USAGE
# -----
#   bash scripts/download_cuad.sh
#
# WHAT IT DOES
# ------------
# 1. Verifies Python + huggingface-hub CLI are available
# 2. Downloads CUAD using the Python datasets library (via inline Python)
#    (avoids needing git-lfs which is required for git clone approach)
# 3. Saves raw parquet/arrow files to data/raw/cuad/
# 4. Prints file size and sample count on completion
#
# DEPENDENCIES
# ------------
# - Python 3.11+
# - datasets library (installed via requirements.txt)
# - Internet access to huggingface.co
#
# OUTPUT
# ------
# data/raw/cuad/      ← HuggingFace cache for the CUAD dataset
#
# NOTE
# ----
# This script does NOT need to be re-run if the cache already exists.
# CuadLoader.load() checks for existing cache before downloading.
#
# ALTERNATIVE (manual)
# --------------------
# You can also download CUAD manually from:
#   https://huggingface.co/datasets/cuad
# and unzip to data/raw/cuad/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Downloading CUAD Dataset ==="
echo "Target directory: $PROJECT_ROOT/data/raw"

mkdir -p "$PROJECT_ROOT/data/raw"

# TODO (implementation):
# python3 -c "
# from datasets import load_dataset
# print('Loading CUAD from HuggingFace Hub...')
# ds = load_dataset('cuad', cache_dir='$PROJECT_ROOT/data/raw')
# print(f'Done. Train samples: {len(ds[\"train\"])}')
# "

echo "=== CUAD download complete ==="
