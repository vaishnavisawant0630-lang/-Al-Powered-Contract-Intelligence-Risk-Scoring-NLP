# Dockerfile
# ===========
# Base image for the Contract Intelligence platform.
#
# WHAT IS INSTALLED
# -----------------
# - Python 3.11 (slim base)
# - Tesseract OCR 5.x (from apt)
# - poppler-utils (pdf2image dependency — pdftoppm binary)
# - All Python packages from requirements.txt
#
# NOTE: en_core_web_lg is NOT installed here (too large for base image).
# It is volume-mounted or downloaded at container startup.
#
# MULTI-STAGE BUILD (Phase 3)
# ----------------------------
# Phase 3 will add a multi-stage build:
#   Stage 1 (builder): install all deps including dev tools
#   Stage 2 (runtime): copy only necessary artifacts → smaller image

FROM python:3.11-slim

LABEL maintainer="contract-intelligence-team"
LABEL description="AI Contract Intelligence Platform — Phase 1"

# ------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Tesseract OCR engine (v5.x in Debian bookworm)
    tesseract-ocr \
    tesseract-ocr-eng \
    # poppler-utils: provides pdftoppm for pdf2image
    poppler-utils \
    # libgomp1: required by some spaCy/numpy builds
    libgomp1 \
    # curl for downloading models/data in scripts
    curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Python environment
# ------------------------------------------------------------------
WORKDIR /app

# Install Python deps before copying code (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install spaCy base model
# (en_core_web_lg is large — download only if not mounted as volume)
# TODO: uncomment when image size is acceptable
# RUN python -m spacy download en_core_web_lg

# ------------------------------------------------------------------
# Application code
# ------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------
# Default command: show help (override in docker-compose)
CMD ["python", "--version"]
