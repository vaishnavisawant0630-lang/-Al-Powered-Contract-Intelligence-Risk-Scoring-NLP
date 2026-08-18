"""
core/config.py
==============
Centralised, validated application configuration.

PURPOSE
-------
All environment variables and tuneable parameters are declared ONCE here
using Pydantic Settings. Every other module calls get_settings() to read
config — no module reads os.environ directly.

HOW IT WORKS
------------
1. Pydantic reads values from environment variables (case-insensitive).
2. If a .env file exists at the project root, python-dotenv loads it first.
3. All fields have type annotations → Pydantic validates types on startup.
4. The Settings object is cached via @lru_cache so it is instantiated once
   per process, not once per import.

USAGE
-----
    from core.config import get_settings

    settings = get_settings()
    print(settings.data_raw_dir)      # Path object, guaranteed to be a string
    print(settings.ocr_dpi)           # int, validated ≥ 72

ADDING NEW CONFIG
-----------------
1. Add a field with type annotation and default value below.
2. Add the matching line to .env.example.
3. Run `python -c "from core.config import get_settings; get_settings()"` to
   validate the new field before any other code runs.

FIELDS DEFINED (grouped by subsystem)
--------------------------------------
    Paths:
        data_raw_dir            Where the unzipped CUAD dataset lives
        data_processed_dir      Where .spacy + .json training files are written
        models_dir              Where spaCy saved models land

    OCR:
        tesseract_cmd           Path to tesseract binary (default: auto-detect)
        ocr_dpi                 Render resolution for pdf2image (default: 300)
        ocr_lang                Tesseract language code (default: eng)
        ocr_char_density_threshold  chars/page below which auto-OCR triggers (default: 50)

    spaCy / NER:
        spacy_base_model        Base model to initialise from (default: en_core_web_lg)
        ner_train_epochs        Training epochs (default: 20)
        ner_batch_size          Mini-batch size (default: 32)
        gpu_id                  -1 = CPU, 0+ = GPU device index (default: -1)

    Data processing:
        cuad_train_split        Fraction of CUAD used for training (default: 0.85)
        cuad_random_seed        Reproducible split seed (default: 42)
        max_text_length         Characters per NERSample chunk (default: 512)

    Logging:
        log_level               DEBUG | INFO | WARNING | ERROR (default: INFO)
        log_format              json | console (default: json in prod, console in dev)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# TODO (implementation): from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings:
    """
    Application-wide configuration loaded from environment / .env file.

    IMPLEMENTATION NOTES (fill in during Phase 1 execution)
    --------------------------------------------------------
    - Inherit from pydantic_settings.BaseSettings
    - Use model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    - All Path fields should use validators to auto-create directories on first use
    - Add @field_validator for ocr_dpi to enforce 72 ≤ dpi ≤ 600
    - Add @field_validator for gpu_id to enforce -1 ≤ id ≤ 8
    """

    # --- Paths ---
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    models_dir: str = "models"

    # --- OCR ---
    tesseract_cmd: str = ""           # empty = auto-detect via shutil.which
    ocr_dpi: int = 300
    ocr_lang: str = "eng"
    ocr_char_density_threshold: int = 50  # chars/page; below this → trigger OCR

    # --- spaCy / NER ---
    spacy_base_model: str = "en_core_web_lg"
    ner_train_epochs: int = 20
    ner_batch_size: int = 32
    gpu_id: int = -1                  # -1 = CPU only (Phase 1 decision)

    # --- Data Processing ---
    cuad_train_split: float = 0.85
    cuad_random_seed: int = 42
    max_text_length: int = 512        # chunk size for NER sample splitting

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "json"          # json = structured prod logs; console = human-readable


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached so environment is read exactly once per process lifetime.
    In tests, call get_settings.cache_clear() before patching env vars.

    Returns
    -------
    Settings
        Fully validated application configuration object.

    Raises
    ------
    pydantic.ValidationError
        If any required env var is missing or has wrong type.
        Process should exit on this error — configuration is mandatory.
    """
    # TODO (implementation): return Settings()
    return Settings()
