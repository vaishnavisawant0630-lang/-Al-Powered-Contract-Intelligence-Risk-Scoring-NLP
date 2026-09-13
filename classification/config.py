"""Training configuration for the clause classifier (Phase 2)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    # Model — fallback chain tried in order until one loads successfully.
    model_name: str = "law-ai/InLegalBERT"
    fallback_models: list[str] = field(default_factory=lambda: [
        "lexlms/legal-roberta-large",
        "roberta-base",
    ])
    num_labels: int = 41
    max_length: int = 512

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch = 32
    learning_rate: float = 2e-5
    num_train_epochs: int = 5
    warmup_ratio: float = 0.06
    fp16: bool = True  # auto-disabled if no CUDA in trainer.py

    # Paths
    train_path: str = "data/processed/cuad_clauses_train.json"
    dev_path: str = "data/processed/cuad_clauses_dev.json"
    output_dir: str = "models/clause_classifier"
    labels_path: str = "classification/config/clause_labels.json"

    def load_labels(self) -> list[str]:
        with open(self.labels_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def ensure_output_dir(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
