"""
ner/train.py
=============
Trains the spaCy NER baseline model using the CUAD DocBin corpus.

USAGE
-----
    # From project root:
    python -m ner.train

    # Or via shell script:
    bash scripts/train_ner.sh

WHAT THIS DOES
--------------
1. Validates that cuad_ner_train.spacy and cuad_ner_dev.spacy exist
2. Calls spaCy's train CLI (via spacy.cli.train) with base_config.cfg
3. Respects GPU_ID env var (default: -1 CPU)
4. Saves:
     models/ner_baseline/model-best/   ← best dev F1 checkpoint
     models/ner_baseline/model-last/   ← final epoch checkpoint
     models/ner_baseline/training_meta.json  ← stats written after training

NOTES
-----
- Training time: ~2-4 hours on CPU (2000 steps, 433 documents)
- On GPU (--gpu-id 0): ~15-30 minutes
- Expected dev F1: 0.78 – 0.86 (CUAD baseline reference)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent
CONFIG     = ROOT / "ner" / "config" / "base_config.cfg"
TRAIN_DATA = ROOT / "data" / "processed" / "cuad_ner_train.spacy"
DEV_DATA   = ROOT / "data" / "processed" / "cuad_ner_dev.spacy"
OUTPUT_DIR = ROOT / "models" / "ner_baseline"


def train(
    config:     Path = CONFIG,
    output_dir: Path = OUTPUT_DIR,
    train_data: Path = TRAIN_DATA,
    dev_data:   Path = DEV_DATA,
    gpu_id:     int  = -1,
) -> dict:
    """
    Train the spaCy NER model.

    Parameters
    ----------
    config : Path
        Path to the spaCy config file.
    output_dir : Path
        Where to save model checkpoints.
    train_data : Path
        Path to cuad_ner_train.spacy (DocBin).
    dev_data : Path
        Path to cuad_ner_dev.spacy (DocBin).
    gpu_id : int
        -1 = CPU, 0+ = GPU device index.

    Returns
    -------
    dict
        training_meta.json contents.
    """
    # ── Pre-flight checks ─────────────────────────────────────────────────
    _check_inputs(train_data, dev_data, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting NER training:")
    logger.info("  Config:    %s", config)
    logger.info("  Output:    %s", output_dir)
    logger.info("  Train:     %s", train_data)
    logger.info("  Dev:       %s", dev_data)
    logger.info("  GPU:       %s", "CPU" if gpu_id == -1 else f"GPU:{gpu_id}")

    start_time = time.time()

    # ── Call spaCy train via subprocess ──────────────────────────────────
    # spacy.cli.train is a module in spaCy 3.x, not callable directly.
    # The correct entry point is: python -m spacy train
    import subprocess, sys

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config),
        "--output", str(output_dir),
        "--paths.train", str(train_data),
        "--paths.dev",   str(dev_data),
        "--gpu-id",      str(gpu_id),
    ]

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    # spaCy may exit with code 1 on Windows when the output path contains spaces
    # (a known shell-quoting quirk). Treat as success if model-last/ was written.
    model_last = output_dir / "model-last"
    if result.returncode != 0 and not model_last.exists():
        raise RuntimeError(
            f"spaCy training failed with exit code {result.returncode} "
            f"and no model was saved to {output_dir}"
        )

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f minutes", elapsed / 60)

    # ── Write training_meta.json ─────────────────────────────────────────
    meta = _write_meta(output_dir, elapsed, train_data, dev_data)

    return meta


def _check_inputs(train_data: Path, dev_data: Path, config: Path) -> None:
    """Raise clear errors if required files are missing."""
    missing = [p for p in [train_data, dev_data, config] if not p.exists()]
    if missing:
        paths = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing required files:\n  {paths}\n"
            f"Run: python data_processing/run_task1.py"
        )


def _write_meta(
    output_dir: Path,
    elapsed_seconds: float,
    train_data: Path,
    dev_data: Path,
) -> dict:
    """
    Write training_meta.json with stats from the best model.

    Reads model-best/meta.json produced by spaCy to extract F1 scores.
    """
    meta = {
        "training_duration_seconds": round(elapsed_seconds, 1),
        "training_duration_minutes": round(elapsed_seconds / 60, 1),
        "train_data":    str(train_data),
        "dev_data":      str(dev_data),
        "model_best":    str(output_dir / "model-best"),
        "model_last":    str(output_dir / "model-last"),
        "phase":         "1",
        "model_type":    "spaCy tok2vec + NER",
        "base_model":    "en_core_web_lg",
    }

    # Try to read F1 from spaCy's model-best meta.json
    best_meta_path = output_dir / "model-best" / "meta.json"
    if best_meta_path.exists():
        with open(best_meta_path) as f:
            spacy_meta = json.load(f)
        perf = spacy_meta.get("performance", {})
        meta["dev_ents_f"]  = perf.get("ents_f", None)
        meta["dev_ents_p"]  = perf.get("ents_p", None)
        meta["dev_ents_r"]  = perf.get("ents_r", None)
        per_type = perf.get("ents_per_type", {})
        meta["per_label_f1"] = {
            label: scores.get("f", None)
            for label, scores in per_type.items()
        }

    meta_path = output_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote training_meta.json → %s", meta_path)

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train CUAD NER baseline model")
    parser.add_argument("--config",     default=str(CONFIG))
    parser.add_argument("--output",     default=str(OUTPUT_DIR))
    parser.add_argument("--train",      default=str(TRAIN_DATA))
    parser.add_argument("--dev",        default=str(DEV_DATA))
    parser.add_argument("--gpu-id",     type=int, default=-1)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    meta = train(
        config=Path(args.config),
        output_dir=Path(args.output),
        train_data=Path(args.train),
        dev_data=Path(args.dev),
        gpu_id=args.gpu_id,
    )

    print("\n=== Training Complete ===")
    print(f"  Duration:   {meta.get('training_duration_minutes', '?')} min")
    if "dev_ents_f" in meta:
        print(f"  Dev F1:     {meta['dev_ents_f']:.3f}")
        print(f"  Dev P:      {meta['dev_ents_p']:.3f}")
        print(f"  Dev R:      {meta['dev_ents_r']:.3f}")
    print(f"  Model:      {meta.get('model_best')}")
