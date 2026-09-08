"""
data_processing/run_task1.py
=============================
Runs Task 1 end-to-end:
  Step 1: Load CUAD via CuadLoader (local JSON first)
  Step 2: Convert to classification JSON Lines  → cuad_clauses_train/dev.json
  Step 3: Convert to spaCy DocBin               → cuad_ner_train/dev.spacy
  Step 4: Print DatasetStats report

Run from project root:
    python data_processing/run_task1.py
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_task1")

ROOT          = Path(__file__).parent.parent
OUTPUT_DIR    = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Make project root importable
sys.path.insert(0, str(ROOT))


def main() -> None:
    # ── Step 1: Load CUAD ────────────────────────────────────────────────
    logger.info("STEP 1 — Loading CUAD dataset")
    from data_processing.cuad_loader import CuadLoader

    # Classification uses 80/20 split (spec §1.4)
    cls_loader  = CuadLoader(train_ratio=0.80)
    train_cls, dev_cls = cls_loader.load()
    logger.info(f"Classification split — train={len(train_cls):,}  dev={len(dev_cls):,}")

    # NER uses 85/15 split
    ner_loader  = CuadLoader(train_ratio=0.85)
    train_ner, dev_ner = ner_loader.load()
    logger.info(f"NER split — train={len(train_ner):,}  dev={len(dev_ner):,}")

    # ── Step 2: Classification JSON Lines ────────────────────────────────
    logger.info("STEP 2 — Converting to clause classification JSON Lines")
    from data_processing.cuad_to_classification import CuadToClassification
    cls_stats = CuadToClassification().convert(train_cls, dev_cls, OUTPUT_DIR)

    # ── Step 3: spaCy DocBin (NER) ───────────────────────────────────────
    logger.info("STEP 3 — Converting to spaCy DocBin (NER)")
    from data_processing.cuad_to_ner import CuadToNer
    ner_stats = CuadToNer().convert(train_ner, dev_ner, OUTPUT_DIR)

    # ── Step 4: Stats report ─────────────────────────────────────────────
    logger.info("STEP 4 — Generating dataset statistics report")
    from data_processing.dataset_stats import DatasetStats
    DatasetStats.print_report(processed_dir=OUTPUT_DIR)

    # ── Final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  TASK 1 COMPLETE — All output files generated")
    print("=" * 65)
    print(f"  cuad_clauses_train.json   {cls_stats['train_records']:>8,} records")
    print(f"  cuad_clauses_dev.json     {cls_stats['dev_records']:>8,} records")
    print(f"  cuad_ner_train.spacy      {ner_stats['train_docs']:>8,} docs  "
          f"{ner_stats['train_entities']:,} entities")
    print(f"  cuad_ner_dev.spacy        {ner_stats['dev_docs']:>8,} docs  "
          f"{ner_stats['dev_entities']:,} entities")
    print(f"  Span conflicts resolved:  {ner_stats['total_conflicts']:>8,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
