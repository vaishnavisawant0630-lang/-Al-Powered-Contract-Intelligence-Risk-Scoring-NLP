"""
ner/
====
spaCy NER training, evaluation, and inference package.

PURPOSE
-------
1. Train a spaCy NER model on the processed CUAD data (train.py)
2. Evaluate model performance per entity type (evaluate.py)
3. Load the model and run inference on new contract text (inference.py)

PUBLIC API
----------
    from ner.inference import load_model, extract_entities, batch_extract, Entity

    load_model("models/ner_baseline/model-best")
    entities = extract_entities("This Agreement between Acme Corp and Beta Inc.")

INTERNAL MODULES
----------------
    base.py             BaseNERModel Protocol
    train.py            Entry-point: train and save the model
    evaluate.py         Per-entity-type P/R/F1 evaluation table
    inference.py        load_model, extract_entities, batch_extract
    config/
        base_config.cfg spaCy training config (CPU, 2000 steps, en_core_web_lg)
"""

from ner.inference import load_model, extract_entities, batch_extract, Entity

__all__ = [
    "load_model",
    "extract_entities",
    "batch_extract",
    "Entity",
]
