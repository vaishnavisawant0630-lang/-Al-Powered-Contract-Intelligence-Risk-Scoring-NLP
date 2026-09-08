"""Fine-tune a legal-domain transformer for 41-way multi-label clause classification."""
from __future__ import annotations

import json
import logging

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import TrainingConfig
from .dataset_builder import build_datasets

logger = logging.getLogger(__name__)

PRIORITY_CLAUSES = [
    "TERMINATION_FOR_CONVENIENCE",
    "GOVERNING_LAW",
    "RENEWAL_TERM",
    "CAP_ON_LIABILITY",
    # NOTE: CUAD has no distinct "Indemnification" label; UNCAPPED_LIABILITY
    # is the closest cost-exposure proxy available in this label set.
    "UNCAPPED_LIABILITY",
]


class WeightedTrainer(Trainer):
    """HuggingFace Trainer subclass using BCEWithLogitsLoss with per-label pos_weight."""

    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device)
        )
        loss = loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def load_model_with_fallback(config: TrainingConfig):
    """Try model_name first, then each fallback in order. Returns (tokenizer, model, name_used)."""
    candidates = [config.model_name, *config.fallback_models]
    last_err = None
    for name in candidates:
        try:
            logger.info("Attempting to load model: %s", name)
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(
                name,
                num_labels=config.num_labels,
                problem_type="multi_label_classification",
            )
            logger.info("Loaded model: %s", name)
            return tokenizer, model, name
        except Exception as e:  # noqa: BLE001 — deliberately broad, this is a fallback chain
            logger.warning("Failed to load %s: %s", name, e)
            last_err = e
    raise RuntimeError(
        f"All model candidates failed to load. Last error: {last_err}"
    )


def macro_micro_f1(preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    probs = 1 / (1 + np.exp(-preds))  # sigmoid
    pred_bin = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    tp = (pred_bin & labels).sum(axis=0)
    fp = (pred_bin & (1 - labels)).sum(axis=0)
    fn = ((1 - pred_bin) & labels).sum(axis=0)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) > 0,
    )

    macro_f1 = float(f1.mean())

    tp_sum, fp_sum, fn_sum = tp.sum(), fp.sum(), fn.sum()
    micro_p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    micro_r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    return {"macro_f1": macro_f1, "micro_f1": float(micro_f1), "per_label_f1": f1}


def make_compute_metrics(labels: list[str]):
    def compute_metrics(eval_pred):
        preds, y_true = eval_pred
        result = macro_micro_f1(preds, y_true)
        out = {"macro_f1": result["macro_f1"], "micro_f1": result["micro_f1"]}
        for name in PRIORITY_CLAUSES:
            if name in labels:
                idx = labels.index(name)
                out[f"f1_{name}"] = float(result["per_label_f1"][idx])
        return out

    return compute_metrics


def train(config: TrainingConfig | None = None):
    logging.basicConfig(level=logging.INFO)
    config = config or TrainingConfig()
    config.ensure_output_dir()

    labels = config.load_labels()
    tokenizer, model, model_used = load_model_with_fallback(config)

    train_ds, dev_ds, pos_weight = build_datasets(config, tokenizer)

    has_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="macro_f1",
        load_best_model_at_end=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16 and has_cuda,
        dataloader_num_workers=2,
        logging_steps=50,
        report_to=[],  # wandb optional — kept off by default so CI never fails on it
    )

    trainer = WeightedTrainer(
        pos_weight=pos_weight,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=make_compute_metrics(labels),
    )

    logger.info("Starting training with model=%s", model_used)
    trainer.train()

    logger.info("Saving final model to %s", config.output_dir)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = trainer.evaluate()
    meta = {
        "model_used": model_used,
        "num_labels": config.num_labels,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "final_eval_metrics": metrics,
    }
    with open(f"{config.output_dir}/training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Training complete. macro_f1=%.4f micro_f1=%.4f",
                metrics.get("eval_macro_f1", -1), metrics.get("eval_micro_f1", -1))
    return trainer, metrics


if __name__ == "__main__":
    train()
