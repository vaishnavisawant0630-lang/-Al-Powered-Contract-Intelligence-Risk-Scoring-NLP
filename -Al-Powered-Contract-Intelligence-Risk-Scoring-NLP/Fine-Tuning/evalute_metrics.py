import os
import numpy as np
import pandas as pd
import json
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# 1 CONFIG

DATA_FILE = "data/processed/clause_dataset.csv"
MODEL_PATH = "models/clause_classifier"
METRICS_DIR = "metrics"
METRICS_FILE = "metrics/metrics.json"          # fixed typo: mertics -> metrics
REPORT_FILE = "metrics/classification_report.txt"

# 2. LOAD DATA

print("=" * 60)
print("LOADING DATA FOR EVALUATION")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

# 3. SAME VALIDATE SPLIT

_, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label_id"])

# 4. DATASET

val_dataset = Dataset.from_pandas(val_df[["text", "label_id"]], preserve_index=False)
val_dataset = val_dataset.rename_column("label_id", "labels")

# 5. TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize(batch):
    return tokenizer(                 # <-- FIX: call tokenizer, not tokenize
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

val_dataset = val_dataset.map(
    tokenize,
    batched=True
)

# Ensure the dataset is in torch format so Trainer.predict gets tensors
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# 6. LOAD TRAINED MODEL

print("\nLoading trained Model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# 7. PREDICTIONS

trainer = Trainer(model=model)

result = trainer.predict(val_dataset)

predictions = np.argmax(result.predictions, axis=-1)
actual = result.label_ids

# 8. METRICS FUNCTION

def calculate_metrics(actual, predictions):
    accuracy = accuracy_score(actual, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

# 9. CALCULATE
metrics = calculate_metrics(actual, predictions)

# 10. DISPLAY

print("\n" + "=" * 60)
print("MODEL METRICS")
print("=" * 60)
print("Accuracy:", round(metrics["accuracy"], 4))
print("Precision:", round(metrics["precision"], 4))
print("Recall:", round(metrics["recall"], 4))
print("F1 Score:", round(metrics["f1_score"], 4))

# 11. CLASSIFICATION REPORT

report = classification_report(actual, predictions, zero_division=0)
print("\nClassification Report:")
print(report)

# SAVE METRICS

os.makedirs(METRICS_DIR, exist_ok=True)

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    file.write(report)

print("\n" + "=" * 60)
print("METRICS SAVED")
print("=" * 60)

print(METRICS_FILE)
print(REPORT_FILE)