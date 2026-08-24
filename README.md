# MODEL TRANING & FINE TUNING - MEMBER 5
# Fine-Tuning – Clause Classification

## Overview

This folder contains the fine-tuned Transformer model used for **legal contract clause classification** in the AI-Powered Contract Intelligence & Risk Scoring project.

The model is trained on the processed CUAD-based clause dataset and is designed to classify extracted contract clauses into predefined legal clause categories.

The fine-tuning pipeline uses:

* Python
* Hugging Face Transformers
* PyTorch
* RoBERTa
* Hugging Face Datasets
* Scikit-learn
* Pandas
* NumPy

---

## Folder Structure

```text
Fine-Tuning/
│
├── train.py
├── evaluate_metrics.py
├── README.md
│
└── clause_classifier/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    ├── vocab.json
    └── merges.txt
```

> **Note:** The trained `model.safetensors` file may be stored separately because of GitHub file-size/storage considerations.

---

# 1. Model Training

## Purpose

The training pipeline fine-tunes a pretrained Transformer model for legal clause classification.

The model learns to map contract clause text to the appropriate clause category.

### Training Flow

```text
CUAD / Processed Dataset
          ↓
Data Cleaning
          ↓
Train / Validation Split
          ↓
Tokenization
          ↓
RoBERTa Fine-Tuning
          ↓
Validation
          ↓
Trained Clause Classifier
```

## Dataset

The project uses a processed clause dataset derived from the **CUAD (Contract Understanding Atticus Dataset)**.

Expected dataset location:

```text
data/processed/clause_dataset.csv
```

The dataset should contain at least:

```text
text
label_id
```

where:

* `text` = contract clause text
* `label_id` = numerical clause category

---

## Model

The baseline Transformer model used for fine-tuning is:

```text
roberta-base
```

The model is trained as a sequence-classification model.

### Example Configuration

```text
Maximum sequence length : 512
Batch size              : 8
Epochs                  : 2
Learning rate           : 2e-5
```

These values can be modified in the training script depending on available hardware and dataset size.

---

# 2. Training the Model

Run the training script from the project root:

```bash
python train.py
```

The training pipeline performs the following operations:

1. Loads the processed clause dataset.
2. Detects the text and label columns.
3. Splits the dataset into training and validation sets.
4. Converts the data into a Hugging Face Dataset.
5. Tokenizes clause text using the RoBERTa tokenizer.
6. Fine-tunes the Transformer model.
7. Evaluates the model on the validation dataset.
8. Saves the trained model and tokenizer.

The trained model is saved to:

```text
models/clause_classifier/
```

or the output directory configured in the training script.

---

# 3. Model Files

A trained Transformer model normally contains files such as:

```text
clause_classifier/
│
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.json
└── merges.txt
```

### Important Files

| File                      | Purpose                                            |
| ------------------------- | -------------------------------------------------- |
| `model.safetensors`       | Contains the trained model weights                 |
| `config.json`             | Model architecture and configuration               |
| `tokenizer.json`          | Tokenizer configuration and vocabulary information |
| `tokenizer_config.json`   | Tokenizer settings                                 |
| `special_tokens_map.json` | Special-token configuration                        |
| `vocab.json`              | RoBERTa vocabulary                                 |
| `merges.txt`              | RoBERTa BPE merge rules                            |

> The tokenizer files should be kept together with the model configuration. The trained `model.safetensors` file is required to run the trained classifier.

---

# 4. Model Evaluation

## Purpose

The evaluation pipeline measures the performance of the trained clause classifier on validation/test data.

Run:

```bash
python evaluate_metrics.py
```

The evaluation script loads:

```text
data/processed/clause_dataset.csv
```

and the trained model from:

```text
models/clause_classifier/
```

---

## Evaluation Metrics

The following metrics are calculated:

### Accuracy

Measures the percentage of correctly classified clauses.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted instances of a class are actually correct.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual instances of a class were correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1-Score

The F1-score combines precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

# 5. Classification Report

The evaluation pipeline generates a classification report containing:

```text
precision
recall
f1-score
support
```

for each clause category.

Example:

```text
              precision    recall    f1-score    support

Class 0          0.XX       0.XX       0.XX        XXX
Class 1          0.XX       0.XX       0.XX        XXX
Class 2          0.XX       0.XX       0.XX        XXX

accuracy                              0.XX        XXX
macro avg         0.XX       0.XX       0.XX        XXX
weighted avg      0.XX       0.XX       0.XX        XXX
```

The actual values depend on the trained model and evaluation dataset.

---

# 6. Evaluation Output

Evaluation results are stored in the `metrics` directory.

```text
metrics/
├── metrics.json
└── classification_report.txt
```

### `metrics.json`

Stores numerical evaluation results in JSON format.

Example:

```json
{
    "accuracy": 0.00,
    "precision": 0.00,
    "recall": 0.00,
    "f1_score": 0.00
}
```

### `classification_report.txt`

Contains the detailed classification report for each clause category.

---

# 7. Running the Complete Fine-Tuning Pipeline

From the project root:

### Step 1 – Train

```bash
python train.py
```

### Step 2 – Evaluate

```bash
python evaluate_metrics.py
```

### Step 3 – Check Results

```text
metrics/metrics.json
metrics/classification_report.txt
```

---

# 8. Requirements

Install the required dependencies:

```bash
pip install torch
pip install transformers
pip install datasets
pip install pandas
pip install numpy
pip install scikit-learn
```

Or install the project's requirements file:

```bash
pip install -r requirements.txt
```

---

# 9. Using the Trained Model

Once the trained model and tokenizer are available:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/clause_classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)
```

A clause can then be tokenized and passed to the classifier for prediction.

```python
text = "The agreement shall remain effective for a period of two years."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

outputs = model(**inputs)

prediction = outputs.logits.argmax(dim=-1).item()

print("Predicted class:", prediction)
```

---

# 10. Important Notes

* The tokenizer used during inference should match the tokenizer used during training.
* The trained `model.safetensors` file is required for actual model inference.
* The model should be loaded from the same model directory structure used during training.
* Keep the label-to-class mapping consistent between training and inference.
* Do not commit API keys, passwords, `.env` files, or other secrets.
* If the trained model is too large for normal GitHub storage, keep the model weights in dedicated model storage or use Git LFS.

---

# 11. Project Role

This fine-tuning component is responsible for:

```text
Contract Text
      ↓
Clause Classification
      ↓
Clause Category
      ↓
Risk Scoring Pipeline
```

The predicted clause category can subsequently be used by the project's **risk scoring and downstream NLP pipeline**.

---

## Summary

The Fine-Tuning module provides the Transformer-based clause classification component of the project.

It supports:

* CUAD-based clause classification
* RoBERTa fine-tuning
* Automated tokenization
* Model evaluation
* Accuracy measurement
* Precision/Recall/F1 evaluation
* Classification reports
* Saved model and tokenizer artifacts

This module forms the **NLP classification layer** of the AI-Powered Contract Intelligence & Risk Scoring system.
