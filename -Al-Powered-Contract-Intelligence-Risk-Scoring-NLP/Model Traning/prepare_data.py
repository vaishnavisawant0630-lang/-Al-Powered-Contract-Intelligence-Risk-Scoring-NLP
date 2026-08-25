import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. CONFIG

INPUT_FILE = "data/raw/master_clauses.csv"

OUTPUT_DIR = "data/processed"

OUTPUT_FILE = "data/processed/clause_dataset.csv"

# 2. CUAD CLAUSE COLUMNS

CLAUSE_COLUMNS = [
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or PerpetualLicense",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary"
]

# 3. LOAD DATA

print("="*60)
print("LOADING MASTER CLAUSES")
print("="*60)

df = pd.read_csv(INPUT_FILE)
print("Original dataset:", df.shape)

# 4. CREATE CLAUSE DATASET

rows = []

for _, row in df.iterrows():
    for clause in CLAUSE_COLUMNS:
        if clause not in df.columns:
            continue

        value = row[clause]
        if pd.isna(value):
            continue

        value = str(value).strip()

        if value == "":
            continue

        # check corresponding Anser column
        answer_column = clause + "-Answer"
        if answer_column in df.columns:
            answer = row[answer_column]
            if pd.isna(answer):
                continue

            answer = str(answer).strip()
            if answer == "":
                continue

            rows.append({
                "text": value,
                "label": clause
            })

# 5. CREATE DATAFRAME

dataset_df = pd.DataFrame(rows)

print("\nPrepared dataset:")
print(dataset_df.shape)

print("\nSample:")
print(dataset_df.head())

# 6. LABEL ENCODING

label_encoder = LabelEncoder()

dataset_df["label_id"] = label_encoder.fit_transform(
    dataset_df["label"]
)

print(
    "\nNumber of categories:",
    len(label_encoder.classes_)
)

print("\nLabel mapping:")

for i, label in enumerate(label_encoder.classes_):

    print(
        i,
        "=",
        label
    )

# 7. SAVE

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

dataset_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# Save label mapping
mapping_file = (
    OUTPUT_DIR +
    "/label_mapping.csv"
)

mapping_df = pd.DataFrame({
    "label_id": range(
        len(label_encoder.classes_)
    ),
    "label": label_encoder.classes_
})

mapping_df.to_csv(
    mapping_file,
    index=False
)

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)

print(
    "Dataset saved:",
    OUTPUT_FILE
)

print(
    "Label mapping saved:",
    mapping_file
)