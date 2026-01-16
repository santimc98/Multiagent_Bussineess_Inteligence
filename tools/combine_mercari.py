import pandas as pd

train = pd.read_csv("data/train.tsv", sep="\t")
test  = pd.read_csv("data/test_stg2.tsv",  sep="\t")

# Unificar id
train = train.rename(columns={"train_id": "id"})
test  = test.rename(columns={"test_id": "id"})

# Split
train["__split"] = "train"
test["__split"]  = "test"

# Target en test
if "price" not in test.columns:
    test["price"] = pd.NA

# Alinear columnas
all_cols = list(dict.fromkeys(list(train.columns) + list(test.columns)))
for c in all_cols:
    if c not in train.columns:
        train[c] = pd.NA
    if c not in test.columns:
        test[c] = pd.NA

combined = pd.concat([train[all_cols], test[all_cols]], ignore_index=True)
combined.to_csv("data/combined_mercari.csv", index=False)

print(f"OK -> data/combined_mercari.csv | rows={len(combined)} | cols={len(combined.columns)}")
