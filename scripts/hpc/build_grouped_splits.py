"""Build grouped splits from silent_signals (informal/Reddit subset only) for HPC."""
import os, json
from datetime import datetime
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold
from utils import sanitize_text_column

SEED = 42
DATA_DIR = "./data"
os.makedirs(f"{DATA_DIR}/splits", exist_ok=True)
os.makedirs(f"{DATA_DIR}/manifests", exist_ok=True)

df_all = load_dataset("SALT-NLP/silent_signals", split="train").to_pandas()
print(f"Loaded {len(df_all):,} total rows")
print(f"Type distribution: {dict(df_all['type'].value_counts())}")

# Filter to informal (Reddit) subset only
df = df_all[df_all["type"] == "Informal"].reset_index(drop=True)
print(f"After filtering to Informal: {len(df):,} rows, {df['dog_whistle_root'].nunique()} roots")

df = sanitize_text_column(df, "content")
print(f"After sanitization: {len(df):,} rows")

X = np.arange(len(df))
y = df["ingroup"].values
groups = df["dog_whistle_root"].values

sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
tv_idx, test_idx = next(iter(sgkf.split(X, y, groups)))
df_tv = df.iloc[tv_idx]
sgkf2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
train_idx, val_idx = next(iter(sgkf2.split(np.arange(len(df_tv)), df_tv["ingroup"].values, df_tv["dog_whistle_root"].values)))

df_train = df_tv.iloc[train_idx].reset_index(drop=True)
df_val = df_tv.iloc[val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

train_roots = set(df_train["dog_whistle_root"])
val_roots = set(df_val["dog_whistle_root"])
test_roots = set(df_test["dog_whistle_root"])
assert train_roots.isdisjoint(val_roots) and train_roots.isdisjoint(test_roots) and val_roots.isdisjoint(test_roots)
print(f"Splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)} | NO root overlap")

for name, sdf in [("train", df_train), ("val", df_val), ("test", df_test)]:
    sdf.to_parquet(f"{DATA_DIR}/splits/{name}.parquet", index=False)

root_to_split = {}
for r in train_roots: root_to_split[r] = "train"
for r in val_roots: root_to_split[r] = "val"
for r in test_roots: root_to_split[r] = "test"

manifest = {
    "seed": SEED, "created_at": datetime.now().isoformat(),
    "dataset": "informal (Reddit) subset only",
    "root_to_split": root_to_split,
    "splits": {name: {"n_rows": len(sdf), "n_roots": sdf["dog_whistle_root"].nunique()}
              for name, sdf in [("train", df_train), ("val", df_val), ("test", df_test)]},
}
with open(f"{DATA_DIR}/manifests/split_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print("Saved splits and manifest.")
