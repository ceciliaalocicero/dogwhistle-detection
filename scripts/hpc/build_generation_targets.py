"""Build structured JSON targets for generation task."""
import os, json
import pandas as pd

DATA_DIR = "./data"
os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)

for split_name in ["train", "val", "test"]:
    df = pd.read_parquet(f"{DATA_DIR}/splits/{split_name}.parquet")
    inputs, targets = [], []
    for _, row in df.iterrows():
        dw = str(row["dog_whistle"])
        content = str(row["content"])
        root = str(row["dog_whistle_root"])
        ingroup = str(row["ingroup"])
        defn = str(row["definition"]) if pd.notna(row.get("definition")) and row.get("definition") else "No standard definition available"
        inputs.append(f"Text: {content}\nMatched term: {dw}\nReturn JSON explaining the coded use.")
        targets.append(json.dumps({"dog_whistle_root": root, "ingroup": ingroup, "definition": defn,
                                   "explanation": f"In this text, '{dw}' is used as a coded reference to {ingroup} ideology. {defn}"}))
    df_out = pd.DataFrame({"input_text": inputs, "target_json": targets,
                           "dog_whistle": df["dog_whistle"].values, "dog_whistle_root": df["dog_whistle_root"].values,
                           "ingroup": df["ingroup"].values, "type": df["type"].values})
    df_out.to_parquet(f"{DATA_DIR}/processed/generation_{split_name}.parquet", index=False)
    print(f"{split_name}: {len(df_out)} rows")
print("Done.")
