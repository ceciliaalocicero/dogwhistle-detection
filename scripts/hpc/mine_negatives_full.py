"""Stage 1 — heuristic silver mining over the full informal_potential pool.

Reads positives from data/splits/, computes per-term targets, streams the
4-shard candidate pool, content-hash-excludes positives + eval sets, writes
data/processed/negatives_stage1_raw.parquet.
"""
import argparse, glob, hashlib, json, os
from collections import Counter
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset


def content_hash(text):
    return hashlib.sha256(" ".join(str(text).lower().strip().split()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--buffer_factor", type=int, default=3)
    ap.add_argument("--min_floor", type=int, default=3)
    ap.add_argument("--hard_ceiling", type=int, default=2000)
    ap.add_argument("--min_text_len", type=int, default=50)
    args = ap.parse_args()

    data = Path(args.data_dir)
    df_pos = pd.concat([pd.read_parquet(data / "splits" / f"{s}.parquet").assign(_split=s)
                        for s in ("train", "val", "test")], ignore_index=True)

    df_det = load_dataset("SALT-NLP/silent_signals_detection")["train"].to_pandas()
    df_dis = load_dataset("SALT-NLP/silent_signals_disambiguation")["train"].to_pandas()
    EXCLUDE = (set(df_pos["content"].map(content_hash))
               | set(df_det["example"].map(content_hash))
               | set(df_dis["content"].map(content_hash)))
    print(f"exclusion hashes: {len(EXCLUDE):,}")

    term_vocab = (df_pos[["dog_whistle", "dog_whistle_root", "definition"]]
                  .drop_duplicates("dog_whistle"))
    pos_counts = df_pos["dog_whistle"].value_counts().to_dict()

    POT_FILES = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--SALT-NLP--informal_potential_dogwhistles/"
        "snapshots/*/data/*.parquet")))
    if not POT_FILES:
        raise SystemExit("informal_potential cache missing — run datasets.load_dataset first")

    pool_dw = Counter()
    for f in POT_FILES:
        pool_dw.update(pq.read_table(f, columns=["dog_whistle"]).column("dog_whistle").to_pylist())

    targets = {}
    for term in term_vocab["dog_whistle"]:
        K = pos_counts.get(term, 0)
        if K == 0:
            continue
        target = min(max(K * args.buffer_factor, args.min_floor),
                     pool_dw.get(term, 0), args.hard_ceiling)
        if target > 0:
            targets[term] = target
    print(f"mining targets: {len(targets)} terms, sum={sum(targets.values()):,}")

    remaining = dict(targets)
    out, seen = [], set()
    for f in POT_FILES:
        tbl = pq.read_table(f, columns=["dog_whistle", "content", "subreddit", "date", "ingroup"])
        for row in tbl.to_pylist():
            term = row.get("dog_whistle")
            if term not in remaining or remaining[term] <= 0:
                continue
            text = (row.get("content") or "").strip()
            if len(text) < args.min_text_len:
                continue
            h = content_hash(text)
            if h in EXCLUDE or h in seen:
                continue
            seen.add(h)
            remaining[term] -= 1
            out.append({**row, "content": text})
        if all(v <= 0 for v in remaining.values()):
            break

    cand = pd.DataFrame(out).merge(
        term_vocab.drop_duplicates("dog_whistle"), on="dog_whistle", how="left")
    out_path = data / "processed" / "negatives_stage1_raw.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(out_path)
    print(f"wrote {out_path}  ({len(cand):,} rows)")


if __name__ == "__main__":
    main()
