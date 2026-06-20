"""Build a random comment-level RQ-B split, stratified by ingroup.

Produces the "alt split" used in Run C of the RQ-B writeup — same
comments as the canonical grouped split, but sampled without the
`dog_whistle_root` grouping constraint. Stratifying by ingroup means
the train/val/test class priors are matched, so any difference in
macro-F1 vs Run A isolates the effect of dropping the root constraint
(and the anti-liberal prior shift it induces).

Usage (run once, locally or on HPC):
    python scripts/hpc/build_alt_split_rq_b.py
Outputs:
    data/final/rq_b_multiclass_random/{train,val,test}.parquet
"""
import argparse, os, json
import numpy as np
import pandas as pd

DEFAULT_IN = "data/final/rq_b_multiclass"
DEFAULT_OUT = "data/final/rq_b_multiclass_random"
SEED = 42  # split is deterministic; downstream training still varies seeds.


def stratified_three_way(df, label_col, val_frac, test_frac, seed):
    """Stratified train/val/test split implemented in pandas (no sklearn).

    Within each label group: shuffle deterministically with `seed`, then
    take the first ceil(test_frac * n) rows as test, the next
    ceil(val_frac * n) as val, the rest as train. Using ceil rather than
    floor guarantees small classes still land at least one row in each
    held-out fold.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for _, group in df.groupby(label_col, sort=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(np.ceil(n * test_frac))) if n >= 3 else 0
        n_val = max(1, int(np.ceil(n * val_frac))) if n >= 3 else 0
        # Ensure at least one row stays in train.
        if n - n_test - n_val < 1:
            n_val = max(0, n - n_test - 1)
        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test:n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val:].tolist())
    return df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default=DEFAULT_IN)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    args = p.parse_args()

    pooled = pd.concat([pd.read_parquet(f"{args.in_dir}/{s}.parquet") for s in ("train", "val", "test")],
                       ignore_index=True)
    print(f"Pooled rows: {len(pooled)}; ingroup classes: {pooled['ingroup'].nunique()}")

    # Drop classes with <3 examples (can't stratify into 3 folds meaningfully).
    ig_counts = pooled["ingroup"].value_counts()
    too_small = ig_counts[ig_counts < 3].index.tolist()
    if too_small:
        print(f"Dropping classes with <3 rows: {too_small} "
              f"(rows: {ig_counts[too_small].to_dict()})")
        pooled = pooled[~pooled["ingroup"].isin(too_small)].reset_index(drop=True)

    train, val, test = stratified_three_way(
        pooled, "ingroup", args.val_frac, args.test_frac, SEED,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for name, df in (("train", train), ("val", val), ("test", test)):
        df = df.reset_index(drop=True)
        df.to_parquet(f"{args.out_dir}/{name}.parquet", index=False)
        print(f"  wrote {name} ({len(df)} rows)  ingroups={df['ingroup'].nunique()}")

    manifest = {
        "source": args.in_dir, "seed": SEED,
        "val_frac": args.val_frac, "test_frac": args.test_frac,
        "stratify": "ingroup",
        "dropped_small_ingroups": too_small,
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "ingroup_distribution": {
            "train": train["ingroup"].value_counts().to_dict(),
            "val": val["ingroup"].value_counts().to_dict(),
            "test": test["ingroup"].value_counts().to_dict(),
        },
    }
    with open(f"{args.out_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
