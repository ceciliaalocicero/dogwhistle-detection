"""Stage 3 — strict per-term 1:1 matching + split assignment.

Writes paired positives_balanced_{split}.parquet and negatives_balanced_{split}.parquet
so the binary trainer sees identical per-term counts on both sides.
"""
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--min_negatives_per_term", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    data = Path(args.data_dir)

    df_pos = pd.concat([pd.read_parquet(data / "splits" / f"{s}.parquet").assign(_split=s)
                        for s in ("train", "val", "test")], ignore_index=True)
    judged = pd.read_parquet(data / "processed" / "negatives_stage2_judged.parquet")

    root_to_split = (df_pos.groupby("dog_whistle_root")["_split"]
                     .agg(lambda s: s.mode().iloc[0]).to_dict())
    K_per_term = df_pos.groupby("dog_whistle").size().to_dict()

    clean = judged[judged["judge_label"] == "non-coded"].copy()
    print(f"clean pool: {len(clean):,}")

    balanced_neg, balanced_pos, dropped = [], [], []
    for term, K in K_per_term.items():
        avail = clean[clean["dog_whistle"] == term].sort_values(
            "judge_confidence", ascending=False)
        n_avail = len(avail)
        if n_avail < args.min_negatives_per_term:
            dropped.append({"dog_whistle": term, "K_positives": int(K),
                            "n_clean": int(n_avail),
                            "reason": "below_min_negatives_floor"})
            continue
        n_pair = min(int(K), int(n_avail))
        picked_neg = avail.head(n_pair).copy()
        picked_neg["_split"] = picked_neg["dog_whistle_root"].map(root_to_split)
        balanced_neg.append(picked_neg)
        picked_pos = df_pos[df_pos["dog_whistle"] == term].sample(
            n=n_pair, random_state=args.seed).copy()
        balanced_pos.append(picked_pos)

    bal = pd.concat(balanced_neg, ignore_index=True) if balanced_neg else pd.DataFrame()
    bal_p = pd.concat(balanced_pos, ignore_index=True) if balanced_pos else pd.DataFrame()

    for s in ("train", "val", "test"):
        neg_sub = (bal[bal["_split"] == s].drop(columns=["_split"], errors="ignore")
                   .assign(label=0, type="Informal",
                           source_dataset="SALT-NLP/informal_potential_dogwhistles"))
        neg_out = data / "processed" / f"negatives_balanced_{s}.parquet"
        neg_sub.to_parquet(neg_out)
        print(f"wrote {neg_out}  ({len(neg_sub):,} rows)")

        pos_sub = (bal_p[bal_p["_split"] == s].drop(columns=["_split"], errors="ignore")
                   .assign(label=1))
        pos_out = data / "processed" / f"positives_balanced_{s}.parquet"
        pos_sub.to_parquet(pos_out)
        print(f"wrote {pos_out}  ({len(pos_sub):,} rows)")

    n_pos_dropped = sum(d["K_positives"] for d in dropped)
    manifest = {
        "stage3_balanced_total_negatives": int(len(bal)),
        "stage3_balanced_total_positives": int(len(bal_p)),
        "stage3_per_split": (bal["_split"].value_counts().to_dict() if len(bal) else {}),
        "stage3_dropped_terms": dropped,
        "stage3_dropped_term_count": len(dropped),
        "stage3_positives_lost_to_drop": int(n_pos_dropped),
        "stage3_min_negatives_per_term": args.min_negatives_per_term,
        "stage2_contamination_rate": float((judged["judge_label"] == "coded").mean()),
        "stage2_unparseable_rate": float((judged["judge_label"] == "unparseable").mean()),
        "stage2_judge_model": str(judged["judge_model"].iloc[0]) if len(judged) else None,
    }
    with open(data / "manifests" / "negatives_adjudication_report.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
