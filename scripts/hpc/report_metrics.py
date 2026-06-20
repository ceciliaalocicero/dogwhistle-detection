"""Aggregate eval results across seeds into a final report.

Walks ``test_results.json`` files and aggregates every numeric leaf key,
recursing into nested dicts (e.g. ``ingroup.accuracy``,
``ingroup.macro_f1``, ``ingroup.per_class.<class>.f1``). Skips non-numeric
fields (lists, strings, sample_outputs). Means + std reported across seeds.

Output schema, per metric key:
    {
        "mean": float,
        "std": float,
        "by_seed": {"42": <value>, "123": <value>, "7": <value>},
        "values": [<value>, ...],   # legacy field, alphabetical seed order
                                    # (e.g. seed_123, seed_42, seed_7)
    }

Authoritative source for any per-seed value is ``by_seed``, keyed by
the integer seed as a string. The ``values`` list is kept for backward
compatibility but is in alphabetical-glob order (seed_123, seed_42, seed_7)
which is a footgun for anyone expecting numerical order. New consumers
must read ``by_seed``. See ``docs/audit_2026-05-08_doc_vs_reality.md``
Tier 1 #2/#3 for the bug class this prevents.
"""
import os, json, argparse, glob, re
import numpy as np

def _flatten(prefix, obj, out):
    """Recursively collect dotted-path numeric leaves into ``out`` dict."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("_"):
                continue
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)

def _seed_from_path(path):
    """Extract integer seed from a path like ``.../seed_42/test_results.json``."""
    m = re.search(r"seed_(\d+)", path)
    return int(m.group(1)) if m else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True)
    args = p.parse_args()

    # Glob in alphabetical order to preserve the legacy ``values`` order.
    paths = sorted(glob.glob(f"{args.results_dir}/seed_*/test_results.json"))
    all_flat = []
    for path in paths:
        with open(path) as f:
            r = json.load(f)
        flat = {}
        _flatten("", r, flat)
        flat["_path"] = path
        flat["_seed"] = _seed_from_path(path)
        all_flat.append(flat)

    if not all_flat:
        print("No results found."); return

    print(f"Found {len(all_flat)} seed results in {args.results_dir}")
    print(f"  Seeds (in alphabetical-glob order, for legacy ``values`` array): "
          f"{[r['_seed'] for r in all_flat]}")

    keys = set()
    for r in all_flat:
        keys.update(k for k in r if not k.startswith("_"))

    report = {}
    headline = ("json_parse_rate", "ingroup.accuracy", "ingroup.macro_f1", "ingroup.weighted_f1",
                "dog_whistle_root.accuracy", "dog_whistle_root.macro_f1",
                "definition.accuracy", "definition.macro_f1",
                "explanation_cosine_mean", "explanation_rouge_l_mean")
    for k in sorted(keys):
        vals = [r[k] for r in all_flat if k in r]
        # by_seed: authoritative per-seed mapping. Keyed by int seed as str.
        # Iterate in numerical seed order for stable JSON.
        by_seed = {
            str(r["_seed"]): r[k]
            for r in sorted(all_flat, key=lambda x: x["_seed"] if x["_seed"] is not None else 1e9)
            if k in r and r["_seed"] is not None
        }
        report[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "by_seed": by_seed,
            "values": vals,
        }

    print("\n=== Headline metrics (mean +/- std, then per-seed) ===")
    for k in headline:
        if k in report:
            r = report[k]
            seed_str = ", ".join(f"{s}={v:.4f}" for s, v in r["by_seed"].items())
            print(f"  {k:35s}  {r['mean']:.4f} +/- {r['std']:.4f}  ({seed_str})")
    print("\n=== All numeric metrics ===")
    for k in sorted(report):
        if k not in headline:
            print(f"  {k:50s}  {report[k]['mean']:.4f} +/- {report[k]['std']:.4f}")

    with open(f"{args.results_dir}/aggregated_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {args.results_dir}/aggregated_results.json")

if __name__ == "__main__":
    main()
