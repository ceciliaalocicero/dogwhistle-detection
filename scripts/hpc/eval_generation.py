"""Course-aligned evaluation for the RQ-C generator.

Two entrypoints:

* ``compute_metrics(df) -> (results_dict, df_augmented)`` — used by
  ``train_generator.py`` right after the model dumps predictions.
* CLI: ``python eval_generation.py <results_dir>...`` — recomputes metrics
  from existing ``test_predictions.parquet`` files (no retraining needed).

Metrics: course-aligned headline numbers + ROUGE for completeness.

* per-field accuracy + per-class precision / recall / F1 + macro-F1
  (treating ``dog_whistle_root``, ``ingroup``, ``definition`` as
  classification problems on the test set) — L3;
* TF-IDF cosine similarity on the ``explanation`` field for the
  free-text portion — L5;
* ROUGE-L on the explanation field (per-example mean) — included as a
  recognised generation metric, kept secondary to the L3/L5 numbers.

Plumbing fixes applied here so EM/parse rates aren't artefacts:

* brace-repair on the raw ``pred_text`` before ``json.loads`` (the
  fine-tuned Flan-T5 omits the outer ``{...}``);
* lowercased + whitespace-stripped string comparison for EM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.metrics.pairwise import cosine_similarity


def repair_json(s: str) -> str:
    """Wrap the predicted body in braces if the model dropped them."""
    s = (s or "").strip()
    if not s:
        return s
    if not s.startswith("{"):
        s = "{" + s
    if not s.endswith("}"):
        s = s + "}"
    return s


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def _safe_json_loads(s: str) -> dict:
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def compute_metrics(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Take a per-row predictions frame, return (results_dict, augmented_df).

    Required columns: ``pred_text``, ``dog_whistle_root``, ``ingroup``,
    ``definition``. Optional: ``target_json`` (used as a sanity check),
    plus any identifying columns to carry through.
    """
    df = df.copy()
    raw = df["pred_text"].astype(str).tolist()
    repaired = [repair_json(s) for s in raw]
    parsed = [_safe_json_loads(s) for s in repaired]

    df["pred_text_repaired"] = repaired
    df["parsed_ok"] = [bool(p) for p in parsed]
    df["pred_root"] = [str(p.get("dog_whistle_root", "")) for p in parsed]
    df["pred_ingroup"] = [str(p.get("ingroup", "")) for p in parsed]
    df["pred_definition"] = [str(p.get("definition", "")) for p in parsed]
    df["pred_explanation"] = [str(p.get("explanation", "")) for p in parsed]

    n = len(df)
    results: dict[str, Any] = {
        "n_samples": n,
        "json_parse_rate": float(df["parsed_ok"].mean()) if n else 0.0,
    }

    # Per-field classification metrics. Macro-F1 is the headline because
    # the class distribution on test is imbalanced (anti-liberal ≈ 34 %).
    field_pairs = [
        ("dog_whistle_root", "pred_root"),
        ("ingroup", "pred_ingroup"),
        ("definition", "pred_definition"),
    ]
    for gold_col, pred_col in field_pairs:
        if gold_col not in df.columns:
            continue
        y_true = df[gold_col].map(_norm).tolist()
        y_pred = df[pred_col].map(_norm).tolist()
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=sorted(set(y_true)), zero_division=0
        )
        per_class = {
            cls: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for cls, p, r, f, s in zip(sorted(set(y_true)), precision, recall, f1, support)
        }
        results[gold_col] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": per_class,
        }

    # Free-text explanation: TF-IDF cosine similarity vs. gold explanation
    # (L5 embedding-similarity metric, computed per row then averaged).
    # Also compute per-example ROUGE-L on explanation for completeness.
    if "target_json" in df.columns:
        gold_expls = []
        for tj in df["target_json"].astype(str):
            g = _safe_json_loads(tj)
            gold_expls.append(str(g.get("explanation", "")))
        pred_expls = df["pred_explanation"].astype(str).tolist()

        nonempty = [(g, p) for g, p in zip(gold_expls, pred_expls) if g.strip() or p.strip()]
        if nonempty:
            vec = TfidfVectorizer().fit([g for g, _ in nonempty] + [p for _, p in nonempty])
            sims = []
            for g, p in zip(gold_expls, pred_expls):
                if not (g.strip() or p.strip()):
                    sims.append(0.0)
                    continue
                m = vec.transform([g, p])
                sims.append(float(cosine_similarity(m[0], m[1])[0, 0]))
            df["explanation_cosine"] = sims
            results["explanation_cosine_mean"] = float(np.mean(sims))
            results["explanation_cosine_median"] = float(np.median(sims))

        # ROUGE-L on the explanation field — secondary metric.
        try:
            import evaluate as hf_evaluate
            rouge = hf_evaluate.load("rouge")
            per_ex = []
            for g, p in zip(gold_expls, pred_expls):
                if not (g.strip() or p.strip()):
                    per_ex.append(0.0)
                    continue
                r = rouge.compute(predictions=[p], references=[g], use_stemmer=True)
                per_ex.append(float(r.get("rougeL", 0.0)))
            df["explanation_rouge_l"] = per_ex
            results["explanation_rouge_l_mean"] = float(np.mean(per_ex)) if per_ex else 0.0
        except Exception as e:
            print(f"[eval_generation] skipping ROUGE-L: {e}", file=sys.stderr)

    return results, df


def _summarise(results_dir: str, results: dict) -> None:
    print(f"\n=== {results_dir} ===")
    print(f"n_samples              {results.get('n_samples'):>6}")
    print(f"json_parse_rate        {results.get('json_parse_rate', 0.0):>6.3f}")
    for fld in ("dog_whistle_root", "ingroup", "definition"):
        m = results.get(fld)
        if m:
            print(
                f"{fld:<22} acc={m['accuracy']:.3f}  "
                f"macro_f1={m['macro_f1']:.3f}  weighted_f1={m['weighted_f1']:.3f}"
            )
    if "explanation_cosine_mean" in results:
        print(f"explanation_cosine     mean={results['explanation_cosine_mean']:.3f}  median={results['explanation_cosine_median']:.3f}")
    if "explanation_rouge_l_mean" in results:
        print(f"explanation_rouge_l    mean={results['explanation_rouge_l_mean']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "results_dirs",
        nargs="+",
        help="Directories containing test_predictions.parquet (one per seed).",
    )
    p.add_argument(
        "--output_name",
        default="test_results.json",
        help="JSON filename to write inside each results dir.",
    )
    args = p.parse_args()

    for d in args.results_dirs:
        pred_p = os.path.join(d, "test_predictions.parquet")
        if not os.path.exists(pred_p):
            print(f"skip {d}: no test_predictions.parquet", file=sys.stderr)
            continue
        df = pd.read_parquet(pred_p)
        results, df_aug = compute_metrics(df)
        df_aug.to_parquet(pred_p, index=False)
        with open(os.path.join(d, args.output_name), "w") as f:
            json.dump(results, f, indent=2)
        _summarise(d, results)


if __name__ == "__main__":
    main()
