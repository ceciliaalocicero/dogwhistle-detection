# `data/manifests/`: provenance + ground-truth numbers

This folder is the **audit trail** for everything in `data/final/` and every numeric claim in the report. If a number quoted in `data/final/README.md`, the per-RQ reports, or the final report isn't directly computable from the parquets, it's pinned in one of these JSON / CSV files.

```
data/manifests/
├── README.md                                ← you are here
├── dataset_profiles.json                    upstream silent_signals row counts + schemas
├── split_manifest.json                      canonical root-grouped split (root → train/val/test)
├── negatives_manifest.json                  RQ-A negatives mining run metadata
├── negatives_adjudication_report.json       Stage 2 / 3 stats: contamination rate, dropped terms
├── term_coded_share.csv                     per-term coded-vs-pool ratios + pejorative flags (696 rows)
├── term_determinism_demo.json               40 grouped-test roots, with alt-split overlap evidence
├── rqb_multiclass_random_manifest.json      Run C alt-split build metadata
├── rqb_temperature.json                     RQ-B temperature scaling result (T*, NLL before/after)
└── model_inventory.json                     all 18 trained variants: per-seed metrics + HF Hub branches
```

---

## File-by-file

### `dataset_profiles.json`
Snapshot of the upstream `silent_signals` corpus and the locked HF eval datasets. Full corpus 16,258 rows (Informal 12,923 + Formal 3,335); 17 ingroups in the Informal subset (top three: `racist` 2,956, `white supremacist` 2,216, `antisemitic` 2,092). Also covers `detection` (101 rows), `disambiguation` (124 rows), and the `informal_potential` negatives pool (~6.03 M rows).

### `split_manifest.json`
The **canonical root-grouped split** (`seed=42`, generated 2026-04-27). Contains `root_to_split` — a 298-entry map from every dog-whistle root to one of `train` / `val` / `test` — plus per-split `n_rows`, `n_roots`, `ingroup_dist`. Source of truth for:
- The "0 train∩test root overlap" property (219 / 39 / 40 disjoint roots).
- The +28 pp anti-liberal prior shift (`525 / 9,064 = 5.8 %` train vs `720 / 2,095 = 34.4 %` test).

### `negatives_manifest.json`
Run metadata for Stage 1 of the RQ-A negatives mining: source = `informal_potential_dogwhistles` (Reddit), `max_per_term=10`, `min_text_len=50`, 6,902 candidate negatives across 695 terms. One term (`food stamps`) had no candidates in the pool. Carries the `noisy_negative_warning` disclosure that some "negatives" may be true coded uses missed by upstream annotation.

### `negatives_adjudication_report.json`
Stage 2 / 3 outputs:
- **Stage 2** (Llama-3.1-8B-Instruct judge): contamination rate **62.11 %**, unparseable rate 0.34 %.
- **Stage 3** (1:1 strict matching, `min_negatives_per_term=3`): final balanced set **8,828 positives + 8,828 negatives** across train / val / test (6,183 / 1,261 / 1,384). Lists all **100 dropped terms** with `K_positives`, `n_clean`, and reason — these are terms that couldn't reach the floor of 3 clean negatives, dominated by pejorative-only forms (`(((echo)))`, `troons`, `Pajeets`, `troon`, …). 1,037 positive rows lost to these drops.

### `term_coded_share.csv` (696 rows)
Per-dog-whistle stats from the negatives feasibility scan. Columns: `dog_whistle`, `n_pos` (published positives), `n_pool` (candidate negatives in the pool), `coded_share = n_pos / n_pool`, `pejorative_flag` (high-coded-share heuristic; **8 terms** above threshold), `zero_pool_flag` (1 term). The 100 actually-dropped terms in the adjudication report are a *superset* of the 8 `pejorative_flag=True` terms — the rest were dropped by the `min_negatives_per_term` floor rather than by being flagged pejorative.

### `term_determinism_demo.json`
The artefact that **backs the central RQ-B finding**. For each of the **40 grouped-test roots**, lists: `gold_ingroup`, an example `surface_form_in_sample` and Reddit comment, plus two flags: `in_alt_train` and `in_alt_test`. Result: **all 40 / 40** test roots appear in alt-split *train* (and 34 / 40 in alt-split *test*). This is the empirical evidence that the alt-split task collapses to glossary lookup, which is what produces Run C's 0.996 macro-F1. Includes a self-contained `explainer` field.

### `rqb_multiclass_random_manifest.json`
Build metadata for `data/final/rq_b_multiclass_random/` (the Run C alt-split). Source = `data/final/rq_b_multiclass`, `seed=42`, `stratify=ingroup`, val/test fractions 0.15 each, no small-class drops. Final counts 9,029 / 1,947 / 1,947 with full per-split `ingroup_distribution` matched to ~0.001 across folds — the design property that lets the A vs C contrast isolate root grouping as the only varying factor.

### `rqb_temperature.json`
Temperature scaling result for the demo's default RQ-B variant (`rqb_term_seed123`, term arm). `T* = 2.26` fitted on val (1,754 rows); NLL drops 2.33 → 1.53; argmax-preserving (only confidence values change). Pre-temp median confidence 0.997 with 55 % of predictions > 0.99 — the diagnostic that motivated calibration. Post-temp: median 0.75, no predictions > 0.95. Apply at inference: `probs = softmax(logits / T*)`.

### `model_inventory.json`
**The single source of truth for every trained variant.** Keyed `tasks → {rqa, rqb, rqc} → variants[]`, 18 variants total (6 per RQ). Each variant entry has: `id`, arm/seed/run_tag, local + HPC artefact paths, predictions path, `hf_branch`, and a `metrics` object with the headline metric and task-specific extras. Each task additionally pins:
- `headline_metric` (e.g. `disambiguation_124_f1` for RQA, `f1_macro` for RQB, `ingroup.macro_f1` for RQC).
- `best_variant`: raw single-seed leader by metric — sometimes a sensitivity-check (e.g. RQB's `term_altsplit_seed42` at 0.9964).
- `default_variant`: defensible demo default — chosen by per-task rules in `scripts/build_model_inventory.py`. Differs from `best_variant` for RQA (per-arm mean wins over lucky-seed peak) and RQB (Run C is excluded as a glossary-determinism artefact).
- `default_variant_rationale`: prose explanation; the place to look first when you need to defend a choice.

---

## Which manifest backs which claim

| Claim | Source |
|---|---|
| 219 / 39 / 40 root counts in train/val/test, 0 overlap | `split_manifest.json` |
| Anti-liberal +28 pp train→test prior shift | derivable from `split_manifest.json` `ingroup_dist` |
| 62.11 % Stage 2 contamination rate | `negatives_adjudication_report.json` |
| 8,828 balanced RQ-A positive/negative pairs | `negatives_adjudication_report.json` |
| 100 dropped terms (negatives mining) | `negatives_adjudication_report.json` |
| 99 % alt-split term-overlap with grouped-test roots | `term_determinism_demo.json` |
| RQ-A `disambiguation_124_f1` = 0.707 ± 0.015 | `model_inventory.json` → `tasks.rqa` |
| RQ-B grouped macro-F1 = 0.353 ± 0.007; Run C = 0.996 | `model_inventory.json` → `tasks.rqb` |
| RQ-C Path A `ingroup.macro_f1` = 0.379 ± 0.014 | `model_inventory.json` → `tasks.rqc` |
| RQ-B T* = 2.26 (calibration) | `rqb_temperature.json` |

---

## Regenerating

| File | Built by |
|---|---|
| `dataset_profiles.json` | `notebooks/eda.ipynb` (initial scan) |
| `split_manifest.json` | `notebooks/eda.ipynb` (split section) |
| `negatives_manifest.json`, `negatives_adjudication_report.json`, `term_coded_share.csv` | `notebooks/negatives.ipynb` + `scripts/hpc/` Stage-2 judge job |
| `term_determinism_demo.json` | `scripts/build_term_determinism_demo.py` |
| `rqb_multiclass_random_manifest.json` | `scripts/hpc/build_alt_split_rq_b.py` |
| `rqb_temperature.json` | calibration step in the HF Space build |
| `model_inventory.json` | `scripts/build_model_inventory.py` (re-run after every new training run) |
