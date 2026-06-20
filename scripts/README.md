# `scripts/`: training pipelines + manifest builders

Two flavours live here:

- **`hpc/`** — every script that runs on the Bocconi SLURM cluster: data prep, the 3-stage negatives mining pipeline, the three RQ trainers (RQ-A binary, RQ-B multiclass, RQ-C generator), evaluation + per-seed aggregation, plus one SBATCH submit script per logical run.
- **`scripts/` top-level** — local utilities that walk `results/` and `data/final/` to emit the canonical manifests in `data/manifests/`. Run on a laptop after results have been pulled from HPC.
- **`configs/`** — four YAMLs (one per training script) carrying every hyperparameter; CLI flags can override individual fields.

```
scripts/
├── README.md                            ← you are here
├── build_model_inventory.py             walks results/ → data/manifests/model_inventory.json
├── build_term_determinism_demo.py       40 grouped-test roots × alt-split overlap → demo manifest
│
├── configs/
│   ├── binary.yaml                      RQ-A: RoBERTa-base, fp32, label-smoothing 0.1, init_head_std
│   ├── multiclass.yaml                  RQ-B: RoBERTa-base, bf16
│   ├── generation.yaml                  RQ-C Path A: Flan-T5-XL + LoRA (r=16, α=32)
│   └── generation_balanced.yaml         RQ-C Path B: same + class_balance_train=1500
│
└── hpc/
    ├── utils.py                         load_config / parse_args / set_seed / sanitize / content_hash
    │
    ├── build_grouped_splits.py          silent_signals (Informal) → data/splits/, root-grouped
    ├── build_generation_targets.py      wraps {input_text, target_json} for RQ-C training
    ├── build_alt_split_rq_b.py          ingroup-stratified random split → rq_b_multiclass_random/
    │
    ├── mine_negatives_full.py           Stage 1: heuristic silver mining over informal_potential
    ├── adjudicate_negatives_vllm.py     Stage 2: vLLM Llama-3.1-8B-Instruct judge ('coded'/'non-coded')
    ├── balance_negatives_full.py        Stage 3: strict per-term 1:1 matching, min 3/term floor
    │
    ├── train_binary_disambiguator.py    RQ-A trainer (term + term_enriched_def arms)
    ├── train_multiclass_ingroup.py      RQ-B trainer (covers Runs A/B/C/D via flags)
    ├── train_generator.py               RQ-C trainer (Flan-T5-XL + LoRA via peft)
    ├── eval_generation.py               repair_json + parse + per-field metrics + TF-IDF cosine + ROUGE-L
    ├── report_metrics.py                aggregates test_results.json across seeds (mean/std + by_seed)
    │
    ├── submit_data_prep.sh              SBATCH: build_grouped_splits + build_generation_targets
    ├── submit_negatives.sh              SBATCH: Stages 1 → 2 → 3 of negatives mining
    ├── submit_binary.sh                 SBATCH: RQ-A 2 arms × 3 seeds + per-arm aggregation
    ├── submit_multiclass.sh             SBATCH: Run A (3 seeds) + Run B (1 seed)
    ├── submit_multiclass_alt.sh         SBATCH: Run C (alt-split, 1 seed)
    ├── submit_multiclass_weighted.sh    SBATCH: Run D (weighted-CE, 1 seed)
    ├── submit_generation.sh             SBATCH: Path A (3 seeds)
    └── submit_generation_balanced.sh    SBATCH: Path B (3 seeds, with class balancing)
```

---

## Pipeline flow

```
[silent_signals / HF Hub]
   │
   ├── build_grouped_splits.py ────────► data/splits/{train,val,test}.parquet + split_manifest.json
   │
   ├── build_generation_targets.py ───► data/processed/generation_{split}.parquet
   │
   └── mine_negatives_full.py (S1) ───► negatives_stage1_raw.parquet
        adjudicate_negatives_vllm (S2)─► negatives_stage2_judged.parquet
        balance_negatives_full (S3) ──► negatives_balanced_{split}.parquet + negatives_adjudication_report.json
                                             │
                                             ▼
                       data/final/rq_a_binary/  ←── (curated by negatives.ipynb post-processing)
                       data/final/rq_b_multiclass/
                       data/final/rq_c_generation/
                                             │
                                             ▼
   ┌────────────── train_binary_disambiguator.py ─────────────► results/binary/format_*/seed_*/
   │
   ├────────────── train_multiclass_ingroup.py ────────────────► results/multiclass/format_*/seed_*/
   │                  ↑ flags route to one of 4 runs:
   │                    Run A: defaults                        (term, grouped split)
   │                    Run B: --input_format text_only        (text-only ablation)
   │                    Run C: --data_subdir rq_b_multiclass_random --run_tag altsplit
   │                    Run D: --class_weighted_loss --run_tag weighted
   │
   ├────────────── train_generator.py ─────────────────────────► results/generation{,_balanced}/seed_*/
   │
   ├── report_metrics.py ──► aggregated_results.json (per arm/path; mean ± std + by_seed)
   │
   └── build_model_inventory.py ──► data/manifests/model_inventory.json
        build_term_determinism_demo.py ──► data/manifests/term_determinism_demo.json
```

---

## Top-level scripts

### `build_model_inventory.py`
Walks `results/{binary,multiclass,generation,generation_balanced}/` and emits `data/manifests/model_inventory.json` (the single source of truth for headline metrics + HF Hub branches). Pinned headline metrics: `disambiguation_124_f1` for RQA, `f1_macro` for RQB, `ingroup.macro_f1` for RQC. Critically, computes both `best_variant` (raw single-seed leader, sometimes a sensitivity check) and `default_variant` (defensible demo default) per task, with a free-text `default_variant_rationale` justifying any divergence — **the default-vs-best rules are encoded here**, see `pick_default_rqa/rqb/rqc()`. RQB explicitly excludes Run C (altsplit, 0.996) from defaulting because that would invert the central RQ-B finding.

### `build_term_determinism_demo.py`
Reads `data/final/rq_b_multiclass/test.parquet` (40 grouped-test roots) and `rq_b_multiclass_random/{train,test}.parquet`. For each grouped-test root, picks one Reddit-comment sample (40–350 chars) and records whether the same root appears in alt-split train and/or test. The output (`data/manifests/term_determinism_demo.json`) is the empirical artefact behind the **64-pp Run A vs Run C gap** — it shows all 40/40 test roots leak into alt-split train, which is why Run C scores 0.996.

---

## `configs/`

| File | Model | Notable settings |
|---|---|---|
| `binary.yaml` | `FacebookAI/roberta-base` | `epochs=5`, `lr=2e-5`, `bs=32`, `max_len=512`, `label_smoothing_factor=0.1`, `init_head_std=0.02`, `fp16=false`. Default arm = `term_enriched_def`; CLI `--input_format` overrides. |
| `multiclass.yaml` | `FacebookAI/roberta-base` | `epochs=5`, `lr=2e-5`, `bs=32`, `bf16=true`. Default arm = `term`. |
| `generation.yaml` | `google/flan-t5-xl` | `epochs=5`, `lr=3e-4`, `bs=4 × grad_accum=4`, LoRA `r=16, α=32, dropout=0.1`, target modules `[q, v]`, `bf16=auto`, `gradient_checkpointing=true`. |
| `generation_balanced.yaml` | same as `generation.yaml` | + `class_balance_train: 1500` (oversample/undersample each ingroup to 1500 rows in train; val/test untouched). |

The DeBERTa-v3-large NaN saga (2026-04-29) is documented in long-form comments at the top of `binary.yaml`. The `init_head_std=0.02` head re-init was added there and is preserved on RoBERTa as a safety belt; it's a no-op on a model that doesn't have head-init pathologies.

---

## `hpc/`

### Data prep

- **`utils.py`** — shared helpers: `load_config(yaml, overrides)`, `parse_args(extra_args=[...])`, `set_seed`, `print_diagnostics` (logs Python/CUDA/GPU/BF16 capability), `sanitize_text_column` (drops null/empty content), `content_hash` (lowercase + whitespace-normalised SHA256, used for cross-set deduplication in negatives mining).
- **`build_grouped_splits.py`** — loads `silent_signals` (Informal subset only), runs `StratifiedGroupKFold(n_splits=7)` for the test fold then `n_splits=6` on the train+val pool for the val fold, asserts disjoint `dog_whistle_root` across all three splits, writes `data/splits/{train,val,test}.parquet` + `data/manifests/split_manifest.json`. Seed 42, deterministic.
- **`build_generation_targets.py`** — for each split, builds the `{input_text: "Text: <content>\\nMatched term: <dw>\\nReturn JSON explaining the coded use.", target_json: "{<root>, <ingroup>, <definition>, <explanation>}"}` pairs. Explanation template is deterministic: `"In this text, '<dw>' is used as a coded reference to <ingroup> ideology. <definition>"`.
- **`build_alt_split_rq_b.py`** — see `data/final/README.md` § Sensitivity-check split. Pools `data/final/rq_b_multiclass/{train,val,test}` and re-splits stratified by `ingroup` (no root grouping). Seed 42, val/test = 0.15 each. Output: `data/final/rq_b_multiclass_random/` + a manifest.

### Negatives mining (Stages 1 → 2 → 3)

- **`mine_negatives_full.py` (Stage 1)** — streams the 4-shard `informal_potential_dogwhistles` pool from the HF cache, computes per-term targets `min(max(K * buffer_factor, min_floor), pool_count, hard_ceiling)` (defaults: 3, 3, 2000), content-hash excludes published positives + the locked detection/disambiguation eval sets, drops content < `min_text_len=50` chars. Writes `data/processed/negatives_stage1_raw.parquet`.
- **`adjudicate_negatives_vllm.py` (Stage 2)** — vLLM-batched LLM judge (default `meta-llama/Llama-3.1-8B-Instruct`, temperature 0, max_tokens 8). Builds a chat-templated prompt asking "is this phrase being used with the coded meaning?" and parses the first 60 chars of the reply into `{coded, non-coded, unparseable}` with a confidence pseudo-score. Writes `negatives_stage2_judged.parquet` with `judge_label`, `judge_confidence`, `judge_model`. Records the contamination rate (≈62%).
- **`balance_negatives_full.py` (Stage 3)** — strict per-term 1:1 matching: for each term with ≥`min_negatives_per_term=3` clean negatives, picks `min(K, n_clean)` of each side (positives + judged-non-coded negatives), assigns to splits via the term's root → split map (mode aggregation across positive rows of that root). Writes `positives_balanced_{split}.parquet` + `negatives_balanced_{split}.parquet` plus `negatives_adjudication_report.json` with the full dropped-terms list (100 terms, 1,037 lost positive rows).

### Trainers

- **`train_binary_disambiguator.py`** — RQ-A. Two input arms via `--input_format {term, term_enriched_def}`. Loads from `data/final/rq_a_binary/{train,val,test}.parquet` (with two fallbacks to legacy paths). Re-initialises pooler + classifier `nn.Linear` modules with N(0, `init_head_std`²) when set (guards against the DeBERTa NaN saga; harmless on RoBERTa). Trains with HF `Trainer`, early-stops on val F1, then evaluates on **all three** RQ-A datasets — internal silver test (2,768), locked detection (101), locked disambiguation (124) — and dumps per-row predictions parquets per dataset. Writes a flat `test_results.json` with prefixed keys (`disambiguation_124_f1`, etc.) plus a nested `_full` block with confusion matrices.
- **`train_multiclass_ingroup.py`** — RQ-B. **One script handles all four runs via flags**: defaults → Run A; `--input_format text_only` → Run B; `--data_subdir rq_b_multiclass_random --run_tag altsplit` → Run C; `--class_weighted_loss --run_tag weighted` → Run D. Builds the **train-only label space** (sorts ingroups present in train; val/test rows with unseen ingroups are dropped, recorded in `drop_report` — 10 `misogynistic` val rows on the canonical split). Custom `WeightedLossTrainer` subclass uses `nn.CrossEntropyLoss(weight=...)` with sklearn-balanced weights when `--class_weighted_loss` is on. Writes `test_results.json` (with `classification_report`, confusion matrix, drop_report) + `label_map.json` + per-row predictions.
- **`train_generator.py`** — RQ-C. Loads Flan-T5-XL, wraps it with peft `LoraConfig` (target modules `q, v`), trains with `Seq2SeqTrainer` and `predict_with_generate=True`, early-stops on val ROUGE-L. Optional `class_balance_train` resamples each ingroup in train to a target count (used for Path B). Calls `eval_generation.compute_metrics` post-train to compute headline numbers, writes `test_predictions.parquet` + `test_results.json` + saves the LoRA adapter weights (`best_model/`).

### Eval + aggregation

- **`eval_generation.py`** — both a library (`compute_metrics(df) → (results, df)` called from `train_generator.py`) and a CLI that recomputes metrics from existing `test_predictions.parquet` files without retraining. Key ops: `repair_json` (wraps in braces if Flan-T5 dropped them), `_safe_json_loads`, per-field accuracy + macro/weighted F1 + per-class P/R/F1 for `dog_whistle_root`/`ingroup`/`definition`, TF-IDF cosine on `explanation` vs gold, ROUGE-L on `explanation` (per-example then averaged).
- **`report_metrics.py`** — aggregates `test_results.json` across seeds for any results subdir. Recursively flattens to dotted-path numeric leaves (`ingroup.per_class.<class>.f1`), then for each metric emits `{mean, std, by_seed: {"42": ..., "123": ..., "7": ...}, values: [...]}`. **Use `by_seed` (numerically keyed), not `values` (alphabetical-glob order: 123, 42, 7)** — the legacy `values` ordering was the source of the seed-swap bug fixed in `docs/audit_2026-05-08_doc_vs_reality.md` Tier 1 #2/#3.

### SBATCH submit scripts

All scripts share the same SLURM header: `--account=3195720 --partition=stud --qos=stud --gpus=1 --cpus-per-task=8 --mail-user=3195720@studbocconi.it`, plus `module load sw/miniconda3 && conda activate dogwhistle`. Time limits: 24 h for trainers, 4 h for data prep / single-seed runs.

| Script | Time | What it runs |
|---|---|---|
| `submit_data_prep.sh` | 4 h | `build_grouped_splits.py` + `build_generation_targets.py` |
| `submit_negatives.sh` | 4 h | Stages 1 → 2 → 3 of negatives mining (Stage 2 takes ~10 min on a 40 GB MIG slice) |
| `submit_binary.sh` | 24 h | 2 arms (`term`, `term_enriched_def`) × 3 seeds (42, 123, 7) = 6 RQ-A runs + per-arm `report_metrics` |
| `submit_multiclass.sh` | 24 h | Run A (3 seeds) + Run B (1 seed at 42) + per-arm `report_metrics` |
| `submit_multiclass_alt.sh` | 4 h | Run C (1 seed at 42) — `--data_subdir rq_b_multiclass_random --run_tag altsplit` |
| `submit_multiclass_weighted.sh` | 4 h | Run D (1 seed at 42) — `--class_weighted_loss --run_tag weighted` |
| `submit_generation.sh` | 24 h | Path A (3 seeds) + `report_metrics` |
| `submit_generation_balanced.sh` | 24 h | Path B (3 seeds with `class_balance_train=1500`) + `report_metrics` |

---

## Quirks & gotchas (read before re-running on HPC)

1. **The submit scripts reference `hpc_scripts/`, not `scripts/hpc/`.** When `scp`-ing this folder onto the cluster, the canonical remote layout is `~/dogwhistle_project/hpc_scripts/` (see `CLAUDE.md` "Practical workflow notes"). The local `scripts/hpc/` rename was never propagated. Fix this on the cluster *before* invoking `sbatch`, or rewrite the script paths.
2. **Matricola `3195720` is hard-coded** in every `.sh` header (`--account` + `--mail-user`). Teammates running the same scripts must override these per their account.
3. **Stage 2 of negatives is the slow + expensive step** (~10 min compute, hours of queue, 40 GB MIG slice for vLLM). The output (`negatives_stage2_judged.parquet` + the balanced parquets in `data/processed/`) is checked in; don't rebuild without reason.
4. **`train_multiclass_ingroup.py` is the single source of truth for all 4 RQ-B runs.** No per-run training scripts — all behavioural variation is via CLI flags. Be careful when reading `submit_multiclass*.sh`: the same `python … train_multiclass_ingroup.py` line means a different run depending on the flags.
5. **`init_head_std=0.02` in `binary.yaml` is the DeBERTa-NaN aftermath.** Kept for safety but not load-bearing on RoBERTa. Document trail: `docs/sessions/2026-04-29_rqa_nan_postmortem.md`.
6. **`report_metrics.py`'s `values` field is in alphabetical-glob order** (`seed_123, seed_42, seed_7`). For per-seed reads, use `by_seed["42"]` etc. — the keyed dict was added specifically to prevent the seed-transcription bug that hit `rq_a_report.md` and `rq_c_report.md`.
7. **`build_model_inventory.py` stubs out RQC Path B variants** if `results/generation_balanced/` doesn't exist locally yet. The stubbed entries set `status: "hpc_only_pull_pending"` so the manifest stays exhaustive — but the metrics will be `null` until you `scp` the seed dirs over.
