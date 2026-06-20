# `notebooks/`: the human-readable end-to-end pipeline

Three Jupyter notebooks that together walk through the full project: dataset exploration → RQ-A negatives mining → RQ-{A,B,C} training, evaluation, and unified reporting. The `.py` scripts in `scripts/hpc/` are the production runtimes that actually execute on the cluster; **the notebooks are the narrative wrapper around them** — many of pipeline.ipynb's cells are literally the same source as the HPC scripts, embedded inline so a reader can follow the logic without jumping between files.

```
notebooks/
├── README.md             ← you are here
├── eda.ipynb             58 cells · upstream `silent_signals` exploration + headline EDA findings
├── negatives.ipynb       33 cells · RQ-A negatives mining (3 stages) + emits HPC production scripts
└── pipeline.ipynb        98 cells · RQ-A/B/C training pipeline + ablations + error analysis + cascade
```

**Read in this order:** `eda` (understand the data) → `negatives` (build the RQ-A balanced pairs) → `pipeline` (train + evaluate + report). They share `data/final/` and `data/manifests/` as input/output state, but each is self-contained — re-running `pipeline.ipynb` doesn't require re-running the others if their outputs are already on disk.

---

## `eda.ipynb` — exploratory data analysis (58 cells)

Section map:

| § | Cells | What it shows |
|---|---|---|
| **1** | 3-4 | Loads all five datasets: `silent_signals` (full + Informal subset), `informal_potential_dogwhistles` (the negatives pool), and the locked HF eval sets (`detection`, `disambiguation`). |
| **1.5** | 5-6 | The 18 ingroup labels — counts and what each one means; whose taxonomy this is (Allen AI glossary). |
| **1.6** | 7-8 | **Determinism property**: asserts that 0 of 696 unique dog-whistle terms map to >1 ingroup. Writes `data/manifests/term_to_ingroup.csv`. This is the load-bearing fact behind the RQ-B Run A vs Run C gap. |
| **2** | 10-11 | Dataset overview — split sizes, feature presence. |
| **3** | 12-13 | Formal (Congressional) vs Informal (Reddit) split + ingroup distribution. |
| **4** | 15-18 | Term frequencies (head, tail, Zipf curve). |
| **5** | 19-24 | Word "map": every dog whistle sized by frequency and coloured by ingroup; per-ingroup word clouds. |
| **6** | 25-27 | Temporal trends (years, seasonality). |
| **7** | 28-35 | Where coded speech lives — top subreddits + Congressional party/chamber breakdowns + 1989-2023 candidate volume. |
| **8** | 36-37 | Content length — how much surrounding context per row. |
| **9** | 38-41 | How "ambiguous" each term is (per-term `coded_share`). |
| **10** | 42-46 | RQ-D term-level clustering — UMAP / KMeans on dog-whistle embeddings. **Scoped but not the headline**; demoted to supplementary in the final report. |
| **11** | 47-51 | Eval-set sanity check — how `detection_101` / `disambiguation_124` overlap with the corpus. |
| **11b** | 52-53 | **Prior shift**: train-vs-test ingroup distribution side-by-side. Annotates `anti-liberal` jumping from 5.8 % → 34.4 % (+28.6 pp), which explains the F1 = 0 on `anti-liberal` in RQ-B Run A. |
| **12** | 54-57 | Headline numbers — the one-screen "what to quote" summary. |

Outputs that downstream artefacts depend on: `data/manifests/term_to_ingroup.csv` (§ 1.6) and the prior-shift annotation in § 11b.

---

## `negatives.ipynb` — RQ-A negatives mining pipeline (33 cells)

The interactive companion to `scripts/hpc/{mine,adjudicate,balance}_negatives_full.py`. Cells 28-31 are `%%writefile` cells that **emit those production scripts directly** — when this notebook runs to completion, the HPC scripts are regenerated as a byproduct.

| § | Cells | What it does |
|---|---|---|
| **0-1** | 1-6 | Setup (`content_hash` helper, paths) + load `data/splits/` and the term vocabulary. |
| **2** | 7-10 | Per-term `coded_share` (n_pos / n_pool) and `pejorative_flag`. **Writes `data/manifests/term_coded_share.csv`** (696 rows). |
| **3** | 11-12 | Per-term capacity targets — how many candidates to mine per term (`BUFFER_FACTOR=3`, floor 3, ceiling 2000). |
| **4** | 13-16 | **Stage 1**: heuristic silver mining. Streams the 4-shard `informal_potential_dogwhistles` pool, content-hash excludes positives + locked eval, drops short content. Output → `data/processed/negatives_stage1_raw.parquet`. |
| **5** | 17-20 | **Stage 2**: LLM-as-judge adjudication. Llama-3.1-8B-Instruct via vLLM (in production) or HF transformers (locally). Parses replies into `{coded, non-coded, unparseable}`. |
| **6** | 21-24 | **Stage 3**: strict per-term 1:1 matching with `min_negatives_per_term=3` floor; assigns to splits via the term's root-to-split map. Drops 100 pejorative-only / floor-failing terms. |
| **7** | 25-26 | **Stage 4**: eval-set verification (no leakage into the locked eval sets) + writes the final `data/manifests/negatives_adjudication_report.json` with contamination rate, dropped terms, totals. |
| **8** | 27-31 | `%%writefile` cells that emit the canonical HPC production scripts (`mine_negatives_full.py`, `adjudicate_negatives_vllm.py`, `balance_negatives_full.py`, `submit_negatives.sh`) into `scripts/hpc/`. |
| **9** | 32 | "How to run" — invocation recipes (laptop sample run vs full HPC run). |

The notebook supports a `FULL` toggle (early in § 0) that switches between a small-sample dry run on a laptop and the full pipeline. **Don't re-run the FULL path locally** unless you need to; Stage 2 alone takes ~10 min on a 40 GB MIG slice and queues for hours.

---

## `pipeline.ipynb` — RQ-A/B/C training, eval, error analysis (98 cells)

This is the long one. It's the canonical narrative document for the project: every research decision, every config choice, every training script, every eval method, and every error analysis lives here in linear reading order. The `.py` files in `scripts/hpc/` are extracted for production use, but **the readable explanation is here**.

| § | Cells | What it does |
|---|---|---|
| **(intro)** | 0-3 | Title, "how to read this notebook", project framing, **research questions + hypotheses** (RQ-A binary disambiguation, RQ-B 17-way ingroup classification, RQ-C structured JSON generation). |
| **0** | 4-6 | Setup & utilities (`set_seed`, paths, common imports). |
| **1** | 7-9 | Inspect & filter dataset, including a Bender & Friedman 2018-style **abbreviated data statement** in § 1b. |
| **2** | 10-14 | Build root-grouped splits (the canonical 70/15/15 over 298 dog-whistle roots). Embeds `build_grouped_splits.py`. |
| **3** | 15-20 | Mine binary negatives — sectional pointer + the canonical `mine_binary_negatives.py` source. Actual interactive work lives in `negatives.ipynb`. |
| **4** | 21-24 | Build generation targets (the `{input_text, target_json}` pairs for RQ-C). Embeds `build_generation_targets.py`. |
| **4b** | 25-27 | Topic modelling exploration. **Scoped but supplementary**, not in the final report headline. |
| **5** | 28-36 | **Train binary disambiguator (RQ-A)**: input formatting, embedded `binary.yaml`, embedded `train_binary_disambiguator.py`, embedded `submit_binary.sh`. Loads HPC results back for downstream analysis. |
| **5b** | 37-39 | RQ-A baselines & ablations (TF-IDF, etc.). HPC-only smoke-train cells are skipped locally. |
| **5c** | 40-43 | **RQ-A error analysis**: per-class confusion, top-N high-confidence misses, precision-recall curves. |
| **6** | 44-52 | **Train multiclass ingroup classifier (RQ-B)**: same structure as § 5 — input format, embedded `multiclass.yaml`, embedded `train_multiclass_ingroup.py`, embedded `submit_multiclass.sh`. |
| **6b** | 53-55 | RQ-B baselines & ablations (text-only, etc.). |
| **7** | 56-70 | **Train explanation generator (RQ-C)**: Flan-T5-XL + LoRA setup, embedded `generation.yaml` + `train_generator.py` + `eval_generation.py` + `submit_generation.sh`. Also includes Path B variant: embedded `generation_balanced.yaml` + `submit_generation_balanced.sh`. |
| **7.3** | 71-73 | RQ-C baselines (template-fill, TF-IDF retrieval). |
| **7.4** | 74-75 | HPC generator output vs the baselines side-by-side. |
| **7.5** | 76-77 | **RQ-C error analysis**: JSON parse failures, per-field confusion. |
| **7.6** | 78-79 | Explanation-augmented classification — **scoped but not executed**. |
| **7.7** | 80 | Threats to validity (RQ-C-specific). |
| **8** | 81-82 | **Unified report**: aggregates `aggregated_results.json` from all RQs into one comparison table. |
| **8.5** | 83-84 | **SHAP token attributions** for the RQ-A model — explainability. |
| **8.7** | 85-86 | **End-to-end cascade pipeline** — RQ-A (gate) → RQ-C (generate explanation only if RQ-A says coded). |
| **9** | 87-97 | Shared HPC scaffolding: how to upload to the cluster, embedded `requirements.txt`, embedded `utils.py`, embedded `report_metrics.py`, embedded `submit_data_prep.sh`. |

The pattern in §§ 5/6/7: each RQ section walks **inputs → config → training script → SLURM submitter → load results back → analysis**, all in one place.

---

## Execution order + dependencies

```
   eda.ipynb                   negatives.ipynb            pipeline.ipynb
       │                              │                         │
       ├── reads silent_signals       ├── reads data/splits/     ├── reads data/final/
       │   (HF Hub)                   │   (built in pipeline §2) │   (downstream of negatives)
       │                              │                         │
       ├── writes term_to_ingroup.csv ├── writes manifests:      ├── reads results/ (post-HPC)
       │                              │   term_coded_share.csv  │   to render error analyses,
       └── writes EDA figures         │   negatives_*.json      │   tables, and unified report
                                      │                         │
                                      └── %%writefile           └── %%writefile
                                          scripts/hpc/*.py         scripts/hpc/*.py + configs/
```

Strictly speaking, `pipeline.ipynb` Section 2 builds the splits, so on a fresh checkout the order is:
1. `pipeline.ipynb` § 1-2 (build splits) →
2. `negatives.ipynb` (build RQ-A balanced pairs) →
3. `pipeline.ipynb` § 4 onwards (train + analyse)

`eda.ipynb` is independent — run any time after the splits exist.

---

## Practical notes

- **The `.py` files extracted from these notebooks are the source of truth on HPC.** When the notebook and `scripts/hpc/<x>.py` diverge, the HPC version wins (it's what actually trained the shipped models). Treat the in-notebook code as documentation; treat the `.py` files as runtime.
- **`pipeline.ipynb` cell 0 was rewritten on 2026-05-08** to reflect actual project state (was previously a project-proposal stub). If you see "we plan to explore topic modelling…" anywhere on cell 0, your copy is stale.
- **Many cells in pipeline.ipynb that look like full training runs are HPC-only and skip locally** with a printed `[skip]` line — they're there for narrative completeness, not to be executed on a laptop.
- **Re-executing `eda.ipynb` end-to-end takes ~3-5 min** on a laptop; `negatives.ipynb` (FULL=False) ~2 min; `pipeline.ipynb` ~10 min skipping HPC-only cells. Output cells are checked in so a grader doesn't have to re-execute anything to read the analysis.
