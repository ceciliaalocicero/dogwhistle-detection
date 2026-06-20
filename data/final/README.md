# `data/final/`: packaged datasets per research question

This folder is the **single source of truth for training/evaluation data** across all three research questions. Every notebook and HPC training script should load from here, not from the raw `data/splits/` directory and not from the Hugging Face Hub at runtime. Going through `data/final/` guarantees:

- everyone runs against the *same* data with the *same* preprocessing,
- HPC jobs don't need internet to fetch eval datasets mid-run,
- the locked eval sets stay out of every training pipeline by virtue of being filed under `eval_*` rather than `train`/`val`/`test`.

```
data/final/
├── README.md                           ← you are here
├── rq_a_binary/                        ← RQ-A: binary disambiguator (coded vs literal)
│   ├── train.parquet                   12,366 rows (6,183 pos + 6,183 neg)
│   ├── val.parquet                      2,522 rows (1,261 pos + 1,261 neg)
│   ├── test.parquet                     2,768 rows (1,384 pos + 1,384 neg) ← INTERNAL test (silver)
│   ├── eval_detection.parquet             101 rows ← LOCKED gold eval
│   └── eval_disambiguation.parquet        124 rows ← LOCKED gold eval
├── rq_b_multiclass/                    ← RQ-B: multiclass ingroup classifier (root-grouped split)
│   ├── train.parquet                    9,064 rows
│   ├── val.parquet                      1,764 rows
│   └── test.parquet                     2,095 rows
├── rq_b_multiclass_random/             ← RQ-B: random (non-grouped) split, sensitivity check only
│   ├── train.parquet                    9,029 rows
│   ├── val.parquet                      1,947 rows
│   └── test.parquet                     1,947 rows
└── rq_c_generation/                    ← RQ-C: generation (root + ingroup + definition)
    ├── train.parquet                    9,064 rows
    ├── val.parquet                      1,764 rows
    └── test.parquet                     2,095 rows
```

---

## RQ-A — Binary disambiguator (`rq_a_binary/`)

**Question.** Given a Reddit comment and a candidate dog-whistle phrase, is the phrase being used in its **coded** sense (label=1) or its **literal/non-coded** sense (label=0)?

### Training files

| File | Rows | Class balance |
|---|---|---|
| `train.parquet` | 12,366 | 50/50 (per term, by construction) |
| `val.parquet` | 2,522 | 50/50 |
| `test.parquet` | 2,768 | 50/50; this is **eval dataset 1** |

**How they were built.** Positives are the published `silent_signals` Informal subset (GPT-4 + manual validation). Negatives are mined from `informal_potential_dogwhistles` and judged by Llama-3.1-8B-Instruct on the Bocconi HPC. Per-term **strict 1:1 matching** with `MIN_NEGATIVES_PER_TERM = 3` floor. 100 pejorative-only terms (e.g. `(((echo)))`, `troons`, `Pajeets`) were dropped because they have no literal sense to disambiguate. The negatives-mining methodology is documented in the report's methods section.

### The three RQ-A eval datasets

| # | File | Rows | Source | Label provenance |
|---|---|---|---|---|
| 1 | `test.parquet` (the internal test split above) | 2,768 | This pipeline | **Silver**: positives from Kruk et al.'s GPT-4 + 14.7% manual-noise estimate; negatives from our Llama judge |
| 2 | `eval_detection.parquet` | 101 | `SALT-NLP/silent_signals_detection` | **Gold**: manually labelled by Kruk et al. |
| 3 | `eval_disambiguation.parquet` | 124 | `SALT-NLP/silent_signals_disambiguation` | **Gold**: manually labelled by Kruk et al. |

The two locked sets (2, 3) are **never** trained on. They are the only honest measure of model performance, since everything else has known label noise. Any pipeline that touches them outside of `model.evaluate()` is a contamination bug.

**Note on disambiguation labels.** The published HF dataset has three label strings: `coded` (67), `non-coded` (31), `noncoded` (26). The last two are the same class (one is a typo); both are mapped to `label=0`. Final distribution in `eval_disambiguation.parquet`: **67 coded / 57 non-coded**.

### Schema (uniform across all 5 RQ-A files)

| Column | Type | Notes |
|---|---|---|
| `dog_whistle` | str | Exact surface form found in the text |
| `dog_whistle_root` | str | Canonical root grouping (used for split-grouping) |
| `ingroup` | str | Target group ("antisemitic", "transphobic", …) |
| `content` | str | The comment text. (For `eval_detection`, the source column was `example`; renamed for consistency.) |
| `definition` | str | Short definition from `silent_signals` (1 sentence) |
| `definition_enriched` | str | Allen AI glossary description (joined on `dog_whistle_root`); empty string for unknown roots |
| `label` | int | 1 = coded, 0 = non-coded |
| `subreddit`, `date`, `type` | str | metadata; present where available |

### Definition ablation

The two definition columns enable a controlled ablation **without rebuilding any data**. Pick one and ignore the other:

```python
# Variant A: short definition
input_text = f"Candidate term: {row.dog_whistle}\nMeaning: {row.definition}\nText: {row.content}\nQuestion: Is this candidate used as a coded dog whistle here?"

# Variant B: enriched definition
input_text = f"Candidate term: {row.dog_whistle}\nMeaning: {row.definition_enriched}\nText: {row.content}\nQuestion: Is this candidate used as a coded dog whistle here?"
```

Run the trainer twice and report both numbers. The enriched variant has **zero coverage holes** (298/298 roots resolved), so this is a clean ablation.

### Quick-start (RQ-A)

```python
import pandas as pd

train = pd.read_parquet('data/final/rq_a_binary/train.parquet')
val   = pd.read_parquet('data/final/rq_a_binary/val.parquet')

# Three eval datasets, report metrics on each
eval_internal = pd.read_parquet('data/final/rq_a_binary/test.parquet')
eval_detection = pd.read_parquet('data/final/rq_a_binary/eval_detection.parquet')
eval_disambiguation = pd.read_parquet('data/final/rq_a_binary/eval_disambiguation.parquet')

print(f"train: {len(train):,} | val: {len(val):,}")
print(f"internal test: {len(eval_internal):,} (silver)")
print(f"detection eval: {len(eval_detection):,} (gold)")
print(f"disambiguation eval: {len(eval_disambiguation):,} (gold)")
```

---

## RQ-B — Multiclass ingroup classifier (`rq_b_multiclass/`)

**Question.** Given a Reddit comment containing a known dog whistle, predict the **`ingroup`** the dog whistle targets (e.g. `antisemitic`, `transphobic`, `homophobic`, …).

### Files

| File | Rows |
|---|---|
| `train.parquet` | 9,064 |
| `val.parquet`   | 1,764 |
| `test.parquet`  | 2,095 |

These are **direct copies of `data/splits/{train,val,test}.parquet`**, the full unfiltered positives. Pejorative-only terms (e.g. `(((echo)))`) are **kept** here, because the multiclass question is well-posed for them: knowing the term identifies the ingroup.

### Schema

Inherits from `data/splits/`. Key columns: `dog_whistle`, `dog_whistle_root`, `ingroup` (the target label), `content`, `definition`, plus metadata. There is **no `label` column**; the target is `ingroup`.

### Why these don't include negatives

RQ-B is a **closed-class classification** over ingroups present in the corpus. The negatives we mined for RQ-A are not labelled with an ingroup (they are the *non-coded* uses of the same surface forms), and including them would require either dropping them or labelling them as a synthetic "no-ingroup" class. Neither is what the research question asks.

### Sensitivity-check split (`rq_b_multiclass_random/`)

A second partition of the same RQ-B positives, **shuffled and re-split without grouping by `dog_whistle_root`** (random / ingroup-stratified). This is the alt-split used as the RQ-B sensitivity check (Run C in the report): it lets the model see train-time variants of test-time roots, so test performance collapses to near-memorisation (macro-F1 ≈ 0.996 vs ≈ 0.353 on the canonical grouped split). That gap is the central RQ-B finding — it quantifies how much of the apparent task difficulty is generalisation to unseen roots versus surface-form lookup. **Do not use this split for headline metrics.** Schema is identical to `rq_b_multiclass/`.

### What the report calls "Run A / B / C / D"

For convenience if you're cross-referencing these parquets against `docs/rq_b_report.md`:

| Run | Split (which folder) | Input arm | Loss | Macro-F1 | Role |
|---|---|---|---|---|---|
| **A** | `rq_b_multiclass/` (grouped) | `term` | plain CE | 0.353 ± 0.007 (3 seeds) | **Headline** |
| **B** | `rq_b_multiclass/` (grouped) | `text_only` | plain CE | 0.314 (1 seed) | Input ablation |
| **C** | `rq_b_multiclass_random/` (alt) | `term` | plain CE | 0.996 (1 seed) | Sensitivity check |
| **D** | `rq_b_multiclass/` (grouped) | `term` | weighted CE | 0.335 (1 seed) | Loss ablation |

All four runs use RoBERTa-base with identical hyperparameters; each varies exactly one thing relative to A. A → B swaps the input arm, A → C swaps the split (the headline contrast), A → D swaps the loss.

---

## RQ-C — Generation (`rq_c_generation/`)

**Question.** Given a Reddit comment containing a known dog whistle, generate a structured target: `{dog_whistle_root, ingroup, definition, explanation}`.

### Files

| File | Rows |
|---|---|
| `train.parquet` | 9,064 |
| `val.parquet`   | 1,764 |
| `test.parquet`  | 2,095 |

Same source as RQ-B (full unfiltered `data/splits/`). The training script is responsible for constructing the JSON `target_json` from `dog_whistle_root`, `ingroup`, `definition`, and the deterministic explanation template:
```
"In this text, '{dog_whistle}' is used as a coded reference to {ingroup} ideology. {definition}"
```

Why no negatives: same reasoning as RQ-B. For negatives the target is undefined.

---

## Cross-cutting notes

### Splitting is grouped by `dog_whistle_root`

All three RQs share the same train/val/test partition rule: every `dog_whistle_root` lives in exactly one split, so the model never sees train-time variants of a test-time root. This is why `dirty jew` and `Dirty Jew!` end up in the same split: they share the root `the_dirty_jew`.

### Label noise budget

| Source | Rate | Notes |
|---|---|---|
| RQ-A positives | ~14.7% | Kruk et al.'s manual sample on 400 rows |
| RQ-A negatives | ≥0% (unmeasured) | Llama-3.1-8B-Instruct judge; not cross-checked against GPT-4 |
| RQ-A locked eval (detection, disambiguation) | ~0% | Manual gold |
| RQ-B / RQ-C | ~14.7% | Same `silent_signals` source |

If RQ-A's F1 plateaus around ~0.85, that is consistent with the union of these noise sources, not necessarily a model problem.

### Where the eval datasets came from

`eval_detection.parquet` and `eval_disambiguation.parquet` are repackaged copies of the published Hugging Face datasets, pulled from the HPC's HF cache rather than re-downloaded, so they're the exact same artefact Kruk et al. published. We renamed `example` → `content` in detection for schema uniformity, and normalised the `noncoded` label typo in disambiguation. No row content is altered.

---

## Using these datasets on HPC

Push the directory once, then point training scripts at it:

```bash
# from your Mac
scp -r data/final/ <your_matricola>@slogin.hpc.unibocconi.it:~/dogwhistle_project/data/final/

# from inside a SLURM script
import pandas as pd
train = pd.read_parquet("./data/final/rq_a_binary/train.parquet")
```

The HPC compute nodes do have internet (we verified during stage 1 of negatives mining), but loading from `data/final/` is faster, deterministic, and doesn't depend on HF Hub being up. For long jobs, set `HF_HUB_OFFLINE=1` once the data is in place.

---

## Reproducibility: how this folder was built

The build is deterministic given the upstream artefacts. The pieces required to rebuild it from scratch are:

1. **Negatives pipeline.** Stage 1/2/3 of the negatives mining produces the balanced positives + negatives parquets. The pipeline is in `notebooks/negatives.ipynb` and the methodology is documented in the report's methods section.
2. **Original splits** at `data/splits/{train,val,test}.parquet` (root-grouped 70/15/15 partition over the `silent_signals` Informal subset). These are the upstream ground truth and are not shipped with the submission to keep size down.
3. **Enriched definitions** at `resources/dog_whistle_roots_enriched.csv`, regenerable via `resources/extract_definitions.py`.
4. **Locked eval datasets** cached locally: pull `silent_signals_{detection,disambiguation}` from the HF Hub via `datasets.load_dataset(...)`, then write the parquets into `eval_detection.parquet` and `eval_disambiguation.parquet`.

The build itself runs in under 5 seconds once the upstream artefacts are in place.

---

## Source

- `notebooks/pipeline.ipynb`: the training pipeline that consumes these files.
- `notebooks/negatives.ipynb`: the negatives-mining pipeline that produces the RQ-A balanced parquets.
- Kruk et al. (2024): [https://aclanthology.org/2024.acl-long.675](https://aclanthology.org/2024.acl-long.675)
