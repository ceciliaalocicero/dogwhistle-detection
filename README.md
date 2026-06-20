# Hidden in Plain Sight: code

Bocconi 20597 (Natural Language Processing), Spring 2026, Group ID **01**.

**Authors:** Facundo Lucero, Alissa Sharuda, Jan Szkulepa, Valerio Costa, Cecilia Lo Cicero.

This `/code` folder is the reproducible companion to `project_report_01.pdf`. Three research questions on dog-whistle detection and disambiguation, all built on open-weight models with no paid APIs:

- **RQ-A.** Binary disambiguator. Given a Reddit comment containing a dog-whistle phrase, is the phrase being used in its **coded** sense (label = 1) or its **literal** sense (label = 0)?
- **RQ-B.** Multiclass classifier. Which **ingroup** (out of 17) does the dog whistle target?
- **RQ-C.** Structured generator. Produce a `(dog_whistle_root, ingroup, definition, explanation)` JSON for the comment.

All training runs on Bocconi HPC (SLURM, A100 80 GB MIG slices). Trained checkpoints are on the Hugging Face Hub — see `model_weights.txt` for URLs and download recipes.

---

## Layout

Each major subfolder has its own README that documents schemas, file formats, and the artefacts it produces. Read those for details.

```
code/
├── README.md             ← you are here
├── LICENSE               MIT
├── model_weights.txt     HF Hub URLs + download recipes for all 18 trained variants
│
├── notebooks/            ← README · the human-readable end-to-end pipeline
│   ├── eda.ipynb         · 58 cells · upstream silent_signals exploration + headline EDA
│   ├── negatives.ipynb   · 33 cells · RQ-A negatives mining (3 stages) → emits HPC scripts
│   └── pipeline.ipynb    · 98 cells · RQ-A/B/C training, eval, ablations, cascade
│
├── scripts/              ← README · production training pipelines + manifest builders
│   ├── build_model_inventory.py        → data/manifests/model_inventory.json
│   ├── build_term_determinism_demo.py  → data/manifests/term_determinism_demo.json
│   ├── configs/                        4 YAMLs (binary, multiclass, generation, generation_balanced)
│   └── hpc/                            data prep, negatives mining, 3 trainers, eval, 8 SBATCH submitters
│
├── data/
│   ├── final/            ← README · canonical train/val/test/eval bundles per RQ
│   │   ├── rq_a_binary/                train, val, test + LOCKED eval_detection (101) + LOCKED eval_disambiguation (124)
│   │   ├── rq_b_multiclass/            train, val, test (root-grouped — canonical)
│   │   ├── rq_b_multiclass_random/     alt-split (ingroup-stratified leak — sensitivity check only)
│   │   └── rq_c_generation/            train, val, test
│   └── manifests/        ← README · provenance JSONs (what backs which numeric claim)
│       ├── split_manifest.json                  root → split (298 roots, disjoint)
│       ├── negatives_{manifest,adjudication_report}.json   3-stage mining metadata
│       ├── term_coded_share.csv                 696 terms, coded vs pool share
│       ├── term_determinism_demo.json           40 grouped-test roots × alt-split overlap
│       ├── rqb_multiclass_random_manifest.json  alt-split build metadata
│       ├── rqb_temperature.json                 RQ-B calibration result (T* = 2.26)
│       ├── dataset_profiles.json                upstream corpus row counts + schemas
│       └── model_inventory.json                 ⭐ single source of truth for all 18 trained variants
│
├── resources/            ← README · Allen AI dog-whistle glossary + enrichment script
│   ├── Glossary_of_Dogwhistles.md       340 entries, source of definitions
│   ├── dog_whistle_roots*.{csv,txt}     298 roots + 696 surface forms
│   ├── allen_glossary.csv               structured parse of the glossary (340 rows)
│   ├── dog_whistle_roots_enriched.csv   roots × glossary fields (298 rows, 298/298 resolved)
│   └── extract_definitions.py           idempotent parser, no network
│
└── results/              ← README · per-seed metrics + per-row predictions
    ├── binary/                          RQ-A: format_term, format_term_enriched_def (3 seeds each)
    ├── multiclass/                      RQ-B: 4 runs (A, B, C, D)
    ├── generation/                      RQ-C Path A (unbalanced; 3 seeds)
    └── generation_balanced/             RQ-C Path B (oversampled to 1500/ingroup; 3 seeds)
```

---

## Models used

| Model | Hugging Face | Used for | Paper |
|---|---|---|---|
| RoBERTa-base | [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) | RQ-A binary disambiguator + RQ-B multiclass | Liu et al. 2019 |
| Flan-T5-XL | [`google/flan-t5-xl`](https://huggingface.co/google/flan-t5-xl) | RQ-C structured generator | Chung et al. 2024 |
| LoRA adapters | (on HF Hub, see `model_weights.txt`) | wraps Flan-T5-XL for RQ-C | Hu et al. 2022 |
| Llama-3.1-8B-Instruct | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Stage-2 negatives judge (RQ-A pipeline) | Grattafiori et al. 2024 |
| Source corpus | [`SALT-NLP/silent_signals`](https://huggingface.co/datasets/SALT-NLP/silent_signals) | upstream data | Kruk et al. 2024 |

Full BibTeX in the report.

---

## Main results

| RQ | Eval surface | Metric | Result | Baseline / contrast |
|---|---|---|---|---|
| **A** | `eval_disambiguation_124` (locked) | F1 | **0.707 ± 0.015** | 0.701 majority-coded (prevalence floor) |
| **A** | `eval_disambiguation_124` (locked) | Accuracy | **0.696 ± 0.010** | 0.540 majority (+15.6 pp) |
| **A** | `eval_disambiguation_124` (locked) | PR-AUC | **0.802 ± 0.010** | 0.50 random (+0.30) |
| **B** | Test (root-grouped, Run A) | macro-F1 | **0.353 ± 0.007** | 0.996 alt-split (Run C, the leaky control) — **the 64-pp gap is the result** |
| **C** | Test (Path A, unbalanced) | `ingroup` macro-F1 | **0.379 ± 0.014** | 0.344 modal-class accuracy |
| **C** | Test (Path A, unbalanced) | JSON parse rate | **0.975 ± 0.017** | 0 without `repair_json()` |

Per-seed values, full classification reports, and confusion matrices are in `results/<rq>/seed_*/test_results.json`. Aggregated mean ± std are in `data/manifests/model_inventory.json`. See `project_report_01.pdf` for full analysis.

---

## Verifying the numbers without re-training

Everything needed to verify the reported numbers is checked in:

- **Per-row predictions** for every (arm × seed) variant: `results/<rq>/<arm>/seed_*/predictions/*.parquet`. Each row carries the gold label, the model prediction, and the confidence/probability. Re-derive any metric from these directly with pandas + sklearn.
- **Per-seed metric JSONs**: `results/<rq>/<arm>/seed_*/test_results.json` (schemas in `results/README.md`).
- **Aggregated headline metrics** (mean ± std + `by_seed` keyed dict): `data/manifests/model_inventory.json`.
- **The artefact behind the central RQ-B finding** (40 grouped-test roots × their alt-split overlap): `data/manifests/term_determinism_demo.json`.

---

## Reproducing from scratch

The full pipeline runs on Bocconi HPC. Start with the narrative documents:

1. **`notebooks/README.md`** — read first. Covers what each notebook does and the execution order (`eda` → `negatives` → `pipeline`).
2. **`scripts/README.md`** — covers the production training pipelines, the `configs/` YAMLs, and the 8 SBATCH submitters. Includes a pipeline-flow diagram.

Cluster environment used end-to-end:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.10.0 + cu128 |
| transformers | 5.6.2 |
| vLLM | 0.19.1 (Stage-2 negatives adjudication only) |
| GPU | A100 80 GB MIG slice (`4g.40gb`) |

For local reproduction without HPC, `requirements.txt` (CPU/GPU) is in the repo root. `pandas>=2.2,<4` and `numpy>=1.26,<3` are bracket-pinned because pandas 3.x changed `groupby.apply` defaults and broke the pipeline notebook on a teammate's laptop on 2026-05-07.

### Running the SBATCH submitters on a different HPC account

The submitters in `scripts/hpc/submit_*.sh` hard-code our group's matricola in `--account=` and `--mail-user=`. To run from another account, edit those two `#SBATCH` lines in each submit script.

SSH access requires your own keypair on `slogin.hpc.unibocconi.it` (off campus also needs the Bocconi VPN). Standard one-time setup:

```bash
ssh-keygen -t rsa -b 4096
ssh-copy-id <your_matricola>@slogin.hpc.unibocconi.it
```

### Cluster-side path layout

For historical reasons, the SLURM submitters call `python hpc_scripts/<name>.py …` and reference configs at `hpc_scripts/configs/…`. **`hpc_scripts/` is the cluster-side name for what this submission ships as `scripts/hpc/`**, and on the cluster the configs sit one level deeper (inside `hpc_scripts/configs/`) rather than as a sibling of `hpc/`. The expected working tree on the cluster is:

```
~/dogwhistle_project/
├── hpc_scripts/            ← scripts/hpc/  + scripts/configs/  flattened together
│   ├── configs/            ← scripts/configs/ from this submission
│   ├── *.py                ← the 11 .py files from scripts/hpc/
│   └── submit_*.sh         ← the 8 SBATCH submitters
└── data/
    └── final_data/         ← data/final/ from this submission
```

This is a vestige of the development workflow; flagging it so the paths in the submitters don't read as a typo. See `scripts/README.md` § Quirks for more.

---

## Locked human-gold eval files

`data/final/rq_a_binary/eval_detection.parquet` (n = 101) and `eval_disambiguation.parquet` (n = 124) are **locked**. They are the only honest measure of RQ-A model performance — every other RQ-A surface carries silver-label noise. Touched only by `model.evaluate()` after model selection on val; never by `Dataset.map(tokenizer, …)` or `pd.concat`. Full policy + label provenance in `data/final/README.md` § "The three RQ-A eval datasets".

---

## License

MIT. See `LICENSE`.
