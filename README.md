# Hidden in Plain Sight: Dog Whistle Detection, Classification, and Explanation

> **Project Summary**
>
> Developed an end-to-end NLP pipeline for detecting coded dog-whistle language on Reddit, classifying ideological ingroups, and generating structured explanations. Fine-tuned RoBERTa for binary and multiclass classification and Flan-T5-XL + LoRA for structured explanation generation. The project's central finding demonstrates that apparent near-perfect performance on ingroup classification is largely explained by evaluation leakage, with macro-F1 dropping from **0.996 to 0.353** when dog-whistle roots are properly held out during testing.

End-to-end NLP pipeline for detecting coded dog-whistle language on Reddit, classifying the targeted ingroup, and generating structured explanations using transformer-based language models.

This project was developed as part of the Natural Language Processing course at Università Bocconi and investigates the challenges of identifying politically and ideologically coded language that appears innocuous on the surface but conveys a hidden meaning to specific audiences.

---

## Project Overview

Dog whistles are phrases whose literal meaning differs from their intended ideological meaning. This ambiguity makes them difficult for traditional toxicity and hate-speech systems to detect.

This project addresses three NLP tasks:

1. **Binary Disambiguation**

   * Determine whether a phrase is used literally or as a coded dog whistle.

2. **Ingroup Classification**

   * Predict the ideological ingroup associated with a coded expression.

3. **Structured Explanation Generation**

   * Generate a structured explanation containing:

     * Dog-whistle root
     * Target ingroup
     * Definition
     * Human-readable explanation

---

## Main Research Finding

A central contribution of this work is demonstrating the impact of evaluation leakage.

When dog-whistle roots are allowed to appear in both training and testing sets:

* **Macro-F1 = 0.996**

When roots are fully held out:

* **Macro-F1 = 0.353**

This shows that much of the apparent performance in prior formulations can be explained by glossary memorization rather than contextual understanding. The result highlights the importance of root-stratified evaluation protocols when working with dog-whistle detection datasets.

---

## Pipeline

```text
Reddit Comments
        ↓
Dog Whistle Detection
     (RoBERTa)
        ↓
Ingroup Classification
     (RoBERTa)
        ↓
Structured Explanation
   (Flan-T5-XL + LoRA)
```

---

## What This Project Demonstrates

* Fine-tuning transformer models for NLP classification
* Context-aware language understanding
* Root-stratified evaluation to prevent data leakage
* Large Language Model (LLM) data curation pipelines
* Parameter-efficient fine-tuning with LoRA
* Structured text generation
* Reproducible experimentation and evaluation
* Error analysis and model interpretability

---

## Dataset

The project uses the **Silent Signals** dataset (Kruk et al., 2024), derived from the Allen AI Glossary of Dog Whistles.

### Dataset Characteristics

* ~13,000 coded Reddit examples
* 298 canonical dog-whistle roots
* 17 ideological ingroups
* English-language Reddit data

To avoid leakage, all train/validation/test splits were grouped by dog-whistle root, forcing models to generalize to previously unseen terms.

---

## Methodology

### Task 1 — Binary Disambiguation

#### Objective

Determine whether a phrase is being used literally or as a coded dog whistle.

#### Model

* RoBERTa-base
* Binary classification head

#### Input

* Target phrase
* Reddit comment context
* Binary classification prompt

#### Evaluation

* F1 Score
* Accuracy
* PR-AUC

---

### Task 2 — Ingroup Classification

#### Objective

Predict which ideological ingroup a coded expression targets.

#### Model

* RoBERTa-base
* 17-class classifier

#### Evaluation Protocol

A root-stratified split was used to prevent memorization of glossary entries and evaluate genuine contextual understanding.

---

### Task 3 — Structured Explanation Generation

#### Objective

Generate machine-readable explanations for coded language.

#### Model

* Flan-T5-XL
* LoRA adapters

#### Output Format

```json
{
  "dog_whistle_root": "...",
  "ingroup": "...",
  "definition": "...",
  "explanation": "..."
}
```

The generation pipeline includes automatic JSON repair to recover malformed outputs.

---

## Key Results

| Task                   | Model             | Result                |
| ---------------------- | ----------------- | --------------------- |
| Binary Disambiguation  | RoBERTa-base      | F1 = 0.707            |
| Binary Disambiguation  | RoBERTa-base      | PR-AUC = 0.802        |
| Ingroup Classification | RoBERTa-base      | Macro-F1 = 0.353      |
| Structured Generation  | Flan-T5-XL + LoRA | Macro-F1 = 0.379      |
| Structured Generation  | Flan-T5-XL + LoRA | 97.5% JSON Parse Rate |

---

## Technical Stack

### Machine Learning

* PyTorch
* Hugging Face Transformers
* RoBERTa-base
* Flan-T5-XL
* LoRA

### Data Processing

* Pandas
* NumPy
* Parquet datasets

### Experimentation

* Multi-seed evaluation
* Root-stratified splitting
* Class balancing experiments
* Confusion matrix analysis
* Reproducible configuration management

---

## Repository Structure

```text
data/           # Processed datasets
notebooks/      # Exploratory analysis
resources/      # Supporting assets
results/        # Experimental outputs and metrics
scripts/        # Training, evaluation, and HPC pipelines
```

---

## Skills Demonstrated

* Natural Language Processing (NLP)
* Transformer Fine-Tuning
* Text Classification
* Explainable AI
* Large Language Models (LLMs)
* LoRA Fine-Tuning
* Data Engineering
* Experimental Design
* Model Evaluation
* Research & Error Analysis

---

## Authors

Facundo Lucero
Alissa Sharuda
Jan Szkulepa
Valerio Costa
Cecilia Lo Cicero

Università Bocconi — Natural Language Processing
