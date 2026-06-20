#!/bin/bash
#SBATCH --job-name=dw-binary
#SBATCH --account=3195720
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3195720@studbocconi.it

set -euo pipefail
module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate dogwhistle
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

mkdir -p logs results/binary

echo "Job $SLURM_JOB_ID started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Two-arm ablation (DESIGN_DEFENSE.md D6):
#   term              -> Candidate term + Text
#   term_enriched_def -> Candidate term + enriched glossary meaning + Text
# 3 seeds per arm = 6 trained models total.
for FORMAT in term term_enriched_def; do
    for SEED in 42 123 7; do
        echo
        echo "==================================================================="
        echo "=== Format=$FORMAT  Seed=$SEED  ($(date))"
        echo "==================================================================="
        python hpc_scripts/train_binary_disambiguator.py \
            --config hpc_scripts/configs/binary.yaml \
            --seed "$SEED" \
            --input_format "$FORMAT"
    done
    echo
    echo "=== Aggregating across seeds for format=$FORMAT ==="
    python hpc_scripts/report_metrics.py --results_dir "results/binary/format_$FORMAT"
done

echo "Job finished at $(date)"
