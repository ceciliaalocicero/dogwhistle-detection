#!/bin/bash
#SBATCH --job-name=dw-gen-balanced
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

mkdir -p logs results/generation_balanced

echo "Job $SLURM_JOB_ID started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

for SEED in 42 123 7; do
    echo "=== Seed $SEED ==="
    python hpc_scripts/train_generator.py --config hpc_scripts/configs/generation_balanced.yaml --seed $SEED
done
python hpc_scripts/report_metrics.py --results_dir results/generation_balanced
echo "Job finished at $(date)"
