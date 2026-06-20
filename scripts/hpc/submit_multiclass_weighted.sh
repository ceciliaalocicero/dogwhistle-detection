#!/bin/bash
#SBATCH --job-name=dw-mc-wgt
#SBATCH --account=3195720
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=04:00:00
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

mkdir -p logs results/multiclass

echo "Job $SLURM_JOB_ID started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Run D: class-weighted CE loss on the canonical grouped split.
# 1 seed, format=term. Output dir: results/multiclass/format_term_weighted/seed_42/
echo
echo "=== Run D: weighted-CE  grouped split  format=term  seed=42 ==="
python hpc_scripts/train_multiclass_ingroup.py \
    --config hpc_scripts/configs/multiclass.yaml \
    --seed 42 \
    --input_format term \
    --class_weighted_loss \
    --run_tag weighted

python hpc_scripts/report_metrics.py --results_dir "results/multiclass/format_term_weighted"

echo "Job finished at $(date)"
