#!/bin/bash
#SBATCH --job-name=dw-multiclass
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

mkdir -p logs results/multiclass

echo "Job $SLURM_JOB_ID started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# RQ-B plan:
#   Run A: format=term       x seeds {42,123,7}  (main, 3-seed)
#   Run B: format=text_only  x seed 42           (ablation, 1-seed)
# Total: 4 trained models. Aggregation runs per format with report_metrics.py.

echo
echo "==================================================================="
echo "=== Run A: format=term  (main, 3 seeds)"
echo "==================================================================="
for SEED in 42 123 7; do
    echo
    echo "--- format=term  seed=$SEED  ($(date)) ---"
    python hpc_scripts/train_multiclass_ingroup.py \
        --config hpc_scripts/configs/multiclass.yaml \
        --seed "$SEED" \
        --input_format term
done
echo
echo "=== Aggregating across seeds for format=term ==="
python hpc_scripts/report_metrics.py --results_dir "results/multiclass/format_term"

echo
echo "==================================================================="
echo "=== Run B: format=text_only  (ablation, 1 seed)"
echo "==================================================================="
python hpc_scripts/train_multiclass_ingroup.py \
    --config hpc_scripts/configs/multiclass.yaml \
    --seed 42 \
    --input_format text_only
echo
echo "=== Aggregating across seeds for format=text_only ==="
python hpc_scripts/report_metrics.py --results_dir "results/multiclass/format_text_only"

echo "Job finished at $(date)"
