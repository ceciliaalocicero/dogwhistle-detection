#!/bin/bash
#SBATCH --job-name=dw-negatives
#SBATCH --account=3195720
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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

mkdir -p logs data/processed data/manifests

echo "=== Stage 1: heuristic silver mining ==="
python hpc_scripts/mine_negatives_full.py --data_dir ./data

echo "=== Stage 2: LLM-as-judge adjudication (vLLM) ==="
python hpc_scripts/adjudicate_negatives_vllm.py --data_dir ./data \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 256

echo "=== Stage 3: per-term 1:1 matching ==="
python hpc_scripts/balance_negatives_full.py --data_dir ./data --min_negatives_per_term 3

echo "=== done ==="
ls -lh data/processed/negatives_balanced_*.parquet
