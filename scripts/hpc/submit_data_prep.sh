#!/bin/bash
#SBATCH --job-name=dw-data-prep
#SBATCH --account=3195720
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
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

# Ensure directories exist
mkdir -p data/manifests data/processed data/splits logs

echo "=== Data Prep ===" && date
python build_grouped_splits.py
python build_generation_targets.py
echo "=== Done ===" && date

# Note: RQ-A negatives are produced by submit_negatives.sh (the 3-stage
# pipeline), not by this script. Run that next.
