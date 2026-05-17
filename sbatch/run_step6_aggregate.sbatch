#!/bin/bash
# Step 6 cross-model aggregation (CPU-only). Run after run_step6.sbatch finishes.
# Submit chained:
#   STEP6_JOBID=$(sbatch --parsable run_step6.sbatch)
#   sbatch --dependency=afterok:$STEP6_JOBID run_step6_aggregate.sbatch

#SBATCH --job-name=emnlp_step6_agg
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-step6agg-%j.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-step6agg-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

CONFIG=/home/anshulk/emnlp2026/config.yaml
echo "[$(date)] aggregation start  host=$(hostname)"

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
$PYTHON --version

$PYTHON compare_residualization_modes.py --config "$CONFIG"

echo "[$(date)] aggregation done"
