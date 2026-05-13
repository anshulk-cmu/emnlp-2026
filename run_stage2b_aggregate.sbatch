#!/bin/bash
# =============================================================================
# Stage 2b aggregator — global BH-FDR + cross-cell comparison tables.
# CPU-only; runs after run_stage2b.sbatch via afterok dependency.
# =============================================================================

#SBATCH --job-name=emnlp_stage2b_agg
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2b-agg-%j.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2b-agg-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

CONFIG=/home/anshulk/emnlp2026/config.yaml

echo "[$(date)] aggregator host=$(hostname)"
source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"

$PYTHON aggregate_stage2b_dsw.py --config "$CONFIG"

echo "[$(date)] aggregator done"
