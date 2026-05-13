#!/bin/bash
# =============================================================================
# Stage 2a aggregator — global BH-FDR + comparison tables.
# CPU-only; runs after run_stage2a.sbatch via afterok dependency.
# =============================================================================

#SBATCH --job-name=emnlp_stage2a_agg
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2a-agg-%j.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2a-agg-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
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

$PYTHON aggregate_stage2a_fourier_helix.py --config "$CONFIG"

echo "[$(date)] aggregator done"
