#!/bin/bash
# =============================================================================
# Stage 3 aggregator — BH-FDR + summary tables over all per-cell results.
# CPU-bound, but the general-partition QoS requires --gres=gpu:1 so we
# request one (idle) GPU.
# =============================================================================

#SBATCH --job-name=emnlp_stage3_agg
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage3-agg-%j.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage3-agg-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

CONFIG=/home/anshulk/emnlp2026/config.yaml

echo "[$(date)] host=$(hostname)"
source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"

$PYTHON aggregate_stage3.py --config "$CONFIG" --alpha 0.10

echo "[$(date)] aggregator done"
