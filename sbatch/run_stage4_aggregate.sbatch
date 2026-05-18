#!/bin/bash
# =============================================================================
# Stage 4 causal aggregator.
# CPU-bound work (concatenate per-cell stage4_summary.csv, join with BSMI-R
# metadata, write cross-cell tables under results/stage4_causal/comparison/),
# but the general-partition QoS requires --gres=gpu:1 so we request one GPU.
# Launch with `--dependency=afterany:$JID_G:$JID_P:$JID_L` after the model
# array jobs.
# =============================================================================

#SBATCH --job-name=emnlp_stage4_agg
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage4-agg-%j.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage4-agg-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail
echo "[$(date)] host=$(hostname)"

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
$PYTHON aggregate_stage4_causal.py --config /home/anshulk/emnlp2026/config.yaml
echo "[$(date)] aggregator done"
