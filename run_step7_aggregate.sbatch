#!/bin/bash
#SBATCH --job-name=emnlp_step7_agg
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-step7agg-%j.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-step7agg-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] === Step 7 aggregator ==="
$PYTHON aggregate_residual_hunting.py --config /home/anshulk/emnlp2026/config.yaml
echo "[$(date)] aggregator done."
