#!/bin/bash
#SBATCH --job-name=emnlp_ccsvd
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

MODELS=("gpt-j-6b" "llama-3.1-8b" "pythia-6.9b")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID model=$MODEL host=$(hostname)"

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry

cd /home/anshulk/emnlp2026
python ccsvd_subspaces.py \
    --config /home/anshulk/emnlp2026/config.yaml \
    --model "$MODEL"

echo "[$(date)] done task=$SLURM_ARRAY_TASK_ID model=$MODEL"
