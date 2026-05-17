#!/bin/bash
# =============================================================================
# Stage 2c — Pythia 6.9B on the preempt partition.
#
# 8 array tasks, 1 A6000 per task, 1 Python worker per GPU.
# Resume-by-`computation_status: complete` makes preempt restarts free.
#
# Note (2026-05-17): single-worker setup — 2-worker packing failed with
# CUDA contention on the previous attempt.
# =============================================================================

#SBATCH --job-name=emnlp_stage2c_pythia
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-pythia-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-pythia-%A_%a.err
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-7
#SBATCH --exclude=babel-n5-20
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

ARRAY_SIZE=8
ARRAY_IDX=$SLURM_ARRAY_TASK_ID
CONFIG=/home/anshulk/emnlp2026/config.yaml
MODEL=pythia-6.9b

echo "[$(date)] array_task=$ARRAY_IDX of $ARRAY_SIZE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"
$PYTHON --version
$PYTHON -c "import gpytorch, torch; print(f'gpytorch={gpytorch.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}')"

$PYTHON stage2c_gplvm.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --sweep all \
    --array-task "$ARRAY_IDX" \
    --array-size "$ARRAY_SIZE"

echo "[$(date)] done array_task=$ARRAY_IDX"
