#!/bin/bash
# =============================================================================
# Stage 2c — GPT-J 6B on the general partition (no preemption).
#
# 8 array tasks, 1 A6000 per task, 1 Python worker per GPU.
#
# Note (2026-05-17): the previous 2-worker-per-GPU packing failed with
# `cudaErrorDevicesUnavailable` on most array tasks — without NVIDIA MPS the
# CUDA driver doesn't reliably share device contexts between two processes
# on the same GPU. Dropping to 1 worker per GPU is the safe, robust setup.
#
# Per-cell parallelism: `parallel=True` already enabled in analyze_cell —
# 3 seeds + 5 hold-out folds run on separate CUDA streams.
#
# Expected wall time on 8 GPUs: ~24–48 hours for all 497 eligible GPT-J cells
# (strong_wolfe LBFGS at production N adds wall vs the parent project).
# =============================================================================

#SBATCH --job-name=emnlp_stage2c_gptj
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-gptj-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-gptj-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

ARRAY_SIZE=8
ARRAY_IDX=$SLURM_ARRAY_TASK_ID
CONFIG=/home/anshulk/emnlp2026/config.yaml
MODEL=gpt-j-6b

echo "[$(date)] array_task=$ARRAY_IDX of $ARRAY_SIZE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"
$PYTHON --version
$PYTHON -c "import gpytorch, torch; print(f'gpytorch={gpytorch.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}')"

# Single worker on this GPU processes its hashed cell stripe.
$PYTHON stage2c_gplvm.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --sweep all \
    --array-task "$ARRAY_IDX" \
    --array-size "$ARRAY_SIZE"

echo "[$(date)] done array_task=$ARRAY_IDX"
