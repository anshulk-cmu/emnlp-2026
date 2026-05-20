#!/bin/bash
# =============================================================================
# Stage 4 causal — GPT-J 6B, general partition, 4 array tasks, A6000 each.
# Runs Method 1 (ablation) + Method 2 (patching) at both subspace and
# geometry granularity for every BSMI-R cell with a declared shape.
# =============================================================================

#SBATCH --job-name=emnlp_stage4_gptj
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage4-gptj-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage4-gptj-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail
ARRAY_SIZE=4
ARRAY_IDX=$SLURM_ARRAY_TASK_ID
CONFIG=/home/anshulk/emnlp2026/config.yaml
MODEL=gpt-j-6b

echo "[$(date)] array_task=$ARRAY_IDX of $ARRAY_SIZE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
$PYTHON --version
$PYTHON -c "import torch, transformers; print(f'torch={torch.__version__} transformers={transformers.__version__}')"

$PYTHON causal_validation.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --mode all \
    --sweep all \
    --array-task "$ARRAY_IDX" \
    --array-size "$ARRAY_SIZE"

echo "[$(date)] done array_task=$ARRAY_IDX"
