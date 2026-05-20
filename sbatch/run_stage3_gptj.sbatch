#!/bin/bash
# =============================================================================
# Stage 3 — Ownership test, GPT-J 6B (general partition).
#
# Per BSMI-R cell with a declared shape: orthogonalise the union basis B_u
# against the pre-registered correlate concepts (configs/stage3.yaml), then
# re-run BSMI-R evidence on Y_orth with alpha and P_hat LOCKED to the raw
# cell's values so log Z numbers are apples-to-apples comparable.
#
# 4 array tasks (each task processes its hashed cell stripe).
# =============================================================================

#SBATCH --job-name=emnlp_stage3_gptj
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage3-gptj-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage3-gptj-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
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
STAGE3_CFG=/home/anshulk/emnlp2026/configs/stage3.yaml
MODEL=gpt-j-6b

echo "[$(date)] array_task=$ARRAY_IDX of $ARRAY_SIZE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"
$PYTHON --version
$PYTHON -c "import numpy, pandas, sklearn; print(f'numpy={numpy.__version__} pandas={pandas.__version__} sklearn={sklearn.__version__}')"

$PYTHON stage3_ownership.py \
    --config "$CONFIG" \
    --stage3-config "$STAGE3_CFG" \
    --model "$MODEL" \
    --mode all \
    --sweep all \
    --array-task "$ARRAY_IDX" \
    --array-size "$ARRAY_SIZE"

echo "[$(date)] done array_task=$ARRAY_IDX"
