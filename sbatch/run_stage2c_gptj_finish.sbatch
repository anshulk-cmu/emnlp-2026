#!/bin/bash
# =============================================================================
# Stage 2c BSMI-R — finish the 6 GPT-J multiplication / mode_off cells that
# the original 4-stripe sweep never reached.
#
# Single A6000, sequential per-cell. Each cell ~1-3 min, total ~10-20 min.
# =============================================================================

#SBATCH --job-name=emnlp_stage2c_gptj_finish
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-gptj-finish-%j.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-gptj-finish-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

CONFIG=/home/anshulk/emnlp2026/config.yaml
S2C_CFG=/home/anshulk/emnlp2026/configs/stage2c.yaml
MODEL=gpt-j-6b
TASK=multiplication
MODE=off

echo "[$(date)] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
$PYTHON --version

declare -a CELLS=(
  "4:b_tens"
  "8:ans_tens"
  "8:min_operand"
  "14:carry_tens"
  "20:ans_tens"
  "24:ans_hundreds"
)

for cell in "${CELLS[@]}"; do
  L=${cell%:*}; C=${cell#*:}
  echo "===== [$(date)] cell layer=$L concept=$C ====="
  T0=$(date +%s)
  $PYTHON stage2c_gplvm.py \
      --config "$CONFIG" \
      --stage2c-config "$S2C_CFG" \
      --model "$MODEL" \
      --task "$TASK" \
      --mode "$MODE" \
      --layer "$L" \
      --concept "$C"
  T1=$(date +%s)
  echo "===== [$(date)] cell layer=$L concept=$C done, wall=$((T1-T0))s ====="
done

echo "[$(date)] all 6 missing cells finished"
