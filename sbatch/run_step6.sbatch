#!/bin/bash
# =============================================================================
# Step 6 — Per-model SLURM array job.
# One array task per model. Each task runs the full Step 6 pipeline on its
# assigned GPU:
#   (1) residualize activations into 3 caches (off, answer, norm)
#   (2) re-fit CCSVD for mode=answer and mode=norm (mode=off reuses Step 5)
#   (3) run LDA (Option A in CCSVD subspace + Option B in full 4096-D) for
#       all 3 modes
# Cross-mode aggregation is a separate post-job (run_step6_aggregate.sbatch).
#
# Resource policy:
#   - 3 array tasks, one per model (GPT-J, Llama, Pythia).
#   - 1 A6000 per task; SLURM auto-pins each task to its own GPU.
#   - 16 CPUs/GPU as user spec.
#   - --time=2-00:00:00 (2 days) for safety; expected wall ~3-6 hours per
#     model with cuML/cupy enabled.
#   - --mem=128G; activations fit easily, leaves headroom for cupy buffers.
# =============================================================================

#SBATCH --job-name=emnlp_step6
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-step6-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-step6-%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

set -euo pipefail

MODELS=("gpt-j-6b" "llama-3.1-8b" "pythia-6.9b")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
CONFIG=/home/anshulk/emnlp2026/config.yaml

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID model=$MODEL host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

# Use absolute conda env python to avoid system-python shadowing on compute nodes.
PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"
$PYTHON --version
$PYTHON -c "import pandas, numpy, scipy, sklearn, cupy, cuml; print(f'pandas={pandas.__version__} numpy={numpy.__version__} cupy={cupy.__version__} cuml={cuml.__version__}')"

# ─── Phase 1: Residualization ────────────────────────────────────────────────
echo "[$(date)] === Phase 1: residualize_activations (model=$MODEL) ==="
$PYTHON residualize_activations.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --modes off,answer,norm \
    --tasks addition,multiplication

# ─── Phase 2: CCSVD re-fits for mode=answer and mode=norm ────────────────────
# mode=off uses the existing Step 5 outputs; we do not re-run it.
for MODE in answer norm; do
    echo "[$(date)] === Phase 2: ccsvd_subspaces (model=$MODEL mode=$MODE) ==="
    $PYTHON ccsvd_subspaces.py \
        --config "$CONFIG" \
        --model "$MODEL" \
        --mode "$MODE"
done

# ─── Phase 3: LDA for all 3 modes ────────────────────────────────────────────
for MODE in off answer norm; do
    echo "[$(date)] === Phase 3: lda_subspaces (model=$MODEL mode=$MODE) ==="
    $PYTHON lda_subspaces.py \
        --config "$CONFIG" \
        --model "$MODEL" \
        --mode "$MODE"
done

echo "[$(date)] done task=$SLURM_ARRAY_TASK_ID model=$MODEL"
