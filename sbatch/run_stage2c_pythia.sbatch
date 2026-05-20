#!/bin/bash
# =============================================================================
# Stage 2c BSMI-R — Pythia 6.9B on the general partition. 4 array tasks,
# A6000 per task. See docs/gplvm.md and the header of run_stage2c_gptj.sbatch.
# =============================================================================

#SBATCH --job-name=emnlp_stage2c_pythia
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-pythia-%A_%a.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-pythia-%A_%a.err
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
MODEL=pythia-6.9b

echo "[$(date)] array_task=$ARRAY_IDX of $ARRAY_SIZE host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader || true

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"
$PYTHON --version
$PYTHON -c "import numpy, pandas, sklearn; print(f'numpy={numpy.__version__} pandas={pandas.__version__} sklearn={sklearn.__version__}')"
$PYTHON -c "import ripser; print(f'ripser={ripser.__version__}')" 2>&1 | head -1 || \
  echo "[warn] ripser not installed; PH module will fall back to gudhi or report status=no_ph_lib"

$PYTHON stage2c_gplvm.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --mode all \
    --sweep all \
    --array-task "$ARRAY_IDX" \
    --array-size "$ARRAY_SIZE"

echo "[$(date)] done array_task=$ARRAY_IDX"
