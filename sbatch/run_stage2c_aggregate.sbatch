#!/bin/bash
# =============================================================================
# Stage 2c aggregator — CPU-only.
# Concatenates per-cell gplvm_results.csv, applies BH-FDR on p_2c, downgrades
# verdicts post-FDR, and writes comparison tables + manifest.
# Launch via:  sbatch --dependency=afterok:$STAGE2C_JID run_stage2c_aggregate.sbatch
# =============================================================================

#SBATCH --job-name=emnlp_stage2c_agg
#SBATCH --output=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-agg-%j.out
#SBATCH --error=/data/user_data/anshulk/emnlp2026/logs/slurm-stage2c-agg-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anshulk@andrew.cmu.edu

# Aggregator is CPU-bound, but the general-partition QoS requires `gres=gpu:1`
# (matches the project's stage2b aggregator convention). Launch with an
# `afterany` dependency on the three per-model worker arrays so a partial
# preempt failure on Llama/Pythia still produces a comparison table from
# whatever did finish:
#
#   JID_G=$(sbatch --parsable sbatch/run_stage2c_gptj.sbatch)
#   JID_L=$(sbatch --parsable sbatch/run_stage2c_llama.sbatch)
#   JID_P=$(sbatch --parsable sbatch/run_stage2c_pythia.sbatch)
#   sbatch --dependency=afterany:$JID_G:$JID_L:$JID_P sbatch/run_stage2c_aggregate.sbatch

set -euo pipefail
echo "[$(date)] host=$(hostname)"

source /data/user_data/anshulk/miniconda3/etc/profile.d/conda.sh
conda activate geometry
cd /home/anshulk/emnlp2026

PYTHON=/data/user_data/anshulk/miniconda3/envs/geometry/bin/python
echo "[$(date)] PYTHON=$PYTHON"

$PYTHON aggregate_stage2c.py --config /home/anshulk/emnlp2026/config.yaml

echo "[$(date)] aggregator done"
