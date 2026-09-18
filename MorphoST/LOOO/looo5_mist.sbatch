#!/bin/bash
#SBATCH --job-name=looo5_mist
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=looo5_mist_%a_%j.log
#SBATCH --array=0-2

# MIST (V3, components 111) LOOO on the SELF-CONTAINED 5-organ pool:
#   COAD(colorectal), CCRCC(kidney), LUNG(lung), PAAD(pancreas), PRAD(prostate).
# Each of the 5 folds holds out one organ and trains ONLY on the other 4. 3 seeds.
# Splits: cross_organ_splits5 (leakage-safe panels, 16-33 genes). UNI+CONCH (1536-dim).
#
# Self-contained: everything needed lives in this LOOO/ directory (vendored `stflow/` shim,
# `morphost.py`, `train.py`, `data.py`, `evaluation.py`, and the splits). Only the HEST DATA
# (h5ad expression + .h5 patch embeddings) lives outside -> set DATA_ROOT / EMBED_ROOT below.
# No cluster? just run the `python train.py ...` block directly.

set -u

# ---- EDIT THESE for your machine ---------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this LOOO/ directory
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/../../dataset}"          # <cohort>/adata/*.h5ad, var_*genes.json
EMBED_ROOT="${EMBED_ROOT:-$REPO_DIR/../../embed_dataroot}" # <cohort>/<encoder>/fp32/*.h5
# export PATH=/path/to/conda/bin:$PATH                     # uncomment/set your python env
# ------------------------------------------------------------------------------------------

cd "$REPO_DIR"
echo "[looo5_mist] node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: No CUDA"; exit 1; }

SEEDS=(1 2 3)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}

NEED=5
TAG="LOOO_C111_seed${SEED}"
ROOT=results_looo5_mist
HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
[ "$HAVE" -ge "$NEED" ] && { echo "[looo5_mist] SKIP $TAG (complete $HAVE/$NEED)"; exit 0; }
[ "$HAVE" -gt 0 ] && echo "[looo5_mist] RESUME $TAG ($HAVE/$NEED)"

echo "[looo5_mist] $(date +%H:%M:%S) START $TAG"
PYTHONPATH="$REPO_DIR" OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py \
  --regime LOOO --version V3 --components 111 --seed "$SEED" \
  --feature_encoder uni_conch \
  --splits_root cross_organ_splits5 \
  --folds colorectal kidney lung pancreas prostate \
  --source_dataroot "$DATA_ROOT" \
  --embed_dataroot "$EMBED_ROOT" \
  --save_root "$ROOT" \
  --val_fraction 0.15 --epochs 100 --patience 20 --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24"
echo "[looo5_mist] $(date +%H:%M:%S) END $TAG"
