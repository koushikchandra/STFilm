#!/bin/bash
#SBATCH --job-name=looo5_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=looo5_bl_%a_%j.log
#SBATCH --array=0-11

# Baselines LOOO on the SELF-CONTAINED 5-organ pool (COAD, CCRCC, LUNG, PAAD, PRAD).
# Models: ST-Net, Hist2ST, BLEEP, STEM (4 models x 3 seeds = 12 tasks). UNI+CONCH (1536-dim).
# All model code (incl. BLEEP -> class BleepEncoder / bleep_train_fold) is inside
# baseline_spatial.py. baseline_spatial.py auto-derives the 5 folds from the split CSVs.
#
# Self-contained: uses the vendored `../stflow/` shim and `../evaluation.py` (both in LOOO/).
# Only the HEST DATA lives outside -> set DATA_ROOT / EMBED_ROOT below.
# No cluster? just run the `python baselines/baseline_spatial.py ...` block directly.

set -u

# ---- EDIT THESE for your machine ---------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # the LOOO/ directory
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/../../dataset}"
EMBED_ROOT="${EMBED_ROOT:-$REPO_DIR/../../embed_dataroot}"
# export PATH=/path/to/conda/bin:$PATH
# ------------------------------------------------------------------------------------------

cd "$REPO_DIR"
echo "[looo5_bl] node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}"
# Runs on any CUDA GPU, or falls back to CPU automatically (baseline_spatial.py picks the device).
# For a different cluster, edit the #SBATCH partition/account/gres above.
python -c "import torch; print('[looo5_bl] cuda:', torch.cuda.is_available())"

MODELS=(stnet hist2st bleep stem)
SEEDS=(1 2 3)
N_MODELS=${#MODELS[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((IDX % N_MODELS))]}
SEED=${SEEDS[$((IDX / N_MODELS))]}

NEED=5
TAG="LOOO_${MODEL}_seed${SEED}"
ROOT=baselines/results_looo5_baselines
HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
[ "$HAVE" -ge "$NEED" ] && { echo "[looo5_bl] SKIP $TAG (complete $HAVE/$NEED)"; exit 0; }
[ "$HAVE" -gt 0 ] && echo "[looo5_bl] RESUME $TAG ($HAVE/$NEED)"

echo "[looo5_bl] $(date +%H:%M:%S) START $TAG"
PYTHONPATH="$REPO_DIR" OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baselines/baseline_spatial.py \
  --model "$MODEL" --regime LOOO --seed "$SEED" \
  --feature_encoder uni_conch \
  --splits_root cross_organ_splits5 \
  --source_dataroot "$DATA_ROOT" \
  --embed_dataroot "$EMBED_ROOT" \
  --save_root "$ROOT" \
  --val_fraction 0.15 --epochs 100 --patience 20 --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[looo5_bl] $(date +%H:%M:%S) END $TAG"
