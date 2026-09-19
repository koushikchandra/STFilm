#!/bin/bash
#SBATCH --job-name=pool5_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=nova18-gpu-1,nova18-gpu-2,nova18-gpu-3,nova18-gpu-4,nova18-gpu-5,nova18-singularity,nova20-frost-1,nova20-matrix
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=pool5_bl_%a_%j.log
#SBATCH --array=0-11

# Baselines POOLED on the SELF-CONTAINED 5-organ pool (no COAD/PAAD):
#   CCRCC(kidney), IDC(breast), LUNG(lung), PRAD(prostate), SKCM(skin).
# Models: ST-Net, Hist2ST, BLEEP, STEM (4 models x 3 seeds = 12 tasks). UNI+CONCH (1536-dim).
# 5-fold pooled leave-patient-out; splits: cross_organ_splits5_nocp/POOLED.
# --exclude drops the v100 (Volta) nodes: they fail under CUDA 13 (cudaErrorNoKernelImage).

set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # the LOOO/ directory
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/../../dataset}"
EMBED_ROOT="${EMBED_ROOT:-$REPO_DIR/../../embed_dataroot}"

cd "$REPO_DIR"
echo "[pool5_bl] node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}"
python -c "import torch; print('[pool5_bl] cuda:', torch.cuda.is_available())"

MODELS=(stnet hist2st bleep stem)
SEEDS=(1 2 3)
N_MODELS=${#MODELS[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$((IDX % N_MODELS))]}
SEED=${SEEDS[$((IDX / N_MODELS))]}

NEED=5
TAG="POOLED_${MODEL}_seed${SEED}"
ROOT=baselines/results_pooled5nocp_baselines
HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
[ "$HAVE" -ge "$NEED" ] && { echo "[pool5_bl] SKIP $TAG (complete $HAVE/$NEED)"; exit 0; }
[ "$HAVE" -gt 0 ] && echo "[pool5_bl] RESUME $TAG ($HAVE/$NEED)"

echo "[pool5_bl] $(date +%H:%M:%S) START $TAG"
PYTHONPATH="$REPO_DIR" OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baselines/baseline_spatial.py \
  --model "$MODEL" --regime POOLED --seed "$SEED" \
  --feature_encoder uni_conch \
  --splits_root cross_organ_splits5_nocp \
  --source_dataroot "$DATA_ROOT" \
  --embed_dataroot "$EMBED_ROOT" \
  --save_root "$ROOT" \
  --val_fraction 0.15 --epochs 100 --patience 20 --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[pool5_bl] $(date +%H:%M:%S) END $TAG"
