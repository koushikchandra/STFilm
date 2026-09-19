#!/bin/bash
#SBATCH --job-name=pool5_mist
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=nova18-gpu-1,nova18-gpu-2,nova18-gpu-3,nova18-gpu-4,nova18-gpu-5,nova18-singularity,nova20-frost-1,nova20-matrix
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=pool5_mist_%a_%j.log
#SBATCH --array=0-2

# MIST (V3, components 111) POOLED on the SELF-CONTAINED 5-organ pool (no COAD/PAAD):
#   CCRCC(kidney), IDC(breast), LUNG(lung), PRAD(prostate), SKCM(skin).
# 5-fold pooled leave-patient-out; every organ present in train+test of every fold. 3 seeds.
# Splits: cross_organ_splits5_nocp/POOLED (leakage-safe 10-gene panel). UNI+CONCH (1536-dim).
# --exclude drops the v100 (Volta) nodes: they fail under CUDA 13 (cudaErrorNoKernelImage).

set -u

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # dir you sbatch from (the LOOO/ dir)
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/../../dataset}"
EMBED_ROOT="${EMBED_ROOT:-$REPO_DIR/../../embed_dataroot}"

cd "$REPO_DIR"
echo "[pool5_mist] node=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0}"
python -c "import torch; print('[pool5_mist] cuda:', torch.cuda.is_available())"

SEEDS=(1 2 3)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}

NEED=5
TAG="POOLED_C111_seed${SEED}"
ROOT=results_pooled5nocp_mist
HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
[ "$HAVE" -ge "$NEED" ] && { echo "[pool5_mist] SKIP $TAG (complete $HAVE/$NEED)"; exit 0; }
[ "$HAVE" -gt 0 ] && echo "[pool5_mist] RESUME $TAG ($HAVE/$NEED)"

echo "[pool5_mist] $(date +%H:%M:%S) START $TAG"
PYTHONPATH="$REPO_DIR" OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py \
  --regime POOLED --version V3 --components 111 --seed "$SEED" \
  --feature_encoder uni_conch \
  --splits_root cross_organ_splits5_nocp \
  --source_dataroot "$DATA_ROOT" \
  --embed_dataroot "$EMBED_ROOT" \
  --save_root "$ROOT" \
  --val_fraction 0.15 --epochs 100 --patience 20 --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24"
echo "[pool5_mist] $(date +%H:%M:%S) END $TAG"
