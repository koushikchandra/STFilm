#!/bin/bash
#SBATCH --job-name=mouse_intra_morph
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=mouse_intra_morph_%a_%j.log
#SBATCH --array=0-20

# MorphoST INTRA on mouse: 7 organs x 3 seeds = 21 tasks.
# Each task runs one organ's 5-fold cross-validation (intra-organ).

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[mouse_intra_morph] node=$(hostname) task=$SLURM_ARRAY_TASK_ID"

ORGANS=(brain liver skin heart bowel kidney muscle)
SEEDS=(1 2 3)

IDX=$SLURM_ARRAY_TASK_ID
ORGAN=${ORGANS[$((IDX / 3))]}
SEED=${SEEDS[$((IDX % 3))]}

SPLITS_DIR="../mouse_cross_organ_splits/INTRA/${ORGAN}"
TAG="INTRA_${ORGAN}_C111_seed${SEED}"
ROOT="../results_mouse_morphost"
NEED=5

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[mouse_intra_morph] SKIP $TAG ($HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[mouse_intra_morph] RESUME $TAG ($HAVE/$NEED)"

echo "[mouse_intra_morph] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py \
  --regime POOLED --version V3 --components 111 --seed "$SEED" \
  --folds 0 1 2 3 4 \
  --splits_dir "$SPLITS_DIR" \
  --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot \
  --save_root "$ROOT" \
  --exp_code "$TAG" \
  --device cuda
echo "[mouse_intra_morph] $(date +%H:%M:%S) END $TAG"
