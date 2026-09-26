#!/bin/bash
#SBATCH --job-name=mouse_morph
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=mouse_morph_%a_%j.log
#SBATCH --array=0-5

# MorphoST (V3, components 111) on mouse HEST: LOOO (7 organs) + POOLED (5 folds), 3 seeds.
# 2 regimes x 3 seeds = 6 array tasks.
# Writes to results_mouse_morphost/; fold-level idempotent -> requeue-safe on scavenger.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[mouse_morph] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; print('[mouse_morph] cuda:', torch.cuda.is_available())"

REGIMES=(LOOO LOOO LOOO POOLED POOLED POOLED)
SEEDS=(1 2 3 1 2 3)

REGIME=${REGIMES[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

MOUSE_LOOO_FOLDS="brain liver skin heart bowel kidney muscle"
MOUSE_POOLED_FOLDS="0 1 2 3 4"

if [ "$REGIME" = "LOOO" ]; then
    FOLDS="$MOUSE_LOOO_FOLDS"
    NEED=7
else
    FOLDS="$MOUSE_POOLED_FOLDS"
    NEED=5
fi

TAG="${REGIME}_C111_seed${S}"
ROOT=../results_mouse_morphost

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[mouse_morph] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[mouse_morph] RESUME $TAG ($HAVE/$NEED)"

echo "[mouse_morph] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py \
  --regime "$REGIME" --version V3 --components 111 --seed "$S" \
  --folds $FOLDS \
  --splits_root ../mouse_cross_organ_splits \
  --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot \
  --save_root "$ROOT" \
  --device cuda
echo "[mouse_morph] $(date +%H:%M:%S) END $TAG"
