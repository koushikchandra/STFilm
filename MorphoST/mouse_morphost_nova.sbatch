#!/bin/bash
#SBATCH --job-name=mouse_morph_n
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=mouse_morph_n_%a_%j.log
#SBATCH --array=2-5

# MorphoST on mouse HEST -- nova partition for idle RTX_6000 nodes.
# Tasks 0,1 (LOOO seed1,2) already running on scavenger; this covers 2-5:
#   2 = LOOO seed3, 3 = POOLED seed1, 4 = POOLED seed2, 5 = POOLED seed3

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[mouse_morph_n] node=$(hostname) task=$SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[mouse_morph_n] cuda:', torch.cuda.is_available())"

REGIMES=(LOOO LOOO LOOO POOLED POOLED POOLED)
SEEDS=(1 2 3 1 2 3)

REGIME=${REGIMES[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

[ "$REGIME" = "LOOO" ] && FOLDS="brain liver skin heart bowel kidney muscle" && NEED=7
[ "$REGIME" = "POOLED" ] && FOLDS="0 1 2 3 4" && NEED=5

TAG="${REGIME}_C111_seed${S}"
ROOT=../results_mouse_morphost

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[mouse_morph_n] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[mouse_morph_n] RESUME $TAG ($HAVE/$NEED)"

echo "[mouse_morph_n] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py \
  --regime "$REGIME" --version V3 --components 111 --seed "$S" \
  --folds $FOLDS \
  --splits_root ../mouse_cross_organ_splits \
  --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot \
  --save_root "$ROOT" \
  --device cuda
echo "[mouse_morph_n] $(date +%H:%M:%S) END $TAG"
