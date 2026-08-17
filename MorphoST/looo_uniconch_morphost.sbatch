#!/bin/bash
#SBATCH --job-name=looo_uc_morph
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=looo_uc_morph_%a_%j.log
#SBATCH --array=0-2

# MorphoST V3 LOOO with UNI+CONCH (1536-dim). 3 seeds, 8 folds each.
# Writes to results_crossorgan_uniconch_morphost/.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[looo_uc_morph] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: No CUDA"; exit 1; }

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

NEED=8
TAG="LOOO_C111_seed${SEED}"
ROOT=results_crossorgan_uniconch_morphost

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[looo_uc_morph] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[looo_uc_morph] RESUME $TAG ($HAVE/$NEED)"

echo "[looo_uc_morph] $(date +%H:%M:%S) START $TAG (uni_conch 1536-dim)"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py \
  --regime LOOO --version V3 --components 111 --seed "$SEED" \
  --feature_encoder uni_conch \
  --splits_root ../cross_organ_splits8 \
  --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 --patience 20 \
  --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24"
echo "[looo_uc_morph] $(date +%H:%M:%S) END $TAG"
