#!/bin/bash
#SBATCH --job-name=mouse_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=mouse_bl_%a_%j.log
#SBATCH --array=0-41

# Feature-matched baselines on mouse HEST: LOOO + POOLED, 7 models, 3 seeds.
# 2 regimes x 7 models x 3 seeds = 42 array tasks.
# Fold names auto-discovered from CSV; idempotent at fold level -> requeue-safe.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[mouse_bl] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; print('[mouse_bl] cuda:', torch.cuda.is_available())"

REGIMES=(LOOO POOLED)
MODELS=(stnet histogene hist2st genedml hyperst egn stem)
SEEDS=(1 2 3)

N_MODELS=${#MODELS[@]}
IDX=$SLURM_ARRAY_TASK_ID
REGIME=${REGIMES[$((IDX / (N_MODELS * 3)))]}
REM=$((IDX % (N_MODELS * 3)))
MODEL=${MODELS[$((REM / 3))]}
SEED=${SEEDS[$((REM % 3))]}

[ "$REGIME" = "LOOO" ] && NEED=7 || NEED=5
TAG="${REGIME}_${MODEL}_seed${SEED}"
ROOT=results_mouse_baselines

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[mouse_bl] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[mouse_bl] RESUME $TAG ($HAVE/$NEED)"

echo "[mouse_bl] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baseline_spatial.py \
  --model "$MODEL" --regime "$REGIME" --seed "$SEED" \
  --feature_encoder uni_v1_official \
  --splits_root mouse_cross_organ_splits \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 \
  --device 0
echo "[mouse_bl] $(date +%H:%M:%S) END $TAG"
