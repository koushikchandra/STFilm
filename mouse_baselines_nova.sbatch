#!/bin/bash
#SBATCH --job-name=mouse_bl_n
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=mouse_bl_n_%a_%j.log
#SBATCH --array=0-29

# Feature-matched baselines on mouse HEST -- nova partition for idle RTX_6000 nodes.
# 2 regimes x 5 models x 3 seeds = 30 tasks.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[mouse_bl_n] node=$(hostname) task=$SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[mouse_bl_n] cuda:', torch.cuda.is_available())"

REGIMES=(LOOO POOLED)
MODELS=(stnet histogene hist2st triplex bleep)
SEEDS=(1 2 3)

IDX=$SLURM_ARRAY_TASK_ID
REGIME=${REGIMES[$((IDX / 15))]}
REM=$((IDX % 15))
MODEL=${MODELS[$((REM / 3))]}
SEED=${SEEDS[$((REM % 3))]}

[ "$REGIME" = "LOOO" ] && NEED=7 || NEED=5
TAG="${REGIME}_${MODEL}_seed${SEED}"
ROOT=results_mouse_baselines

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[mouse_bl_n] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[mouse_bl_n] RESUME $TAG ($HAVE/$NEED)"

echo "[mouse_bl_n] $(date +%H:%M:%S) START $TAG"
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
echo "[mouse_bl_n] $(date +%H:%M:%S) END $TAG"
