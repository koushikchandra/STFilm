#!/bin/bash
#SBATCH --job-name=sti_cx_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=sti_cx_bl_%a_%j.log
#SBATCH --array=0-35

# 2 regimes x 6 models x 3 seeds = 36 tasks.
# LOOO: 8 organ folds (need=8); POOLED: 5 folds (need=5).
# Nested inner-val (val_fraction=0.15), same protocol as STImage intra-organ Table 2.
# Writes to results_corrected_stimage_crossorgan_baselines/.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[scx] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[scx] cuda:', torch.cuda.is_available())"

REGIMES=(LOOO POOLED)
MODELS=(stnet histogene hist2st triplex genedml hyperst)
SEEDS=(1 2 3)
N_REGIMES=${#REGIMES[@]}; N_MODELS=${#MODELS[@]}; N_SEEDS=${#SEEDS[@]}

IDX=$SLURM_ARRAY_TASK_ID
REGIME=${REGIMES[$((IDX / (N_MODELS * N_SEEDS)))]}
REM=$((IDX % (N_MODELS * N_SEEDS)))
MODEL=${MODELS[$((REM / N_SEEDS))]}
SEED=${SEEDS[$((REM % N_SEEDS))]}

NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
TAG="${REGIME}_${MODEL}_seed${SEED}"
ROOT=results_corrected_stimage_crossorgan_baselines

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -eq "$NEED" ]; then
    echo "[scx] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[scx] RESUME $TAG ($HAVE/$NEED)"

echo "[scx] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baseline_spatial.py \
  --model "$MODEL" --regime "$REGIME" --seed "$SEED" \
  --feature_encoder uni_v1_official \
  --splits_root stimage_splits \
  --source_dataroot dataset_stimage \
  --embed_dataroot embed_dataroot_stimage \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[scx] $(date +%H:%M:%S) END $TAG"
