#!/bin/bash
#SBATCH --job-name=hest_cx_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=hest_cx_bl_%a_%j.log
#SBATCH --array=0-35

# 2 regimes x 6 models x 3 seeds = 36 tasks.
# LOOO: 8 organ folds (need=8); POOLED: 5 folds (need=5).
# Nested inner-val (val_fraction=0.15), same protocol as HEST intra-cohort Table 1.
# Writes to results_corrected_crossorgan_baselines/.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[hcx] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[hcx] cuda:', torch.cuda.is_available())"

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
ROOT=results_corrected_crossorgan_baselines

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -eq "$NEED" ]; then
    echo "[hcx] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[hcx] RESUME $TAG ($HAVE/$NEED)"

echo "[hcx] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baseline_spatial.py \
  --model "$MODEL" --regime "$REGIME" --seed "$SEED" \
  --feature_encoder uni_v1_official \
  --splits_root cross_organ_splits8 \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 --patience 20 \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[hcx] $(date +%H:%M:%S) END $TAG"
