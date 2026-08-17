#!/bin/bash
#SBATCH --job-name=looo_uc_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=looo_uc_bl_%a_%j.log
#SBATCH --array=0-17

# LOOO cross-organ with UNI+CONCH (1536-dim).
# 6 non-STEM models x 3 seeds = 18 tasks. Each task runs 8 LOOO folds.
# Writes to results_crossorgan_uniconch_baselines/.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[looo_uc_bl] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; print('[looo_uc_bl] cuda:', torch.cuda.is_available())"

MODELS=(stnet hist2st hyperst bleep m2ost egn)
SEEDS=(1 2 3)
N_MODELS=${#MODELS[@]}

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % N_MODELS))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / N_MODELS))]}

NEED=8
TAG="LOOO_${MODEL}_seed${SEED}"
ROOT=results_crossorgan_uniconch_baselines

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[looo_uc_bl] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[looo_uc_bl] RESUME $TAG ($HAVE/$NEED)"

echo "[looo_uc_bl] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baseline_spatial.py \
  --model "$MODEL" --regime LOOO --seed "$SEED" \
  --feature_encoder uni_conch \
  --splits_root cross_organ_splits8 \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 --patience 20 \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[looo_uc_bl] $(date +%H:%M:%S) END $TAG"
