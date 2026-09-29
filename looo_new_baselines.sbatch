#!/bin/bash
#SBATCH --job-name=looo_new_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=looo_new_bl_%a_%j.log
#SBATCH --array=0-11

# LOOO cross-organ for BLEEP, M2OST, EGN, STEM.
# 4 models x 3 seeds = 12 tasks. LOOO needs 8 organ folds each.
# Writes to results_corrected_crossorgan_baselines/.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[looo_new_bl] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; print('[looo_new_bl] cuda:', torch.cuda.is_available())"

MODELS=(bleep m2ost egn stem)
SEEDS=(1 2 3)

N_SEEDS=${#SEEDS[@]}
IDX=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$((IDX / N_SEEDS))]}
SEED=${SEEDS[$((IDX % N_SEEDS))]}

NEED=8
TAG="LOOO_${MODEL}_seed${SEED}"
ROOT=results_corrected_crossorgan_baselines

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[looo_new_bl] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[looo_new_bl] RESUME $TAG ($HAVE/$NEED)"

echo "[looo_new_bl] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baseline_spatial.py \
  --model "$MODEL" --regime LOOO --seed "$SEED" \
  --feature_encoder uni_v1_official \
  --splits_root cross_organ_splits8 \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs $([ "$MODEL" = "stem" ] && echo 500 || echo 100) \
  --n_steps $([ "$MODEL" = "stem" ] && echo 1000 || echo 50) \
  --device 0
echo "[looo_new_bl] $(date +%H:%M:%S) END $TAG"
