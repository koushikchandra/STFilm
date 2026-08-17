#!/bin/bash
#SBATCH --job-name=looo_uc_stem
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=looo_uc_stem_%a_%j.log
#SBATCH --array=0-2

# STEM LOOO with UNI+CONCH (1536-dim). 3 seeds, separate job due to T=1000 DDPM.
# Each task runs 8 LOOO folds (training on 7 organs per fold).
# Writes to results_crossorgan_uniconch_baselines/.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[looo_uc_stem] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; print('[looo_uc_stem] cuda:', torch.cuda.is_available())"

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

NEED=8
TAG="LOOO_stem_seed${SEED}"
ROOT=results_crossorgan_uniconch_baselines

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[looo_uc_stem] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[looo_uc_stem] RESUME $TAG ($HAVE/$NEED)"

echo "[looo_uc_stem] $(date +%H:%M:%S) START $TAG (uni_conch, epochs=100, patience=20, T=1000)"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baseline_spatial.py \
  --model stem --regime LOOO --seed "$SEED" \
  --feature_encoder uni_conch \
  --splits_root cross_organ_splits8 \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 --patience 20 --n_steps 1000 \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[looo_uc_stem] $(date +%H:%M:%S) END $TAG"
