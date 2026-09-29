#!/bin/bash
#SBATCH --job-name=sti_intra_bl
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=sti_intra_bl_%a_%j.log
#SBATCH --array=0-17

# 6 models x 3 seeds = 18 tasks.
# INTRA regime: per-organ 5-fold (same structure as HEST intra-cohort Table 1).
# Fold names are {organ}_{k} (brain_0..skin_4), 40 folds total.
# Writes to results_corrected_stimage_intra_baselines/.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[sti_ib] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[sti_ib] cuda:', torch.cuda.is_available())"

MODELS=(stnet histogene hist2st triplex genedml hyperst)
SEEDS=(1 2 3)
N_SEEDS=${#SEEDS[@]}

IDX=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$((IDX / N_SEEDS))]}
SEED=${SEEDS[$((IDX % N_SEEDS))]}

TAG="INTRA_${MODEL}_seed${SEED}"
ROOT=results_corrected_stimage_intra_baselines
NEED=40  # 8 organs x 5 folds

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -eq "$NEED" ]; then
    echo "[sti_ib] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[sti_ib] RESUME $TAG ($HAVE/$NEED)"

echo "[sti_ib] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 \
  python baseline_spatial.py \
  --model "$MODEL" --regime INTRA --seed "$SEED" \
  --feature_encoder uni_v1_official \
  --splits_root stimage_splits \
  --source_dataroot dataset_stimage \
  --embed_dataroot embed_dataroot_stimage \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[sti_ib] $(date +%H:%M:%S) END $TAG"
