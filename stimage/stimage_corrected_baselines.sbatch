#!/bin/bash
#SBATCH --job-name=sti_corr_bl
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=sti_corr_bl_%a_%j.log
#SBATCH --array=0-17

# 6 models x 3 seeds = 18 tasks.
# Writes to results_corrected_stimage_baselines/ with nested inner-val (val_fraction=0.15).
# Regime: STImage-1K4M LOOO only (8 organ folds).

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[sti_bl] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[sti_bl] cuda:', torch.cuda.is_available())"

MODELS=(stnet histogene hist2st triplex genedml hyperst)
SEEDS=(1 2 3)
N_MODELS=${#MODELS[@]}
N_SEEDS=${#SEEDS[@]}

IDX=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$((IDX / N_SEEDS))]}
SEED=${SEEDS[$((IDX % N_SEEDS))]}

TAG="LOOO_${MODEL}_seed${SEED}"
ROOT=results_corrected_stimage_baselines
NEED=8  # 8 organ folds

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -eq "$NEED" ]; then
    echo "[sti_bl] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[sti_bl] RESUME $TAG ($HAVE/$NEED)"

echo "[sti_bl] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 \
  python baseline_spatial.py \
  --model "$MODEL" --regime LOOO --seed "$SEED" \
  --feature_encoder uni_v1_official \
  --splits_root stimage_splits \
  --source_dataroot dataset_stimage \
  --embed_dataroot embed_dataroot_stimage \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 --patience 20 \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[sti_bl] $(date +%H:%M:%S) END $TAG"
