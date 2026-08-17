#!/bin/bash
#SBATCH --job-name=mouse_intra_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=mouse_intra_bl_%a_%j.log
#SBATCH --array=0-146

# Mouse INTRA: 7 baselines x 7 organs x 3 seeds = 147 tasks.
# Each task runs one organ's 5-fold cross-validation (intra-organ).

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[mouse_intra_bl] node=$(hostname) task=$SLURM_ARRAY_TASK_ID"

MODELS=(stnet histogene hist2st genedml hyperst egn stem)
ORGANS=(brain liver skin heart bowel kidney muscle)
SEEDS=(1 2 3)

N_ORGANS=${#ORGANS[@]}; N_SEEDS=${#SEEDS[@]}
IDX=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$((IDX / (N_ORGANS * N_SEEDS)))]}
REM=$((IDX % (N_ORGANS * N_SEEDS)))
ORGAN=${ORGANS[$((REM / N_SEEDS))]}
SEED=${SEEDS[$((REM % N_SEEDS))]}

SPLITS_DIR="mouse_cross_organ_splits/INTRA/${ORGAN}"
TAG="INTRA_${ORGAN}_${MODEL}_seed${SEED}"
ROOT="results_mouse_intra_baselines"
NEED=5

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[mouse_intra_bl] SKIP $TAG ($HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[mouse_intra_bl] RESUME $TAG ($HAVE/$NEED)"

echo "[mouse_intra_bl] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python baseline_spatial.py \
  --model "$MODEL" --regime INTRA --seed "$SEED" \
  --splits_dir "$SPLITS_DIR" \
  --exp_code "$TAG" \
  --feature_encoder uni_v1_official \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 \
  --device 0
echo "[mouse_intra_bl] $(date +%H:%M:%S) END $TAG"
