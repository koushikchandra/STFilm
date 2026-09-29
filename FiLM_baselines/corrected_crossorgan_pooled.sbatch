#!/bin/bash
#SBATCH --job-name=cx_pool
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2
#SBATCH --output=cx_pool_%a_%j.log
#SBATCH --array=0-59
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/FiLM_baselines
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# CORRECTED cross-organ POOLED baselines: nested validation (inner 15% val split, select on val,
# score test once). Writes to results_corrected_crossorgan/ -- NEVER the leaky results_film_baselines/.
# Fold-level idempotency in film_baselines.py makes preemption+requeue safe.
COMBOS=(
 "histogene none" "histogene desc" "histogene local"
 "hist2st none"   "hist2st desc"   "hist2st local"
 "triplex none"   "triplex desc"   "triplex local"
 "stnet none"     "stnet desc"     "stnet local"
 "deepspace none" "deepspace desc" "deepspace local"
 "mlpprobe none"  "mlpprobe desc"  "mlpprobe local"
 "bleep none"     "bleep desc"
)
NCOMBO=${#COMBOS[@]}          # 20
SEEDS=(1 2 3)
sidx=$(( SLURM_ARRAY_TASK_ID / NCOMBO ))
cidx=$(( SLURM_ARRAY_TASK_ID % NCOMBO ))
SEED=${SEEDS[$sidx]}
set -- ${COMBOS[$cidx]}
MODEL=$1; FILM=$2
echo "[cx_pool] START $MODEL-$FILM seed$SEED on $(hostname) gpu=$CUDA_VISIBLE_DEVICES"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python film_baselines.py --model "$MODEL" --film "$FILM" --regime POOLED --seed "$SEED" \
  --val_fraction 0.15 \
  --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot --feature_encoder uni_v1_official \
  --save_root results_corrected_crossorgan --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[cx_pool] DONE $MODEL-$FILM seed$SEED"
