#!/bin/bash
#SBATCH --job-name=hest_base
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2
#SBATCH --output=hest_base_%a_%j.log
#SBATCH --array=0-20
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# Feature-matched baselines under STFlow's per-cohort HEST protocol. 7 models x 3 seeds, each runs
# all 10 cohorts. Scavenger (preemptible); fold-level idempotency resumes. Apples-to-apples columns
# for the MorphoST-V3 Table-1 comparison.
MODELS=(histogene hist2st stnet deepspace mlpprobe triplex bleep)
NM=${#MODELS[@]}          # 7
SEEDS=(1 2 3)
sidx=$(( SLURM_ARRAY_TASK_ID / NM ))
midx=$(( SLURM_ARRAY_TASK_ID % NM ))
SEED=${SEEDS[$sidx]}
MODEL=${MODELS[$midx]}
echo "[hest_base] START $MODEL seed$SEED on $(hostname)"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python hest_baselines.py --model "$MODEL" --seed "$SEED" --cohort all \
  --source_dataroot dataset --embed_dataroot embed_dataroot \
  --feature_encoder uni_v1_official --save_root results_hest_baselines --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[hest_base] DONE $MODEL seed$SEED"
