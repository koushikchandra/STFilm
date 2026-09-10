#!/bin/bash
#SBATCH --job-name=hest_uc_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=hest_uc_bl_%a_%j.log
#SBATCH --array=0-17

# Intra-cohort baselines with UNI+CONCH (1536-dim) for fair dual-encoder comparison.
# 6 models x 3 seeds = 18 tasks. Runs all 10 HEST cohorts per task.
# Results → results_corrected_hest_uniconch_baselines/

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[hest_uc_bl] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: No CUDA"; exit 1; }

MODELS=(stnet hist2st hyperst bleep m2ost egn)
SEEDS=(1 2 3)
NM=${#MODELS[@]}

MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % NM))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / NM))]}

ROOT=results_corrected_hest_uniconch_baselines

echo "[hest_uc_bl] $(date +%H:%M:%S) START $MODEL seed$SEED (uni_conch 1536-dim)"
PYTHONPATH=STFlow OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python hest_baselines.py \
  --model "$MODEL" --seed "$SEED" --cohort all \
  --feature_encoder uni_conch \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[hest_uc_bl] $(date +%H:%M:%S) END $MODEL seed$SEED"
