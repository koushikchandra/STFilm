#!/bin/bash
#SBATCH --job-name=hest_uc_stem
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=hest_uc_stem_%a_%j.log
#SBATCH --array=0-2

# STEM with UNI+CONCH (1536-dim) intra-cohort — 3 seeds.
# Separate from other baselines due to T=1000 DDPM (much slower).
# Results → results_corrected_hest_uniconch_baselines/

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[hest_uc_stem] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: No CUDA"; exit 1; }

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

ROOT=results_corrected_hest_uniconch_baselines

echo "[hest_uc_stem] $(date +%H:%M:%S) START STEM seed$SEED (uni_conch 1536-dim, epochs=100 patience=20)"
PYTHONPATH=STFlow OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python hest_baselines.py \
  --model stem --seed "$SEED" --cohort all \
  --feature_encoder uni_conch \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --epochs 100 --patience 20 --n_steps 1000 \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[hest_uc_stem] $(date +%H:%M:%S) END STEM seed$SEED"
