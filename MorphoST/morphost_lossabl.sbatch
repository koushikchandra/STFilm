#!/bin/bash
#SBATCH --job-name=morph_mse
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2
#SBATCH --output=morph_mse_%a_%j.log
#SBATCH --array=0-29
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
COHORTS=(CCRCC COAD READ HCC IDC LYMPH_IDC LUNG PAAD PRAD SKCM); SEEDS=(1 2 3)
C=${COHORTS[$(( SLURM_ARRAY_TASK_ID % 10 ))]}; S=${SEEDS[$(( SLURM_ARRAY_TASK_ID / 10 ))]}
echo "[mse] $C seed$S on $(hostname)"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_hest.py --version V3 --seed "$S" --cohort "$C" --corr_weight 0.0 \
  --source_dataroot ../dataset --embed_dataroot ../embed_dataroot \
  --feature_encoder uni_v1_official --save_root results_hest_mse --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
