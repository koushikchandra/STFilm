#!/bin/bash
#SBATCH --job-name=morph_corr_hest
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=morph_corr_hest_%a_%j.log
#SBATCH --array=0-59
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
COHORTS=(CCRCC COAD READ HCC IDC LYMPH_IDC LUNG PAAD PRAD SKCM)
SEEDS=(1 2 3)
C=${COHORTS[$((SLURM_ARRAY_TASK_ID % 10))]}
S=${SEEDS[$(((SLURM_ARRAY_TASK_ID / 10) % 3))]}
CW=$((SLURM_ARRAY_TASK_ID / 30))
if [ "$CW" -eq 0 ]; then WEIGHT=0.0; LOSS=mse; else WEIGHT=0.5; LOSS=corr; fi
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_hest.py --version V3 --seed "$S" --cohort "$C" --corr_weight "$WEIGHT" \
  --source_dataroot ../dataset --embed_dataroot ../embed_dataroot \
  --save_root "results_corrected_hest_${LOSS}" --device cuda
