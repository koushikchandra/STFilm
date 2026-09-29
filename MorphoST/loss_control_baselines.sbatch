#!/bin/bash
#SBATCH --job-name=loss_ctrl_base
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=loss_ctrl_base_%a_%j.log
#SBATCH --array=0-89
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
MODELS=(stnet triplex mlpprobe)
COHORTS=(CCRCC COAD READ HCC IDC LYMPH_IDC LUNG PAAD PRAD SKCM)
SEEDS=(1 2 3)
M=${MODELS[$((SLURM_ARRAY_TASK_ID / 30))]}
C=${COHORTS[$((SLURM_ARRAY_TASK_ID % 10))]}
S=${SEEDS[$(((SLURM_ARRAY_TASK_ID / 10) % 3))]}
PYTHONPATH=STFlow python hest_baselines.py --model "$M" --cohort "$C" --seed "$S" \
  --corr_weight 0.5 --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_root results_corrected_hest_baselines_corr --device 0
