#!/bin/bash
#SBATCH --job-name=recent_base_hest
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=recent_base_hest_%a_%j.log
#SBATCH --array=0-59
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# Feature-matched recent baselines (HyperST CVPR'26, Gene-DML) under the PER-COHORT HEST protocol,
# nested validation. Writes into results_corrected_hest_baselines so emit_final/emit_percohort pick
# them up as new Table 1/11 columns. Fold-level idempotent -> requeue-safe on scavenger.
MODELS=(hyperst genedml)
COHORTS=(CCRCC COAD READ HCC IDC LYMPH_IDC LUNG PAAD PRAD SKCM)
SEEDS=(1 2 3)
M=${MODELS[$((SLURM_ARRAY_TASK_ID / 30))]}
C=${COHORTS[$((SLURM_ARRAY_TASK_ID % 10))]}
S=${SEEDS[$(((SLURM_ARRAY_TASK_ID / 10) % 3))]}
echo "[recent_base_hest] START $M $C seed$S on $(hostname)"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python hest_baselines.py --model "$M" --cohort "$C" --seed "$S" \
  --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_root results_corrected_hest_baselines --device 0
echo "[recent_base_hest] DONE $M $C seed$S"
