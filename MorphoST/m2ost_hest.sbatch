#!/bin/bash
#SBATCH --job-name=m2ost_hest
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=m2ost_hest_%a_%j.log
#SBATCH --array=0-29
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# Feature-matched M2OST (AAAI 2025) under the PER-COHORT HEST protocol, nested validation, SAME
# splits/epochs (100)/patience(20)/val_fraction(0.15) as MorphoST and every other baseline. Writes
# into results_corrected_hest_baselines so emit picks it up as a new Table 1/11 column.
COHORTS=(CCRCC COAD READ HCC IDC LYMPH_IDC LUNG PAAD PRAD SKCM)
SEEDS=(1 2 3)
C=${COHORTS[$((SLURM_ARRAY_TASK_ID % 10))]}
S=${SEEDS[$((SLURM_ARRAY_TASK_ID / 10))]}
echo "[m2ost_hest] START m2ost $C seed$S on $(hostname)"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python hest_baselines.py --model m2ost --cohort "$C" --seed "$S" \
  --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_root results_corrected_hest_baselines --device 0
echo "[m2ost_hest] DONE m2ost $C seed$S"
