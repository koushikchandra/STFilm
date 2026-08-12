#!/bin/bash
#SBATCH --job-name=morpho_hest
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2
#SBATCH --output=morpho_hest_%a_%j.log
#SBATCH --array=0-29
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# MorphoST V3 under STFlow's PER-COHORT protocol (within-organ k-fold CV), for a Table-1-style
# comparison. 10 HEST cohorts x 3 seeds. Scavenger (preemptible); fold-level idempotency resumes.
COHORTS=(CCRCC COAD READ HCC IDC LYMPH_IDC LUNG PAAD PRAD SKCM)
NC=${#COHORTS[@]}          # 10
SEEDS=(1 2 3)
sidx=$(( SLURM_ARRAY_TASK_ID / NC ))
cidx=$(( SLURM_ARRAY_TASK_ID % NC ))
SEED=${SEEDS[$sidx]}
COHORT=${COHORTS[$cidx]}
echo "[morpho_hest] START $COHORT seed$SEED on $(hostname)"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_hest.py --version V3 --seed "$SEED" --cohort "$COHORT" \
  --source_dataroot ../dataset --embed_dataroot ../embed_dataroot \
  --feature_encoder uni_v1_official --save_root results_hest --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[morpho_hest] DONE $COHORT seed$SEED"
