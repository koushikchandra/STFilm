#!/bin/bash
#SBATCH --job-name=morph_pooled
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=morph_pooled_%a_%j.log
#SBATCH --array=0-2
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# MorphoST full model (components 111) on the cross-organ POOLED regime, 3 seeds.
# Fills the MorphoST-POOLED cells of Table 4 (tab:crossorgan). Writes tag POOLED_C111_seedX
# alongside the LOOO factorial results; fold-level idempotent so requeue is safe.
SEEDS=(1 2 3)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --regime POOLED --version V3 --components 111 --seed "$S" \
  --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot --save_root results_corrected_factorial --device cuda
