#!/bin/bash
#SBATCH --job-name=morph_geometry
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=morph_geometry_%a_%j.log
#SBATCH --array=0-8
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
SEEDS=(1 2 3); S=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}; CFG=$((SLURM_ARRAY_TASK_ID / 3))
case "$CFG" in
  0) TAG=distance_rbf; EXTRA="";;
  1) TAG=distance_nobias; EXTRA="--no-use_distance_bias";;
  2) TAG=absolute_xy; EXTRA="--coordinate_mode absolute";;
esac
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --regime LOOO --version V3 --seed "$S" --exp_code "LOOO_geom_${TAG}_seed${S}" \
  --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot --save_root results_corrected_geometry --device cuda $EXTRA
