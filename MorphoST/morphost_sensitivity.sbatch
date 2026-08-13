#!/bin/bash
#SBATCH --job-name=morph_sensitivity
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=morph_sensitivity_%a_%j.log
#SBATCH --array=0-38
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
SEEDS=(1 2 3); S=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}; CFG=$((SLURM_ARRAY_TASK_ID / 3))
ARGS=""; TAG=""
case "$CFG" in
  0) TAG=k4; ARGS="--k 4";;
  1) TAG=k8; ARGS="--k 8";;
  2) TAG=k16; ARGS="--k 16";;
  3) TAG=k32; ARGS="--k 32";;
  4) TAG=l2; ARGS="--n_layers 2";;
  5) TAG=l4; ARGS="--n_layers 4";;
  6) TAG=l6; ARGS="--n_layers 6";;
  7) TAG=d128; ARGS="--dim 128";;
  8) TAG=d256; ARGS="--dim 256";;
  9) TAG=d512; ARGS="--dim 512";;
  10) TAG=rbf8; ARGS="--num_rbf 8";;
  11) TAG=rbf16; ARGS="--num_rbf 16";;
  12) TAG=rbf32; ARGS="--num_rbf 32";;
esac
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --regime LOOO --version V3 --seed "$S" --exp_code "LOOO_sens_${TAG}_seed${S}" \
  --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot --save_root results_corrected_sensitivity --device cuda $ARGS
