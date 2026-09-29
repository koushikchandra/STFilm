#!/bin/bash
#SBATCH --job-name=morph_abl3n
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=morph_abl3_%a_%j.log
#SBATCH --array=0-9
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
VERS=(V0 V1 V2 V4 V5); SEEDS=(2 3)
V=${VERS[$(( SLURM_ARRAY_TASK_ID / 2 ))]}; S=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 2 ))]}
echo "[abl3] $V seed$S on $(hostname)"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --version "$V" --regime LOOO --seed "$S" \
  --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot --save_root results_morphost --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
