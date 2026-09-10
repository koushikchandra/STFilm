#!/bin/bash
#SBATCH --job-name=morph_stin
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=morph_sti_%a_%j.log
#SBATCH --array=0-5
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
REG=(LOOO POOLED); SEEDS=(1 2 3)
R=${REG[$(( SLURM_ARRAY_TASK_ID / 3 ))]}; S=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
echo "[sti] $R seed$S on $(hostname)"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --version V3 --organ_set stimage --regime "$R" --seed "$S" \
  --splits_root ../stimage_splits --source_dataroot ../dataset_stimage \
  --embed_dataroot ../embed_dataroot_stimage --save_root results_morphost_stimage --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
