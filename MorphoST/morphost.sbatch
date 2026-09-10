#!/bin/bash
#SBATCH --job-name=morphost
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=morphost_v%a_%j.log
#SBATCH --array=0-5
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
VERSIONS=(V0 V1 V2 V3 V4 V5)
V=${VERSIONS[$SLURM_ARRAY_TASK_ID]}
# Go/no-go: full ablation on HEST LOOO, seed 1 (one version per array task).
bash run_morphost.sh "$V" LOOO 1
echo "[morphost] DONE $V"
