#!/bin/bash
#SBATCH --job-name=gp_extract2
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=gp_extract2_%j.log
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
export HF_TOKEN="$(cat .hf/token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo "[gpx2] node=$(hostname) $(date)"
PYTHONPATH=STFlow python extract_gigapath_only.py --encoder gigapath \
  --weights_root weights_root --source_dataroot dataset --embed_dataroot embed_dataroot --batch_size 64
echo "[gpx2] DONE rc=$? $(date)"
