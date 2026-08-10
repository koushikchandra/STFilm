#!/bin/bash
#SBATCH --job-name=stimage_build
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=stimage_build_%j.log

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[build] node=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[build] cuda available:', torch.cuda.is_available())"

RAW=/lustre/hdd/LAS/weile-lab/howlader/STFilm/stimage/raw
python -u stimage/build_stimage.py \
    --manifest stimage/manifest.csv --raw_dir "$RAW" \
    --embed_root embed_dataroot_stimage --src_root dataset_stimage \
    --weights_root weights_root --batch 256 --context_scale 4.0 --device cuda
echo "[build] DONE job=$SLURM_JOB_ID"
