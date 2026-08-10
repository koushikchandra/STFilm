#!/bin/bash
#SBATCH --job-name=gp_extract
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=gp_extract_%j.log
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
export HF_TOKEN="$(cat .hf/token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo "[gpx] node=$(hostname) job=$SLURM_JOB_ID $(date)"
PYTHONPATH=STFlow python STFlow/stflow/app/hest/benchmark.py \
  --datasets CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM \
  --encoders gigapath --method ridge \
  --weights_root weights_root --source_dataroot dataset --embed_dataroot embed_dataroot \
  --precision fp32 --batch_size 64
echo "[gpx] DONE rc=$? $(date)"
