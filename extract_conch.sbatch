#!/bin/bash
#SBATCH --job-name=conch_extract
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=conch_extract_%a_%j.log
#SBATCH --array=0-9

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[conch_extract] node=$(hostname) task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: No CUDA"; exit 1; }

COHORTS=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM)
COHORT=${COHORTS[$SLURM_ARRAY_TASK_ID]}

echo "[conch_extract] $(date +%H:%M:%S) extracting CONCH for $COHORT"

PYTHONPATH=STFlow OMP_NUM_THREADS=6 \
python STFlow/stflow/app/hest/benchmark.py \
    --datasets "$COHORT" \
    --encoders conch_v1_official \
    --weights_root weights_root \
    --source_dataroot dataset \
    --embed_dataroot embed_dataroot \
    --batch_size 256

echo "[conch_extract] $(date +%H:%M:%S) DONE $COHORT"
