#!/bin/bash
#SBATCH --job-name=bigcap
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=bigcap_s%a_%j.log
#SBATCH --array=0-2
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
bash run_bigcap.sh 0 "$SLURM_ARRAY_TASK_ID" 3
echo "[bigcap] DONE shard=$SLURM_ARRAY_TASK_ID"
