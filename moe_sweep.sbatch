#!/bin/bash
#SBATCH --job-name=uni8_moe
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=uni8_moe_s%a_%j.log
#SBATCH --array=0-5

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[moe] node=$(hostname) job=$SLURM_JOB_ID shard=$SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[moe] cuda available:', torch.cuda.is_available())" 2>&1 | grep -i "cuda available"

# 6 configs (2 regimes x 3 seeds), 6 array shards -> 1 config per GPU.
bash run_uni_moe.sh 0 "$SLURM_ARRAY_TASK_ID" 6
echo "[moe] DONE shard=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID"
