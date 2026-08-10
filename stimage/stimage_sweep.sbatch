#!/bin/bash
#SBATCH --job-name=stimage_sweep
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=stimage_sweep_s%a_%j.log
#SBATCH --array=0-8

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[sti] node=$(hostname) job=$SLURM_JOB_ID shard=$SLURM_ARRAY_TASK_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[sti] cuda available:', torch.cuda.is_available())" 2>&1 | grep -i "cuda available"

# 18 configs (3 films x 2 regimes x 3 seeds), 9 array shards -> 2 configs per GPU.
bash stimage/run_stimage.sh 0 "$SLURM_ARRAY_TASK_ID" 9
echo "[sti] DONE shard=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID"
