#!/bin/bash
#SBATCH --job-name=geneprog
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=geneprog_s%a_%j.log
#SBATCH --array=0-5
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
echo "[gp] node=$(hostname) job=$SLURM_JOB_ID shard=$SLURM_ARRAY_TASK_ID"
bash run_geneprog.sh 0 "$SLURM_ARRAY_TASK_ID" 6
echo "[gp] DONE shard=$SLURM_ARRAY_TASK_ID"
