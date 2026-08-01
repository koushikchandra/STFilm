#!/bin/bash
#SBATCH --job-name=uni8_resume
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=uni8_resume_%j.log

# Resume the 8-organ UNI sweep on a healthy GPU node.
# run_uni_shard.sh skips the 106 completed folds and finishes the remaining 11.
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

# Make the base conda env's python available in the batch shell.
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[resume] node=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
echo "[resume] fabricmanager=$(systemctl is-active nvidia-fabricmanager 2>/dev/null)"
python -c "import torch; print('[resume] cuda available:', torch.cuda.is_available())" 2>&1 | grep -i "cuda available"

bash run_uni_shard.sh 0 0 1
echo "[resume] DONE job=$SLURM_JOB_ID"
