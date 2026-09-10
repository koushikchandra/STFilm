#!/bin/bash
#SBATCH --job-name=recent_film
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=recent_film_%a_%j.log
#SBATCH --array=0-35
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/FiLM_baselines
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
MODELS=(genedml feast)
FILMS=(none desc local)
REGIMES=(LOOO POOLED)
SEEDS=(1 2 3)
idx=$SLURM_ARRAY_TASK_ID
seed=${SEEDS[$((idx % 3))]}; idx=$((idx / 3))
regime=${REGIMES[$((idx % 2))]}; idx=$((idx / 2))
film=${FILMS[$((idx % 3))]}; idx=$((idx / 3))
model=${MODELS[$idx]}
echo "START $model $film $regime seed$seed"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python recent_film_baselines.py --model "$model" --film "$film" --regime "$regime" --seed "$seed" \
 --splits_root ../cross_organ_splits8 --source_dataroot ../dataset --embed_dataroot ../embed_dataroot \
 --save_root results_recent_film --device 0
