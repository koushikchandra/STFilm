#!/bin/bash
#SBATCH --job-name=film_base
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=film_base_%a_%j.log
#SBATCH --array=0-13
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/FiLM_baselines
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# All model x film combos, HEST LOOO seed 1. 4 regression models x {none,desc,local} + bleep x {none,desc}.
COMBOS=(
 "histogene none" "histogene desc" "histogene local"
 "hist2st none"   "hist2st desc"   "hist2st local"
 "triplex none"   "triplex desc"   "triplex local"
 "stnet none"     "stnet desc"     "stnet local"
 "bleep none"     "bleep desc"
)
set -- ${COMBOS[$SLURM_ARRAY_TASK_ID]}
MODEL=$1; FILM=$2
echo "[film_base] START $MODEL-$FILM"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python film_baselines.py --model "$MODEL" --film "$FILM" --regime LOOO --seed 1 \
  --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot --feature_encoder uni_v1_official \
  --save_root results_film_baselines --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[film_base] DONE $MODEL-$FILM"
