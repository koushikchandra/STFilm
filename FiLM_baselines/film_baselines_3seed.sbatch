#!/bin/bash
#SBATCH --job-name=film_3seed
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=film_3seed_%a_%j.log
#SBATCH --array=0-59
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/FiLM_baselines
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# "Is FiLM general?" at 3 seeds. 6 regression backbones x {none,desc,local} + bleep x {none,desc}.
# NEW backbones: deepspace (weak MLP), mlpprobe (strong residual-MLP UNI probe). Both context-poor.
# Seed1 for the 5 original combos is already on disk -> fold-level idempotency skips it instantly.
COMBOS=(
 "histogene none" "histogene desc" "histogene local"
 "hist2st none"   "hist2st desc"   "hist2st local"
 "triplex none"   "triplex desc"   "triplex local"
 "stnet none"     "stnet desc"     "stnet local"
 "deepspace none" "deepspace desc" "deepspace local"
 "mlpprobe none"  "mlpprobe desc"  "mlpprobe local"
 "bleep none"     "bleep desc"
)
NCOMBO=${#COMBOS[@]}          # 20
SEEDS=(1 2 3)
sidx=$(( SLURM_ARRAY_TASK_ID / NCOMBO ))
cidx=$(( SLURM_ARRAY_TASK_ID % NCOMBO ))
SEED=${SEEDS[$sidx]}
set -- ${COMBOS[$cidx]}
MODEL=$1; FILM=$2
echo "[film_3seed] START $MODEL-$FILM seed$SEED"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python film_baselines.py --model "$MODEL" --film "$FILM" --regime LOOO --seed "$SEED" \
  --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot --feature_encoder uni_v1_official \
  --save_root results_film_baselines --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[film_3seed] DONE $MODEL-$FILM seed$SEED"
