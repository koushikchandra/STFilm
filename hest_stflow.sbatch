#!/bin/bash
#SBATCH --job-name=hest_stflow
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2
#SBATCH --output=hest_stflow_%a_%j.log
#SBATCH --array=0-29
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# VANILLA STFlow (--film none) under its own per-cohort HEST protocol on the SAME UNI features, so it
# is a proper feature-matched column in the MorphoST-V3 Table-1 (not just the paper's number). Paper
# config: n_layers 4, hidden 128, S=5, ZINB, MSE, lr 5e-4. 10 cohorts x 3 seeds.
COHORTS=(CCRCC COAD READ HCC IDC LYMPH_IDC LUNG PAAD PRAD SKCM)
NC=${#COHORTS[@]}
SEEDS=(1 2 3)
sidx=$(( SLURM_ARRAY_TASK_ID / NC ))
cidx=$(( SLURM_ARRAY_TASK_ID % NC ))
SEED=${SEEDS[$sidx]}
COHORT=${COHORTS[$cidx]}
SAVE=results_hest_stflow/${COHORT}_STFlow_seed${SEED}
if [ -f "$SAVE/results_kfold.json" ]; then
  echo "[hest_stflow] SKIP $COHORT seed$SEED (results_kfold.json exists)"; exit 0
fi
echo "[hest_stflow] START $COHORT seed$SEED on $(hostname)"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python STFlow/stflow/app/flow/train.py \
  --datasets "$COHORT" --film none --feature_encoder uni_v1_official --seed "$SEED" \
  --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_dir "$SAVE" --batch_size 2 --n_layers 4 --hidden_dim 128 --n_sample_steps 5 \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[hest_stflow] DONE $COHORT seed$SEED"
