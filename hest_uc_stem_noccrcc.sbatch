#!/bin/bash
#SBATCH --job-name=uc_stem_nc
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=uc_stem_nc_%a_%j.log
#SBATCH --array=0-17

# STEM UNI+CONCH — 6 cohorts (excl. CCRCC) x 3 seeds = 18 tasks.
# CCRCC is covered by job 12029074 which is still running.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[uc_stem_nc] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: No CUDA"; exit 1; }

COHORTS=(COAD IDC LUNG PAAD PRAD SKCM)
SEEDS=(1 2 3)
NC=${#COHORTS[@]}

COHORT=${COHORTS[$((SLURM_ARRAY_TASK_ID % NC))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / NC))]}

ROOT=results_corrected_hest_uniconch_baselines
SAVE_DIR="$ROOT/${COHORT}_stem_seed${SEED}"

if [ -f "$SAVE_DIR/results_kfold.json" ]; then
    echo "[uc_stem_nc] SKIP $COHORT seed$SEED already done"; exit 0
fi

echo "[uc_stem_nc] $(date +%H:%M:%S) START $COHORT seed$SEED (uni_conch 1536-dim, epochs=100 patience=20)"
PYTHONPATH=STFlow OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python hest_baselines.py \
  --model stem --seed "$SEED" --cohort "$COHORT" \
  --feature_encoder uni_conch \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --epochs 100 --patience 20 --n_steps 1000 \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[uc_stem_nc] $(date +%H:%M:%S) END $COHORT seed$SEED"
