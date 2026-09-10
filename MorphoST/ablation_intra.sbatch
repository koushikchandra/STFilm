#!/bin/bash
#SBATCH --job-name=morph_abl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=ablation_intra_%a_%j.log
#SBATCH --array=0-47

# Factorial intra-cohort ablation: 8 variants x 2 cohorts x 3 seeds = 48 tasks.
# Cohorts chosen as best MorphoST intra-cohort performers: SKCM, LUNG.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[morph_abl] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: No CUDA"; exit 1; }

VARIANTS=(000 001 010 011 100 101 110 111)
COHORTS=(SKCM LUNG)
SEEDS=(1 2 3)

NV=${#VARIANTS[@]}
NC=${#COHORTS[@]}

VARIANT=${VARIANTS[$((SLURM_ARRAY_TASK_ID % NV))]}
COHORT=${COHORTS[$(( (SLURM_ARRAY_TASK_ID / NV) % NC ))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / (NV * NC)))]}

SAVE_ROOT="results_ablation_intra/C${VARIANT}"
SAVE_DIR="${SAVE_ROOT}/${COHORT}_V3_seed${SEED}"
if [ -f "$SAVE_DIR/results_kfold.json" ]; then
    echo "[morph_abl] SKIP $COHORT C$VARIANT seed$SEED"; exit 0
fi

echo "[morph_abl] $(date +%H:%M:%S) START $COHORT C$VARIANT seed$SEED"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_hest.py \
  --version V3 --components "$VARIANT" \
  --seed "$SEED" --cohort "$COHORT" \
  --feature_encoder uni_v1_official \
  --corr_weight 0.5 \
  --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot \
  --save_root "$SAVE_ROOT" \
  --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[morph_abl] $(date +%H:%M:%S) END $COHORT C$VARIANT seed$SEED"
