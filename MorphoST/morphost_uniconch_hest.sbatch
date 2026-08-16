#!/bin/bash
#SBATCH --job-name=morph_uc_hest
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=morph_uc_hest_%a_%j.log
#SBATCH --array=0-20

# MorphoST (V3, context) with UNI+CONCH (1536-dim) on HEST-1k per-cohort protocol.
# 7 cohorts x 3 seeds = 21 tasks.
# Run AFTER: (1) extract_conch.sbatch completes, (2) concat_uni_conch.py has run.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[morph_uc_hest] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: No CUDA"; exit 1; }

# 7 cohorts in Table 14 (HCC/LYMPH_IDC/READ excluded)
COHORTS=(CCRCC COAD IDC LUNG PAAD PRAD SKCM)
SEEDS=(1 2 3)
NC=${#COHORTS[@]}

COHORT=${COHORTS[$((SLURM_ARRAY_TASK_ID % NC))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / NC))]}

SAVE_DIR="results_corrected_hest_uniconch/${COHORT}_V3_seed${SEED}"
if [ -f "$SAVE_DIR/results_kfold.json" ]; then
    echo "[morph_uc_hest] SKIP $COHORT seed$SEED"; exit 0
fi

echo "[morph_uc_hest] $(date +%H:%M:%S) START $COHORT seed$SEED (uni_conch 1536-dim)"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_hest.py \
  --version V3 --seed "$SEED" --cohort "$COHORT" \
  --feature_encoder uni_conch \
  --corr_weight 0.5 \
  --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot \
  --save_root "results_corrected_hest_uniconch" \
  --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[morph_uc_hest] $(date +%H:%M:%S) END $COHORT seed$SEED"
