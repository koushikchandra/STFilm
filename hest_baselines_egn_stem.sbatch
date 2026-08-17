#!/bin/bash
#SBATCH --job-name=hest_egnstem
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=hest_egnstem_%a_%j.log
#SBATCH --array=0-59

# HEST-1k intra-cohort (per-cohort k-fold) for EGN and STEM.
# 2 models x 10 cohorts x 3 seeds = 60 tasks.
# Writes to results_corrected_hest_baselines/ (same root as other baselines).

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[hest_egnstem] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; print('[hest_egnstem] cuda:', torch.cuda.is_available())"

MODELS=(egn stem)
COHORTS=(CCRCC COAD READ HCC IDC LYMPH_IDC LUNG PAAD PRAD SKCM)
SEEDS=(1 2 3)

N_COHORTS=${#COHORTS[@]}; N_SEEDS=${#SEEDS[@]}
IDX=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$((IDX / (N_COHORTS * N_SEEDS)))]}
REM=$((IDX % (N_COHORTS * N_SEEDS)))
COHORT=${COHORTS[$((REM / N_SEEDS))]}
SEED=${SEEDS[$((REM % N_SEEDS))]}

TAG="${COHORT}_${MODEL}_seed${SEED}"
ROOT=results_corrected_hest_baselines
SAVE_DIR="$ROOT/$TAG"
NEED=$(ls dataset/"$COHORT"/splits/train_*.csv 2>/dev/null | wc -l)

HAVE=$(find "$SAVE_DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ] && [ "$NEED" -gt 0 ]; then
    echo "[hest_egnstem] SKIP $TAG ($HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[hest_egnstem] RESUME $TAG ($HAVE/$NEED)"

echo "[hest_egnstem] $(date +%H:%M:%S) START $TAG (need=$NEED)"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python hest_baselines.py \
  --model "$MODEL" --seed "$SEED" --cohort "$COHORT" \
  --feature_encoder uni_v1_official \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs $([ "$MODEL" = "stem" ] && echo 500 || echo 100) \
  --n_steps $([ "$MODEL" = "stem" ] && echo 1000 || echo 50) \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[hest_egnstem] $(date +%H:%M:%S) END $TAG"
