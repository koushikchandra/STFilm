#!/bin/bash
#SBATCH --job-name=intra_g10_bl
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=intra_g10_bl_%a_%j.log
#SBATCH --array=0-11

# Intra-cohort baselines (ST-Net / Hist2ST / HyperST / BLEEP) with the 10-gene panel x 3 seeds.
# 12 tasks: model-minor (4) x seed-major (3). Each task loops over 7 cohorts.
# Same config as Table 2 (tab:main): UNI+CONCH (1536-dim), 100 epochs, patience 20.
# Save root: results_intra_g10_baselines/. STEM runs separately in intra_g10_stem.sbatch.
# Fold-level idempotency: already-complete folds skipped on resume/requeue.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[intra_g10_bl] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; print('[intra_g10_bl] cuda:', torch.cuda.is_available())"

MODELS=(stnet hist2st hyperst bleep)
GENE_LIST=var_10genes.json
SEEDS=(1 2 3)
COHORTS=(CCRCC COAD IDC LUNG PAAD PRAD SKCM)

NM=${#MODELS[@]}   # 4
midx=$(( SLURM_ARRAY_TASK_ID % NM ))
sidx=$(( SLURM_ARRAY_TASK_ID / NM ))

MODEL=${MODELS[$midx]}
SEED=${SEEDS[$sidx]}
SAVE_ROOT="results_intra_g10_baselines"

echo "[intra_g10_bl] model=$MODEL gene_list=$GENE_LIST seed=$SEED"

for COHORT in "${COHORTS[@]}"; do
    NEED=$(ls dataset/"$COHORT"/splits/train_*.csv 2>/dev/null | wc -l)
    HAVE=$(find "$SAVE_ROOT/${COHORT}_${MODEL}_seed${SEED}" -name "fold_*_results.json" 2>/dev/null | wc -l)
    if [ "$HAVE" -ge "$NEED" ] && [ "$NEED" -gt 0 ]; then
        echo "[intra_g10_bl] SKIP $COHORT $MODEL (complete $HAVE/$NEED)"; continue
    fi
    [ "$HAVE" -gt 0 ] && echo "[intra_g10_bl] RESUME $COHORT $MODEL ($HAVE/$NEED folds done)"
    echo "[intra_g10_bl] $(date +%H:%M:%S) START $COHORT $MODEL"
    PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python hest_baselines.py \
      --model "$MODEL" --seed "$SEED" --cohort "$COHORT" \
      --source_dataroot dataset --embed_dataroot embed_dataroot \
      --feature_encoder uni_conch --gene_list "$GENE_LIST" \
      --save_root "$SAVE_ROOT" \
      --val_fraction 0.15 --epochs 100 --patience 20 --device 0 \
      2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
    echo "[intra_g10_bl] $(date +%H:%M:%S) END $COHORT $MODEL"
done

echo "[intra_g10_bl] DONE model=$MODEL seed=$SEED"
