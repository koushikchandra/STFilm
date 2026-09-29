#!/bin/bash
#SBATCH --job-name=intra_g10_morph
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=intra_g10_morph_%a_%j.log
#SBATCH --array=0-2

# MorphoST intra-cohort with the 10-gene HVG panel (var_10genes.json) x 3 seeds.
# 3 tasks: one per seed. Each task loops over 7 cohorts.
# Same config as Table 2 (tab:main): UNI+CONCH (1536-dim), dim256/4L/4H/k8/rbf16,
# lr1e-3, corr_weight0.5, 100 epochs, patience 20 -- only the gene panel differs.
# Save root: MorphoST/results_intra_g10_morphost/
# Fold-level idempotency: already-complete folds are skipped on resume.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[intra_g10_morph] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; print('[intra_g10_morph] cuda:', torch.cuda.is_available())"

GENE_LIST=var_10genes.json
SEEDS=(1 2 3)
COHORTS=(CCRCC COAD IDC LUNG PAAD PRAD SKCM)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SAVE_ROOT="results_intra_g10_morphost"

echo "[intra_g10_morph] gene_list=$GENE_LIST seed=$SEED save_root=$SAVE_ROOT"

for COHORT in "${COHORTS[@]}"; do
    NEED=$(ls ../dataset/"$COHORT"/splits/train_*.csv 2>/dev/null | wc -l)
    HAVE=$(find "$SAVE_ROOT/${COHORT}_V3_seed${SEED}" -name "fold_*_results.json" 2>/dev/null | wc -l)
    if [ "$HAVE" -ge "$NEED" ] && [ "$NEED" -gt 0 ]; then
        echo "[intra_g10_morph] SKIP $COHORT (complete $HAVE/$NEED)"; continue
    fi
    [ "$HAVE" -gt 0 ] && echo "[intra_g10_morph] RESUME $COHORT ($HAVE/$NEED folds done)"
    echo "[intra_g10_morph] $(date +%H:%M:%S) START $COHORT"
    PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python train_hest.py \
      --version V3 --seed "$SEED" --cohort "$COHORT" \
      --source_dataroot ../dataset --embed_dataroot ../embed_dataroot \
      --feature_encoder uni_conch --gene_list "$GENE_LIST" \
      --save_root "$SAVE_ROOT" \
      --dim 256 --n_layers 4 --n_heads 4 --k 8 --num_rbf 16 \
      --lr 1e-3 --weight_decay 0.01 --epochs 100 --patience 20 \
      --corr_weight 0.5 --max_spots 3000 --val_fraction 0.15 \
      2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
    echo "[intra_g10_morph] $(date +%H:%M:%S) END $COHORT"
done

echo "[intra_g10_morph] DONE seed=$SEED"
