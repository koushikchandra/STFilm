#!/bin/bash
#SBATCH --job-name=intra_g10_stem
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=intra_g10_stem_%a_%j.log
#SBATCH --array=0-20

# STEM intra-cohort with the 10-gene panel x 3 seeds x 7 cohorts = 21 tasks.
# One (seed, cohort) per task because STEM (500 epochs, T=1000 DDPM) is slow.
# Layout: seed-major (7 each), cohort-minor (1 each). UNI+CONCH (1536-dim).
# Save root: results_intra_g10_baselines/ (shared with other baselines).

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[intra_g10_stem] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; print('[intra_g10_stem] cuda:', torch.cuda.is_available())"

GENE_LIST=var_10genes.json
SEEDS=(1 2 3)
COHORTS=(CCRCC COAD IDC LUNG PAAD PRAD SKCM)
N_COHORTS=${#COHORTS[@]}   # 7

sidx=$(( SLURM_ARRAY_TASK_ID / N_COHORTS ))
cidx=$(( SLURM_ARRAY_TASK_ID % N_COHORTS ))
SEED=${SEEDS[$sidx]}
COHORT=${COHORTS[$cidx]}
SAVE_ROOT="results_intra_g10_baselines"

TAG="${COHORT}_stem_seed${SEED}"
NEED=$(ls dataset/"$COHORT"/splits/train_*.csv 2>/dev/null | wc -l)
HAVE=$(find "$SAVE_ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)

echo "[intra_g10_stem] gene_list=$GENE_LIST seed=$SEED cohort=$COHORT"

if [ "$HAVE" -ge "$NEED" ] && [ "$NEED" -gt 0 ]; then
    echo "[intra_g10_stem] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[intra_g10_stem] RESUME $TAG ($HAVE/$NEED folds done)"

echo "[intra_g10_stem] $(date +%H:%M:%S) START $TAG (need=$NEED)"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python hest_baselines.py \
  --model stem --seed "$SEED" --cohort "$COHORT" \
  --source_dataroot dataset --embed_dataroot embed_dataroot \
  --feature_encoder uni_conch --gene_list "$GENE_LIST" \
  --save_root "$SAVE_ROOT" \
  --val_fraction 0.15 --epochs 500 --patience 20 --n_steps 1000 --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[intra_g10_stem] $(date +%H:%M:%S) END $TAG"
