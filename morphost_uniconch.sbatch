#!/bin/bash
#SBATCH --job-name=morphost_uc
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=nova21-gpu-10
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=morphost_uc_%a_%j.log
#SBATCH --array=0-5

# MorphoST (context FiLM) with UNI+CONCH (1536-dim) encoder.
# 2 regimes x 3 seeds = 6 tasks.
# Run AFTER concat_uni_conch.py has written all uni_conch h5 files.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[morphost_uc] node=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || { echo "ERROR: No CUDA"; exit 1; }

REGIMES=(LOOO POOLED)
SEEDS=(1 2 3)

IDX=$SLURM_ARRAY_TASK_ID
REGIME=${REGIMES[$((IDX / 3))]}
SEED=${SEEDS[$((IDX % 3))]}

NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
TAG="${REGIME}_context_uniconch_seed${SEED}"
ROOT=results_uniconch

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -ge "$NEED" ]; then
    echo "[morphost_uc] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[morphost_uc] RESUME $TAG ($HAVE/$NEED)"

echo "[morphost_uc] $(date +%H:%M:%S) START $TAG (feature=uni_conch 1536-dim)"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_cross_organ.py \
  --regime "$REGIME" --film context --seed "$SEED" \
  --feature_encoder uni_conch \
  --splits_root cross_organ_splits8 \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root "$ROOT" --exp_code "$TAG" \
  --n_layers 8 --hidden_dim 384 --pairwise_hidden_dim 384 --batch_size 1 \
  --lr 1e-3 --epochs 100 --dump_preds --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
echo "[morphost_uc] $(date +%H:%M:%S) END $TAG"
