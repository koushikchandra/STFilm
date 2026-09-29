#!/bin/bash
#SBATCH --job-name=morphost_sti_intra
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=morphost_sti_intra_%a_%j.log
#SBATCH --array=0-2

# 3 seeds; each seed trains MorphoST V3 on STImage-1K4M INTRA (per-organ 5-fold).
# Writes to MorphoST/results_corrected_stimage_intra/.

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

SEED=$((SLURM_ARRAY_TASK_ID + 1))
echo "[mi] node=$(hostname) job=$SLURM_JOB_ID seed=$SEED gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[mi] cuda:', torch.cuda.is_available())"

TAG="INTRA_V3_seed${SEED}"
ROOT=MorphoST/results_corrected_stimage_intra
NEED=40  # 8 organs x 5 folds

HAVE=$(find "$ROOT/$TAG" -name "fold_*_results.json" 2>/dev/null | wc -l)
if [ "$HAVE" -eq "$NEED" ]; then
    echo "[mi] SKIP $TAG (complete $HAVE/$NEED)"; exit 0
fi
[ "$HAVE" -gt 0 ] && echo "[mi] RESUME $TAG ($HAVE/$NEED)"

echo "[mi] $(date +%H:%M:%S) START $TAG"
PYTHONPATH=STFlow OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 \
  python MorphoST/train_hest.py \
  --regime INTRA --seed "$SEED" --version V3 \
  --components 111 \
  --feature_encoder uni_v1_official \
  --splits_root stimage_splits \
  --source_dataroot dataset_stimage \
  --embed_dataroot embed_dataroot_stimage \
  --save_root "$ROOT" \
  --val_fraction 0.15 \
  --epochs 100 --patience 20 \
  --device 0 \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24"
echo "[mi] $(date +%H:%M:%S) END $TAG seed=$SEED"
