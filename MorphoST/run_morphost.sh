#!/bin/bash
# Run one MorphoST config. Usage: <version V0..V5> <regime LOOO|POOLED> <seed>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
VERSION="$1"; REGIME="${2:-LOOO}"; SEED="${3:-1}"
echo "[morphost] $(date +%H:%M:%S) START $VERSION $REGIME seed$SEED"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --regime "$REGIME" --version "$VERSION" --seed "$SEED" \
  --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot --feature_encoder uni_v1_official \
  --save_root results_morphost --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[morphost] $(date +%H:%M:%S) END $VERSION $REGIME seed$SEED"
