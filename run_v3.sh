#!/bin/bash
# V3 (film=desc) cross-organ sweep over POOLED and LOOO, seeds 1-3.
# V1 (context) and V2 (meta) already exist on disk; this fills the V3 column.
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

for SEED in 1 2 3; do
  for REGIME in POOLED LOOO; do
    echo "[v3] $(date +%H:%M:%S) START regime=$REGIME film=desc seed=$SEED"
    PYTHONPATH=STFlow python train_cross_organ.py \
      --regime "$REGIME" --film desc --seed "$SEED" \
      --device 0 \
      --exp_code "${REGIME}_desc_seed${SEED}" \
      2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
    echo "[v3] $(date +%H:%M:%S) END   regime=$REGIME film=desc seed=$SEED"
  done
done
echo "V3_ALL_DONE"
