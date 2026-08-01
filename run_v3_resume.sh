#!/bin/bash
# Resume the interrupted V3 sweep: only the missing seed-3 configs.
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

for REGIME in POOLED LOOO; do
  echo "[v3] $(date +%H:%M:%S) START regime=$REGIME film=desc seed=3"
  PYTHONPATH=STFlow python train_cross_organ.py \
    --regime "$REGIME" --film desc --seed 3 \
    --device 0 \
    --exp_code "${REGIME}_desc_seed3" \
    2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
  echo "[v3] $(date +%H:%M:%S) END   regime=$REGIME film=desc seed=3"
done
echo "V3_RESUME_DONE"
