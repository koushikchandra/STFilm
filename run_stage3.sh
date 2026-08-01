#!/bin/bash
# Stage 3 cross-organ sweep: V1 (film=context) vs V2 (film=meta, organ metadata)
# over POOLED and LOOO regimes, seeds 1-3. Matched seeds/splits per pair.
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

for SEED in 1 2 3; do
  for REGIME in POOLED LOOO; do
    for FILM in context meta; do
      echo "[stage3] $(date +%H:%M:%S) START regime=$REGIME film=$FILM seed=$SEED"
      PYTHONPATH=STFlow python train_cross_organ.py \
        --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --device 0 \
        --exp_code "${REGIME}_${FILM}_seed${SEED}" \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[stage3] $(date +%H:%M:%S) END   regime=$REGIME film=$FILM seed=$SEED"
    done
  done
done
echo "STAGE3_ALL_DONE"
