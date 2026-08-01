#!/bin/bash
# UNI-encoder cross-organ sweep: V1(context) vs V2(meta) vs V3(desc),
# POOLED + LOOO, seeds 1-3. Matched seeds/splits/panels to the resnet run.
# Ordered LOOO-first per seed so the key transfer numbers land earliest.
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

for SEED in 1 2 3; do
  for REGIME in LOOO POOLED; do
    for FILM in context meta desc; do
      echo "[uni] $(date +%H:%M:%S) START regime=$REGIME film=$FILM seed=$SEED"
      PYTHONPATH=STFlow python train_cross_organ.py \
        --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --feature_encoder uni_v1_official \
        --save_root results_cross_organ_uni \
        --device 0 \
        --exp_code "${REGIME}_${FILM}_seed${SEED}" \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[uni] $(date +%H:%M:%S) END   regime=$REGIME film=$FILM seed=$SEED"
    done
  done
done
echo "UNI_ALL_DONE"
