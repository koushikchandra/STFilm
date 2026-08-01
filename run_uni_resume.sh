#!/bin/bash
# Resume the UNI sweep, skipping configs that already have the full fold count.
# LOOO -> 6 folds expected, POOLED -> 5. Partial dirs are wiped and re-run.
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

ROOT=results_cross_organ_uni
for SEED in 1 2 3; do
  for REGIME in LOOO POOLED; do
    NEED=6; [ "$REGIME" = "POOLED" ] && NEED=5
    for FILM in context meta desc; do
      DIR="$ROOT/${REGIME}_${FILM}_seed${SEED}"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      if [ "$HAVE" -eq "$NEED" ]; then
        echo "[uni] SKIP ${REGIME}_${FILM}_seed${SEED} (complete: $HAVE/$NEED)"
        continue
      fi
      [ "$HAVE" -gt 0 ] && { echo "[uni] WIPE partial $DIR ($HAVE/$NEED)"; rm -rf "$DIR"; }
      # Guard: if the GPU is not usable, abort the whole sweep instead of
      # crash-looping through every remaining config in seconds.
      if ! PYTHONPATH=STFlow python gpu_ready.py >/dev/null 2>&1; then
        echo "[uni] $(date +%H:%M:%S) ABORT: GPU not ready before ${REGIME}_${FILM}_seed${SEED}"
        exit 3
      fi
      echo "[uni] $(date +%H:%M:%S) START regime=$REGIME film=$FILM seed=$SEED"
      PYTHONPATH=STFlow python train_cross_organ.py \
        --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --feature_encoder uni_v1_official \
        --save_root "$ROOT" \
        --device 0 \
        --exp_code "${REGIME}_${FILM}_seed${SEED}" \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[uni] $(date +%H:%M:%S) END   regime=$REGIME film=$FILM seed=$SEED"
    done
  done
done
echo "UNI_ALL_DONE"
