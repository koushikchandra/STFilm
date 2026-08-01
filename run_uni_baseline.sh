#!/bin/bash
# True upstream-STFlow baseline (film=none, no per-layer adaLN) on the same
# 8-organ UNI splits as the headline sweep. 3 seeds x {LOOO,POOLED}.
# Skips already-complete configs; per-fold resume handled by the trainer.
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

DEV="${1:-0}"
ROOT=results_cross_organ_uni8
FILM=none
for SEED in 1 2 3; do
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    DIR="$ROOT/${REGIME}_${FILM}_seed${SEED}"
    HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
    if [ "$HAVE" -eq "$NEED" ]; then
      echo "[base-g$DEV] SKIP ${REGIME}_${FILM}_seed${SEED} (complete: $HAVE/$NEED)"
      continue
    fi
    [ "$HAVE" -gt 0 ] && echo "[base-g$DEV] RESUME $DIR ($HAVE/$NEED folds done)"
    if ! PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV python gpu_ready.py >/dev/null 2>&1; then
      echo "[base-g$DEV] $(date +%H:%M:%S) ABORT: GPU $DEV not ready before ${REGIME}_${FILM}_seed${SEED}"
      exit 3
    fi
    echo "[base-g$DEV] $(date +%H:%M:%S) START regime=$REGIME film=$FILM seed=$SEED"
    PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
      OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
      python train_cross_organ.py \
      --regime "$REGIME" --film "$FILM" --seed "$SEED" \
      --feature_encoder uni_v1_official \
      --splits_root cross_organ_splits8 \
      --save_root "$ROOT" \
      --device 0 \
      --exp_code "${REGIME}_${FILM}_seed${SEED}" \
      2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
    echo "[base-g$DEV] $(date +%H:%M:%S) END   regime=$REGIME film=$FILM seed=$SEED"
  done
done
echo "UNI_BASELINE_DONE"
