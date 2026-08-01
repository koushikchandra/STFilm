#!/bin/bash
# Final apple-to-apple: V3 (desc) vs V0 (none), identical tuned HPs (lr=1e-3, default capacity),
# on the real 8-organ UNI splits. 2 films x 2 regimes x 3 seeds = 12 configs.
# Usage: run_uni_final.sh <device> <shard_idx> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_final_uni8
SPLITS=cross_organ_splits8
LR=1e-3
i=0
for FILM in none desc; do
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for SEED in 1 2 3; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      TAG="${REGIME}_${FILM}_seed${SEED}"
      DIR="$ROOT/$TAG"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      if [ "$HAVE" -eq "$NEED" ]; then
        echo "[fin-g$DEV] SKIP $TAG (complete: $HAVE/$NEED)"; continue
      fi
      [ "$HAVE" -gt 0 ] && echo "[fin-g$DEV] RESUME $TAG ($HAVE/$NEED)"
      if ! PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV python gpu_ready.py >/dev/null 2>&1; then
        echo "[fin-g$DEV] $(date +%H:%M:%S) ABORT: GPU $DEV not ready before $TAG"; exit 3
      fi
      echo "[fin-g$DEV] $(date +%H:%M:%S) START $TAG"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
        OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
        python train_cross_organ.py \
        --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --feature_encoder uni_v1_official \
        --splits_root $SPLITS \
        --save_root "$ROOT" \
        --exp_code "$TAG" \
        --lr $LR \
        --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[fin-g$DEV] $(date +%H:%M:%S) END   $TAG"
    done
  done
done
echo "UNI_FINAL_SHARD${SHARD}_DONE"
