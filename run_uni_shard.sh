#!/bin/bash
# One shard of the UNI sweep, pinned to a single GPU.
# Usage: run_uni_shard.sh <device> <shard_idx> <n_shards>
# Enumerates all (seed,regime,film) configs in a fixed order and only runs those
# whose position mod n_shards == shard_idx. Skips already-complete configs and
# aborts the shard cleanly if its GPU dies (no crash-loop).
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_cross_organ_uni8
i=0
for SEED in 1 2 3; do
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for FILM in context meta desc; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      DIR="$ROOT/${REGIME}_${FILM}_seed${SEED}"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      if [ "$HAVE" -eq "$NEED" ]; then
        echo "[uni-g$DEV] SKIP ${REGIME}_${FILM}_seed${SEED} (complete: $HAVE/$NEED)"
        continue
      fi
      [ "$HAVE" -gt 0 ] && echo "[uni-g$DEV] RESUME $DIR ($HAVE/$NEED folds done, trainer skips them)"
      if ! PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV python gpu_ready.py >/dev/null 2>&1; then
        echo "[uni-g$DEV] $(date +%H:%M:%S) ABORT: GPU $DEV not ready before ${REGIME}_${FILM}_seed${SEED}"
        exit 3
      fi
      echo "[uni-g$DEV] $(date +%H:%M:%S) START regime=$REGIME film=$FILM seed=$SEED"
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
      echo "[uni-g$DEV] $(date +%H:%M:%S) END   regime=$REGIME film=$FILM seed=$SEED"
    done
  done
done
echo "UNI_SHARD${SHARD}_DONE"
