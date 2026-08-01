#!/bin/bash
# V4 hybrid (FiLM/adaLN + cross-attention to K prototype tokens) on the real 8-organ UNI splits.
# Identical HPs to results_final_uni8 (lr=1e-3, default capacity) so it slots into the same
# comparison table. 1 film x 2 regimes x 3 seeds = 6 configs.
# Usage: run_uni_hybrid.sh <device> <shard_idx> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_hybrid_uni8
SPLITS=cross_organ_splits8
LR=1e-3
NPROTO=8
i=0
for FILM in hybrid; do
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for SEED in 1 2 3; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      TAG="${REGIME}_${FILM}_seed${SEED}"
      DIR="$ROOT/$TAG"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      if [ "$HAVE" -eq "$NEED" ]; then
        echo "[hyb-g$DEV] SKIP $TAG (complete: $HAVE/$NEED)"; continue
      fi
      [ "$HAVE" -gt 0 ] && echo "[hyb-g$DEV] RESUME $TAG ($HAVE/$NEED)"
      if ! PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV python gpu_ready.py >/dev/null 2>&1; then
        echo "[hyb-g$DEV] $(date +%H:%M:%S) ABORT: GPU $DEV not ready before $TAG"; exit 3
      fi
      echo "[hyb-g$DEV] $(date +%H:%M:%S) START $TAG"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
        OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
        python train_cross_organ.py \
        --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --feature_encoder uni_v1_official \
        --splits_root $SPLITS \
        --save_root "$ROOT" \
        --exp_code "$TAG" \
        --n_proto $NPROTO \
        --lr $LR \
        --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[hyb-g$DEV] $(date +%H:%M:%S) END   $TAG"
    done
  done
done
echo "UNI_HYBRID_SHARD${SHARD}_DONE"
