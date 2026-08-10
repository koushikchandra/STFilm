#!/bin/bash
# V5 MoE: morphology-routed mixture-of-experts (film=moe) on the real 8-organ UNI splits.
# Each spot routes over E expert MLPs on the invariant scalar stream (equivariance preserved);
# router optionally sees mean-pooled slide prototypes. Dense soft routing by default
# (moe_top_k=0) to avoid dead experts on small HEST cohorts. Same HPs as results_final_uni8
# (lr=1e-3) so it slots into the same comparison table. 1 film x 2 regimes x 3 seeds = 6 configs.
# Usage: run_uni_moe.sh <device> <shard_idx> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_moe_uni8
SPLITS=cross_organ_splits8
LR=1e-3
NEXPERTS=4
NPROTO=8
i=0
for FILM in moe; do
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for SEED in 1 2 3; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      TAG="${REGIME}_${FILM}_seed${SEED}"
      DIR="$ROOT/$TAG"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      if [ "$HAVE" -eq "$NEED" ]; then
        echo "[moe-g$DEV] SKIP $TAG (complete: $HAVE/$NEED)"; continue
      fi
      [ "$HAVE" -gt 0 ] && echo "[moe-g$DEV] RESUME $TAG ($HAVE/$NEED)"
      if ! PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV python gpu_ready.py >/dev/null 2>&1; then
        echo "[moe-g$DEV] $(date +%H:%M:%S) ABORT: GPU $DEV not ready before $TAG"; exit 3
      fi
      echo "[moe-g$DEV] $(date +%H:%M:%S) START $TAG"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
        OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
        python train_cross_organ.py \
        --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --feature_encoder uni_v1_official \
        --splits_root $SPLITS \
        --save_root "$ROOT" \
        --exp_code "$TAG" \
        --n_experts $NEXPERTS --moe_top_k 0 --use_prototypes_in_router \
        --n_proto $NPROTO \
        --lr $LR \
        --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[moe-g$DEV] $(date +%H:%M:%S) END   $TAG"
    done
  done
done
echo "UNI_MOE_SHARD${SHARD}_DONE"
