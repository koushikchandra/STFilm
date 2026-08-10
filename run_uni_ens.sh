#!/bin/bash
# Phase 3: seed-ensembling. Re-run none/desc/local WITH --dump_preds so we can average
# best-epoch predictions across seeds (encoder-swap lever is blocked: only UNI weights local).
# Writes to a NEW root (results_ens_uni8) so results_final_uni8/results_local_uni8 stay untouched.
# 3 films x 2 regimes x 3 seeds = 18 configs. Usage: run_uni_ens.sh <device> <shard_idx> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_ens_uni8
SPLITS=cross_organ_splits8
LR=1e-3
i=0
for FILM in none desc local; do
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for SEED in 1 2 3; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      TAG="${REGIME}_${FILM}_seed${SEED}"
      DIR="$ROOT/$TAG"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      if [ "$HAVE" -eq "$NEED" ]; then
        echo "[ens-g$DEV] SKIP $TAG (complete: $HAVE/$NEED)"; continue
      fi
      [ "$HAVE" -gt 0 ] && echo "[ens-g$DEV] RESUME $TAG ($HAVE/$NEED)"
      if ! PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV python gpu_ready.py >/dev/null 2>&1; then
        echo "[ens-g$DEV] $(date +%H:%M:%S) ABORT: GPU $DEV not ready before $TAG"; exit 3
      fi
      echo "[ens-g$DEV] $(date +%H:%M:%S) START $TAG"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
        OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
        python train_cross_organ.py \
        --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --feature_encoder uni_v1_official \
        --splits_root $SPLITS \
        --save_root "$ROOT" \
        --exp_code "$TAG" \
        --lr $LR \
        --dump_preds \
        --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[ens-g$DEV] $(date +%H:%M:%S) END   $TAG"
    done
  done
done
echo "UNI_ENS_SHARD${SHARD}_DONE"
