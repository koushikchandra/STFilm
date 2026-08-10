#!/bin/bash
# STFiLM on GigaPath features (baseline config: epochs 100, no schedule) -> clean encoder swap vs UNI.
# Usage: <device> <shard> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_gigapath; SPLITS=cross_organ_splits8; SEEDS="${SEEDS:-1 2 3}"
i=0
for FILM in none desc local; do
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for SEED in $SEEDS; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      TAG="${REGIME}_${FILM}_seed${SEED}"; DIR="$ROOT/$TAG"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      [ "$HAVE" -eq "$NEED" ] && { echo "[gp-g$DEV] SKIP $TAG ($HAVE/$NEED)"; continue; }
      echo "[gp-g$DEV] $(date +%H:%M:%S) START $TAG"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV OMP_NUM_THREADS=5 \
        python train_cross_organ.py --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --feature_encoder gigapath --splits_root $SPLITS \
        --source_dataroot dataset --embed_dataroot embed_dataroot \
        --save_root "$ROOT" --lr 1e-3 --epochs 100 --dump_preds --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[gp-g$DEV] $(date +%H:%M:%S) END   $TAG"
    done
  done
done
echo "GIGAPATH_SHARD${SHARD}_DONE"
