#!/bin/bash
# Lever #8 (capacity bump) on the best variant (local): depth 4->6, width 128->256.
# Everything else identical to run_localg/run_uni_local -> clean comparison to `local`.
# Usage: <dev> <shard> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_bigcap_uni8; SPLITS=cross_organ_splits8; SEEDS="${SEEDS:-1 2 3}"
i=0
for REGIME in LOOO POOLED; do
  NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
  for SEED in $SEEDS; do
    if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    TAG="${REGIME}_local_bigcap_seed${SEED}"; DIR="$ROOT/$TAG"
    HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
    [ "$HAVE" -eq "$NEED" ] && { echo "[bc-g$DEV] SKIP $TAG"; continue; }
    echo "[bc-g$DEV] $(date +%H:%M:%S) START $TAG (n_layers=6 hidden=256)"
    PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV OMP_NUM_THREADS=5 \
      python train_cross_organ.py --regime "$REGIME" --film local --seed "$SEED" \
      --feature_encoder uni_v1_official --splits_root $SPLITS \
      --source_dataroot dataset --embed_dataroot embed_dataroot \
      --save_root "$ROOT" --exp_code "$TAG" \
      --n_layers 6 --hidden_dim 256 --pairwise_hidden_dim 256 \
      --lr 1e-3 --epochs 100 --dump_preds --device 0 \
      2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
    echo "[bc-g$DEV] $(date +%H:%M:%S) END   $TAG"
  done
done
echo "BIGCAP_SHARD${SHARD}_DONE"
