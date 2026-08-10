#!/bin/bash
# CoMRA v4: correlation loss + feature-noise + distance-jitter augmentation (on top of v3 recipe:
# cosine LR, contrastive lambda=0.1, dumped preds). Usage: <device> <shard> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_comra_uni8; SPLITS=cross_organ_splits8; SEEDS="${SEEDS:-1 2 3}"; TAG=comra-v4
i=0
for REGIME in LOOO POOLED; do
  NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
  for SEED in $SEEDS; do
    if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    RT="${REGIME}_${TAG}_seed${SEED}"; DIR="$ROOT/$RT"
    HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
    [ "$HAVE" -eq "$NEED" ] && { echo "[comra4-g$DEV] SKIP $RT ($HAVE/$NEED)"; continue; }
    echo "[comra4-g$DEV] $(date +%H:%M:%S) START $RT"
    PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV OMP_NUM_THREADS=5 \
      python newmodel.py --regime "$REGIME" --seed "$SEED" \
      --run_tag "$TAG" --lambda_con 0.1 --corr_weight 0.5 --feat_noise 0.1 --dist_jitter 0.05 --dump_preds \
      --feature_encoder uni_v1_official --splits_root $SPLITS --save_root "$ROOT" --device 0 \
      2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
    echo "[comra4-g$DEV] $(date +%H:%M:%S) END   $RT"
  done
done
echo "COMRA4_SHARD${SHARD}_DONE"
