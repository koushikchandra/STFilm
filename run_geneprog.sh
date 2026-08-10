#!/bin/bash
# Shared gene-program factorization: factor (A@B, shared basis) vs direct (plain head), SAME encoder.
# The controlled test of the novel mechanism. Usage: <device> <shard> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_geneprog_uni8; SPLITS=cross_organ_splits8; SEEDS="${SEEDS:-1 2 3}"
i=0
for HEAD in factor direct; do
  TAG="gp-${HEAD}"
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for SEED in $SEEDS; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      RT="${REGIME}_${TAG}_seed${SEED}"; DIR="$ROOT/$RT"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      [ "$HAVE" -eq "$NEED" ] && { echo "[gp-g$DEV] SKIP $RT ($HAVE/$NEED)"; continue; }
      echo "[gp-g$DEV] $(date +%H:%M:%S) START $RT"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV OMP_NUM_THREADS=5 \
        python geneprog.py --regime "$REGIME" --seed "$SEED" --head "$HEAD" --run_tag "$TAG" \
        --feature_encoder uni_v1_official --splits_root $SPLITS --save_root "$ROOT" --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[gp-g$DEV] $(date +%H:%M:%S) END   $RT"
    done
  done
done
echo "GP_SHARD${SHARD}_DONE"
