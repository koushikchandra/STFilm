#!/bin/bash
# CoMRA v3: cosine LR + warmup + longer budget + dumped preds; contrastive ablation
# (comra-con: lambda=0.1, comra-con0: lambda=0). Usage: <device> <shard> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_comra_uni8; SPLITS=cross_organ_splits8; SEEDS="${SEEDS:-1 2 3}"
i=0
for COND in "comra-con:0.1" "comra-con0:0.0"; do
  TAG="${COND%%:*}"; LAM="${COND##*:}"
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for SEED in $SEEDS; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      RT="${REGIME}_${TAG}_seed${SEED}"; DIR="$ROOT/$RT"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      [ "$HAVE" -eq "$NEED" ] && { echo "[comra3-g$DEV] SKIP $RT ($HAVE/$NEED)"; continue; }
      echo "[comra3-g$DEV] $(date +%H:%M:%S) START $RT (lambda=$LAM)"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV OMP_NUM_THREADS=5 \
        python newmodel.py --regime "$REGIME" --seed "$SEED" \
        --run_tag "$TAG" --lambda_con "$LAM" --dump_preds \
        --feature_encoder uni_v1_official --splits_root $SPLITS --save_root "$ROOT" --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[comra3-g$DEV] $(date +%H:%M:%S) END   $RT"
    done
  done
done
echo "COMRA3_SHARD${SHARD}_DONE"
