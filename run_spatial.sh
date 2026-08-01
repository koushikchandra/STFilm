#!/bin/bash
# Feature-matched spatial baselines (HisToGene-style, Hist2ST-style) on UNI 8-organ splits.
# Mirrors run_uni_final.sh: shard configs across GPUs, GPU-ready guard, per-fold resume.
# Usage: run_spatial.sh <device> <shard_idx> <n_shards>   (SEEDS env overrides seed list)
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_spatial_uni8
SPLITS=cross_organ_splits8
SEEDS="${SEEDS:-1 2 3}"

i=0
for MODEL in histogene hist2st; do
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for SEED in $SEEDS; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      TAG="${REGIME}_${MODEL}_seed${SEED}"
      DIR="$ROOT/$TAG"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      if [ "$HAVE" -eq "$NEED" ]; then
        echo "[sp-g$DEV] SKIP $TAG (complete: $HAVE/$NEED)"; continue
      fi
      [ "$HAVE" -gt 0 ] && echo "[sp-g$DEV] RESUME $TAG ($HAVE/$NEED)"
      if ! PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV python gpu_ready.py >/dev/null 2>&1; then
        echo "[sp-g$DEV] $(date +%H:%M:%S) ABORT: GPU $DEV not ready before $TAG"; exit 3
      fi
      echo "[sp-g$DEV] $(date +%H:%M:%S) START $TAG"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
        OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
        python baseline_spatial.py \
        --model "$MODEL" --regime "$REGIME" --seed "$SEED" \
        --feature_encoder uni_v1_official \
        --splits_root $SPLITS --save_root "$ROOT" --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[sp-g$DEV] $(date +%H:%M:%S) END   $TAG"
    done
  done
done
echo "SPATIAL_SHARD${SHARD}_DONE"
