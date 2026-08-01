#!/bin/bash
# V3 (desc) hyperparameter coordinate-search on the seed-42 5-fold CV tuning split.
# Each config = POOLED, 5 folds, film=desc, UNI. Score later by mean pearson over folds.
# Usage: run_uni_tune.sh <device> <shard_idx> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/transcriptomic_histopathology

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_tune_uni8
SPLITS=cross_organ_splits8_tune
SEED=42
NEED=5   # POOLED = 5 folds

# config = "lr:n_layers:hidden_dim:dropout"
CONFIGS=(
  "2e-4:4:128:0.2"
  "5e-4:4:128:0.2"
  "1e-3:4:128:0.2"
  "5e-4:4:128:0.1"
  "5e-4:4:128:0.3"
  "5e-4:6:128:0.2"
  "5e-4:4:256:0.2"
  "5e-4:6:256:0.2"
)

i=0
for CFG in "${CONFIGS[@]}"; do
  if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  IFS=':' read -r LR NL HID DO <<< "$CFG"
  TAG="lr${LR}_nl${NL}_h${HID}_do${DO}"
  DIR="$ROOT/$TAG"
  HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
  if [ "$HAVE" -eq "$NEED" ]; then
    echo "[tune-g$DEV] SKIP $TAG (complete: $HAVE/$NEED)"; continue
  fi
  [ "$HAVE" -gt 0 ] && echo "[tune-g$DEV] RESUME $TAG ($HAVE/$NEED)"
  if ! PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV python gpu_ready.py >/dev/null 2>&1; then
    echo "[tune-g$DEV] $(date +%H:%M:%S) ABORT: GPU $DEV not ready before $TAG"; exit 3
  fi
  echo "[tune-g$DEV] $(date +%H:%M:%S) START $TAG"
  PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
    OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
    python train_cross_organ.py \
    --regime POOLED --film desc --seed $SEED \
    --feature_encoder uni_v1_official \
    --splits_root $SPLITS \
    --save_root "$ROOT" \
    --exp_code "$TAG" \
    --lr "$LR" --n_layers "$NL" --hidden_dim "$HID" --dropout "$DO" \
    --device 0 \
    2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
  echo "[tune-g$DEV] $(date +%H:%M:%S) END   $TAG"
done
echo "UNI_TUNE_SHARD${SHARD}_DONE"
