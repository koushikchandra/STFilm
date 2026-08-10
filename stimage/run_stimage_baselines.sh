#!/bin/bash
# Feature-matched baselines (HisToGene, Hist2ST, ST-Net, TRIPLEX, BLEEP) on the STImage-1K4M
# cross-organ splits — the second-benchmark counterpart to run_spatial.sh. Same UNI features,
# same leakage-safe LOOO/POOLED splits, same metric, so baselines sit next to STFiLM.
# Per-fold resume (complete tags SKIP). Usage: run_stimage_baselines.sh <device> <shard> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_stimage_spatial_uni8
SPLITS=stimage_splits
NLOO=8   # 8 organs
NPOOL=5  # 5 pooled folds
i=0
for MODEL in histogene hist2st stnet triplex bleep; do
  for REGIME in LOOO POOLED; do
    NEED=$NLOO; [ "$REGIME" = "POOLED" ] && NEED=$NPOOL
    for SEED in 1 2 3; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      TAG="${REGIME}_${MODEL}_seed${SEED}"
      DIR="$ROOT/$TAG"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      if [ "$HAVE" -eq "$NEED" ]; then
        echo "[stib-g$DEV] SKIP $TAG (complete: $HAVE/$NEED)"; continue
      fi
      [ "$HAVE" -gt 0 ] && echo "[stib-g$DEV] RESUME $TAG ($HAVE/$NEED)"
      echo "[stib-g$DEV] $(date +%H:%M:%S) START $TAG"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
        OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
        python baseline_spatial.py \
        --model "$MODEL" --regime "$REGIME" --seed "$SEED" \
        --feature_encoder uni_v1_official \
        --splits_root $SPLITS \
        --source_dataroot dataset_stimage \
        --embed_dataroot embed_dataroot_stimage \
        --save_root "$ROOT" --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[stib-g$DEV] $(date +%H:%M:%S) END   $TAG"
    done
  done
done
echo "STIMAGE_BASELINE_SHARD${SHARD}_DONE"
