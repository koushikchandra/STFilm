#!/bin/bash
# STImage-1K4M second benchmark: film none/desc/local x LOOO/POOLED x 3 seeds on the
# leakage-safe cross-organ STImage splits. Mirrors run_uni_ens.sh but points at the STImage
# roots and passes --organ_set stimage. Usage: run_stimage.sh <device> <shard_idx> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_stimage_uni8
SPLITS=stimage_splits
LR=1e-3
NLOO=8   # 8 organs
NPOOL=5  # 5 pooled folds
i=0
for FILM in none desc local; do
  for REGIME in LOOO POOLED; do
    NEED=$NLOO; [ "$REGIME" = "POOLED" ] && NEED=$NPOOL
    for SEED in 1 2 3; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      TAG="${REGIME}_${FILM}_seed${SEED}"
      DIR="$ROOT/$TAG"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      if [ "$HAVE" -eq "$NEED" ]; then
        echo "[sti-g$DEV] SKIP $TAG (complete: $HAVE/$NEED)"; continue
      fi
      [ "$HAVE" -gt 0 ] && echo "[sti-g$DEV] RESUME $TAG ($HAVE/$NEED)"
      echo "[sti-g$DEV] $(date +%H:%M:%S) START $TAG"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
        OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
        python train_cross_organ.py \
        --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --organ_set stimage \
        --feature_encoder uni_v1_official \
        --splits_root $SPLITS \
        --source_dataroot dataset_stimage \
        --embed_dataroot embed_dataroot_stimage \
        --save_root "$ROOT" \
        --exp_code "$TAG" \
        --lr $LR \
        --dump_preds \
        --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[sti-g$DEV] $(date +%H:%M:%S) END   $TAG"
    done
  done
done
echo "STIMAGE_SHARD${SHARD}_DONE"
