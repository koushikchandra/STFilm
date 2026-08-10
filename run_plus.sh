#!/bin/bash
# STFiLM++ : cosine LR + AdamW(wd) + correlation loss, stacked on the current STFiLM config.
# Writes to results_plus_uni8 (existing results untouched). Usage: <device> <shard> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_plus_uni8; SPLITS=cross_organ_splits8; SEEDS="${SEEDS:-1 2 3}"
i=0
for FILM in none desc local; do
  for REGIME in LOOO POOLED; do
    NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
    for SEED in $SEEDS; do
      if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
      i=$((i+1))
      TAG="${REGIME}_${FILM}_seed${SEED}"; DIR="$ROOT/$TAG"
      HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
      [ "$HAVE" -eq "$NEED" ] && { echo "[plus-g$DEV] SKIP $TAG ($HAVE/$NEED)"; continue; }
      echo "[plus-g$DEV] $(date +%H:%M:%S) START $TAG"
      PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV OMP_NUM_THREADS=5 \
        python train_cross_organ.py --regime "$REGIME" --film "$FILM" --seed "$SEED" \
        --feature_encoder uni_v1_official --splits_root $SPLITS \
        --source_dataroot dataset --embed_dataroot embed_dataroot \
        --save_root "$ROOT" --lr 1e-3 --lr_schedule cosine --epochs 150 --warmup 10 --patience 40 \
        --optimizer adamw --weight_decay 0.01 --corr_weight 0.5 --dump_preds --device 0 \
        2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
      echo "[plus-g$DEV] $(date +%H:%M:%S) END   $TAG"
    done
  done
done
echo "PLUS_SHARD${SHARD}_DONE"
