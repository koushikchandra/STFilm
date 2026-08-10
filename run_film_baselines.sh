#!/bin/bash
# "Conditioning is general": add FiLM (desc/local) on top of HisToGene/Hist2ST/TRIPLEX backbones.
# Additive-only: writes to results_spatial_film_uni8; the `none` reference is the existing
# results_spatial_uni8 run for each backbone. Per-fold resume. Usage: <device> <shard> <n_shards>
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm

DEV="$1"; SHARD="$2"; NSHARDS="$3"
ROOT=results_spatial_film_uni8
SPLITS=cross_organ_splits8
SEEDS="${SEEDS:-1 2 3}"
i=0
for MODEL in histogene hist2st triplex; do
  for FILM in desc local; do
    for REGIME in LOOO POOLED; do
      NEED=8; [ "$REGIME" = "POOLED" ] && NEED=5
      for SEED in $SEEDS; do
        if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
        i=$((i+1))
        TAG="${REGIME}_${MODEL}-${FILM}_seed${SEED}"
        DIR="$ROOT/$TAG"
        HAVE=$(find "$DIR" -name "fold_*_results.json" 2>/dev/null | wc -l)
        if [ "$HAVE" -eq "$NEED" ]; then echo "[film-g$DEV] SKIP $TAG ($HAVE/$NEED)"; continue; fi
        [ "$HAVE" -gt 0 ] && echo "[film-g$DEV] RESUME $TAG ($HAVE/$NEED)"
        if ! PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV python gpu_ready.py >/dev/null 2>&1; then
          echo "[film-g$DEV] $(date +%H:%M:%S) ABORT: GPU $DEV not ready before $TAG"; exit 3
        fi
        echo "[film-g$DEV] $(date +%H:%M:%S) START $TAG"
        PYTHONPATH=STFlow CUDA_VISIBLE_DEVICES=$DEV \
          OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
          python baseline_film.py \
          --model "$MODEL" --film "$FILM" --regime "$REGIME" --seed "$SEED" \
          --feature_encoder uni_v1_official \
          --splits_root $SPLITS --save_root "$ROOT" --device 0 \
          2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24|accum.append"
        echo "[film-g$DEV] $(date +%H:%M:%S) END   $TAG"
      done
    done
  done
done
echo "FILM_SHARD${SHARD}_DONE"
