#!/bin/bash
#SBATCH --job-name=abl_pool
#SBATCH --partition=scavenger
#SBATCH --account=weile-lab
#SBATCH --qos=scavenger
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --exclude=nova18-gpu-1,nova18-gpu-2,nova18-gpu-3,nova18-gpu-4,nova18-gpu-5,nova18-singularity,nova20-frost-1,nova20-matrix
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=abl_pool_%a_%j.log
#SBATCH --array=0-47

# ABLATION on POOLED (5-fold pooled, no-COAD/PAAD 5-organ set), MIST only, UNI+CONCH (1536-dim).
# 16 configs x 3 seeds = 48 array tasks. Each task = one (config, seed); it loops folds 0-4.
# Model: ContextMIST (morphost_context.py) — same architecture, streams toggled by --config.
# --exclude drops v100 (Volta) nodes: they fail under CUDA 13. EDIT #SBATCH lines for your cluster.

set -u

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # dir you sbatch from (the LOOO/ dir)
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/../../dataset}"
EMBED_ROOT="${EMBED_ROOT:-$REPO_DIR/../../embed_dataroot}"
cd "$REPO_DIR"

# recipe = "TAG|CONFIG|EXTRA_ARGS"  (TAG names the output dir; CONFIG selects streams; EXTRA overrides)
RECIPES=(
  # --- context-stream factorial (7: all non-empty L/G/S combos) ---
  "full|full|"                                  # 1. Full MIST            L G S
  "global|global|"                              # 2. Only Global          _ G _
  "global_local|global_local|"                  # 3. Global + Local       L G _
  "global_slide|global_slide|"                  # 4. Global + Slide       _ G S
  "local|local|"                                #    Local only           L _ _
  "slide|slide|"                                #    Slide only           _ _ S
  "local_slide|local_slide|"                    #    Local + Slide        L _ S
  # --- local-mechanism ablations ---
  "global_random|global_random|"                #    random neighbours (vs spatial kNN)
  "full_nodist|full|--no-use_distance_bias"     #    full, RBF distance-bias OFF
  "global_local_nodist|global_local|--no-use_distance_bias"  # G+L, distance-bias OFF
  # --- capacity / k sweep (on full) ---
  "full_k4|full|--k 4"
  "full_k16|full|--k 16"
  "full_L2|full|--n_layers 2"
  "full_L6|full|--n_layers 6"
  "full_d128|full|--dim 128"
  "full_d512|full|--dim 512"
)
SEEDS=(1 2 3)
NR=${#RECIPES[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
IFS='|' read -r TAG CONFIG EXTRA <<< "${RECIPES[$((IDX % NR))]}"
SEED=${SEEDS[$((IDX / NR))]}

echo "[abl_pool] node=$(hostname) task=$IDX  tag=$TAG config=$CONFIG extra='$EXTRA' seed=$SEED"
python -c "import torch; print('[abl_pool] cuda:', torch.cuda.is_available())"

NEED=5
EXP="POOLED_${TAG}_seed${SEED}"
ROOT=results_ablation
HAVE=$(find "$ROOT/$EXP" -name "fold_*_results.json" 2>/dev/null | wc -l)
[ "$HAVE" -ge "$NEED" ] && { echo "[abl_pool] SKIP $EXP (complete $HAVE/$NEED)"; exit 0; }
[ "$HAVE" -gt 0 ] && echo "[abl_pool] RESUME $EXP ($HAVE/$NEED)"

echo "[abl_pool] $(date +%H:%M:%S) START $EXP"
PYTHONPATH="$REPO_DIR" OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train_ablation.py \
  --regime POOLED --config "$CONFIG" --seed "$SEED" --exp_code "$EXP" $EXTRA \
  --feature_encoder uni_conch \
  --splits_root cross_organ_splits5_nocp \
  --source_dataroot "$DATA_ROOT" \
  --embed_dataroot "$EMBED_ROOT" \
  --save_root "$ROOT" \
  --val_fraction 0.15 --epochs 100 --patience 20 --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|fa.py:24"
echo "[abl_pool] $(date +%H:%M:%S) END $EXP"
