#!/bin/bash
#SBATCH --job-name=morpho_rec
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=morpho_rec_%a_%j.log
#SBATCH --array=0-3
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# Recover the morphology-conditioning novelty (seed1 LOOO). Two hypotheses for why V4/V5 regressed:
#  H1 local desc redundant with local kNN attn -> use global-only conditioning
#  H2 conditioner overfits small folds        -> add cond_dropout
# code                version scope   cond_dropout
CFG=(
 "R1_V4_global    V4 global 0.0"
 "R2_V4_both_dp3  V4 both   0.3"
 "R3_V5_global_dp3 V5 global 0.3"
 "R4_V5_both_dp3  V5 both   0.3"
)
set -- ${CFG[$SLURM_ARRAY_TASK_ID]}
CODE=$1; VER=$2; SCOPE=$3; CD=$4
echo "[rec] START $CODE (ver=$VER scope=$SCOPE cond_dropout=$CD)"
PYTHONPATH=../STFlow OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python train.py --regime LOOO --version "$VER" --seed 1 \
  --morph_scope "$SCOPE" --cond_dropout "$CD" --exp_code "LOOO_${CODE}_seed1" \
  --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
  --embed_dataroot ../embed_dataroot --feature_encoder uni_v1_official \
  --save_root results_morphost --device cuda \
  2>&1 | grep -vE "FutureWarning|pynvml|UserWarning"
echo "[rec] DONE $CODE"
