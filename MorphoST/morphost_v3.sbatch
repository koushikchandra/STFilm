#!/bin/bash
#SBATCH --job-name=morpho_v3
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --exclude=nova22-amp-7,nova21-gpu-2,nova21-gpu-10
#SBATCH --output=morpho_v3_%a_%j.log
#SBATCH --array=0-4
set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH
# Invest in best version V3: LOOO seeds 2,3 + POOLED seeds 1,2,3
COMBOS=("LOOO 2" "LOOO 3" "POOLED 1" "POOLED 2" "POOLED 3")
set -- ${COMBOS[$SLURM_ARRAY_TASK_ID]}
REGIME=$1; SEED=$2
bash run_morphost.sh V3 "$REGIME" "$SEED"
echo "[morpho_v3] DONE $REGIME seed$SEED"
