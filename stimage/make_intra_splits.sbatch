#!/bin/bash
#SBATCH --job-name=sti_intra_splits
#SBATCH --partition=nova
#SBATCH --account=weile-lab
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=sti_intra_splits_%j.log

set -u
cd /lustre/hdd/LAS/weile-lab/howlader/STFilm
export PATH=/lustre/hdd/LAS/weile-lab/howlader/miniconda3/bin:$PATH

echo "[splits] node=$(hostname) job=$SLURM_JOB_ID"
python stimage/make_splits.py \
    --src_root dataset_stimage \
    --out stimage_splits \
    --k_pooled 5 --seed 0
echo "[splits] DONE"
