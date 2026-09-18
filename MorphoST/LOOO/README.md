# 5-Organ Cross-Organ LOOO (self-contained test case)

Leave-one-organ-out (LOOO) transfer on a **self-contained 5-organ pool**:
**COAD** (colorectal), **CCRCC** (kidney), **LUNG** (lung), **PAAD** (pancreas), **PRAD** (prostate).
Each of the 5 folds holds out one organ for testing and trains **only on the other 4**.
Gene panels are recomputed leakage-safe from the training organs (16–33 genes/fold).

Everything needed to run is inside this directory (a vendored subset of `stflow/` is bundled),
so you can clone just `LOOO/` and run — **only the HEST data lives outside** (see below).

**Five models are evaluated:** `ST-Net`, `Hist2ST`, `BLEEP` (NeurIPS'23), `STEM`, and **`MIST`** (ours).
Model → command name: `ST-Net=stnet`, `Hist2ST=hist2st`, `BLEEP=bleep`, `STEM=stem` (all via
`baselines/baseline_spatial.py`); `MIST` via `train.py`.

## Layout
```
LOOO/
├── looo5_mist.sbatch            # runs MIST (our model), 3 seeds x 5 folds
├── train.py  morphost.py        # MIST trainer + model
├── data.py  evaluation.py       # data loading + metrics
├── make_splits_5organ.py        # regenerates cross_organ_splits5/ (already generated)
├── cross_organ_splits5/LOOO/    # splits/ (train_/test_ CSVs) + genes_<organ>.json panels
├── stflow/                      # vendored STFlow helpers (no external STFlow repo needed)
└── baselines/
    ├── looo5_baselines.sbatch   # runs ST-Net, Hist2ST, BLEEP, STEM (4 models x 3 seeds)
    └── baseline_spatial.py      # all baseline models incl. BLEEP (class BleepEncoder / bleep_train_fold)
```

## Data you must provide (not in this repo — too large)
**See [DATA.md](DATA.md) for how to download it.** In short, point `DATA_ROOT` and `EMBED_ROOT`
(top of each `.sbatch`) at:
- `DATA_ROOT/<COHORT>/adata/<sample>.h5ad` — expression, and `var_50genes.json`
- `EMBED_ROOT/<COHORT>/<encoder>/fp32/<sample>.h5` — frozen patch embeddings
  (encoder used here: `uni_conch`, 1536-dim)

Cohorts needed: `COAD  CCRCC  LUNG  PAAD  PRAD`.

## Run — all five models (ST-Net, Hist2ST, BLEEP, STEM, MIST)

First set the data paths (see [DATA.md](DATA.md)):
```bash
export DATA_ROOT=~/looo5_data/dataset
export EMBED_ROOT=~/looo5_data/embed_dataroot
```

### On SLURM (any GPU — A100 not required)
```bash
sbatch looo5_mist.sbatch                 # MIST (ours)
sbatch baselines/looo5_baselines.sbatch  # ST-Net, Hist2ST, BLEEP, STEM (4 baselines x 3 seeds)
```

### Without a cluster (GPU if present, else CPU)
MIST:
```bash
PYTHONPATH=$PWD python train.py --regime LOOO --version V3 --components 111 --seed 1 \
  --feature_encoder uni_conch --splits_root cross_organ_splits5 \
  --folds colorectal kidney lung pancreas prostate \
  --source_dataroot "$DATA_ROOT" --embed_dataroot "$EMBED_ROOT" \
  --save_root results_looo5_mist --epochs 100 --patience 20 --device cuda
```
Each baseline — set `MODEL` to `stnet`, `hist2st`, `bleep`, then `stem`:
```bash
for MODEL in stnet hist2st bleep stem; do
  PYTHONPATH=$PWD python baselines/baseline_spatial.py --model $MODEL --regime LOOO --seed 1 \
    --feature_encoder uni_conch --splits_root cross_organ_splits5 \
    --source_dataroot "$DATA_ROOT" --embed_dataroot "$EMBED_ROOT" \
    --save_root baselines/results_looo5_baselines --epochs 100 --patience 20 --device 0
done
```
Repeat with `--seed 2` and `--seed 3` for the 3-seed averages.

Results: `results_looo5_mist/LOOO_C111_seed*/` (MIST) and
`baselines/results_looo5_baselines/LOOO_<model>_seed*/` (baselines) — per-fold JSON + `results_kfold.json`.

## Dependencies
`pip install -r requirements.txt` (Python 3.10+). Runs on **any** CUDA GPU (not A100-specific —
MIST is 4.88M params and the encoder features are precomputed), or on **CPU** (slower); the
trainers auto-detect the device.

## Regenerating the splits (optional)
```
python make_splits_5organ.py --hest_root $DATA_ROOT --out_root cross_organ_splits5 \
    --regimes B --check_availability
```
