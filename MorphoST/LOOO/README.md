# 5-Organ Cross-Organ LOOO (self-contained test case)

Leave-one-organ-out (LOOO) transfer on a **self-contained 5-organ pool**:
**COAD** (colorectal), **CCRCC** (kidney), **LUNG** (lung), **PAAD** (pancreas), **PRAD** (prostate).
Each of the 5 folds holds out one organ for testing and trains **only on the other 4**.
Gene panels are recomputed leakage-safe from the training organs (16–33 genes/fold).

Everything needed to run is inside this directory (a vendored subset of `stflow/` is bundled),
so you can clone just `LOOO/` and run — **only the HEST data lives outside** (see below).

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
Point `DATA_ROOT` and `EMBED_ROOT` (top of each `.sbatch`) at:
- `DATA_ROOT/<COHORT>/adata/<sample>.h5ad` — expression, and `var_50genes.json`
- `EMBED_ROOT/<COHORT>/<encoder>/fp32/<sample>.h5` — frozen patch embeddings
  (encoder used here: `uni_conch`, 1536-dim)

Cohorts needed: `COAD  CCRCC  LUNG  PAAD  PRAD`.

## Run
On SLURM:
```
sbatch looo5_mist.sbatch                 # MIST
sbatch baselines/looo5_baselines.sbatch  # baselines
```
Without a cluster, run the `python ...` block from inside each `.sbatch` directly
(set `DATA_ROOT`/`EMBED_ROOT` first, e.g. `export DATA_ROOT=/path/to/dataset`).

Results: `results_looo5_mist/LOOO_C111_seed*/` and
`baselines/results_looo5_baselines/LOOO_<model>_seed*/` (per-fold + `results_kfold.json`).

## Dependencies
`pip install -r requirements.txt` (Python 3.10+, a CUDA GPU for training).

## Regenerating the splits (optional)
```
python make_splits_5organ.py --hest_root $DATA_ROOT --out_root cross_organ_splits5 \
    --regimes B --check_availability
```
