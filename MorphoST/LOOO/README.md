# 5-Organ Cross-Organ POOLED + LOOO (self-contained test cases)

> **This `POOLED` branch adds the 5-fold POOLED run** on the no-COAD/PAAD organ set
> **CCRCC** (kidney), **IDC** (breast), **LUNG** (lung), **PRAD** (prostate), **SKCM** (skin).
> See **[POOLED run](#pooled-run-5-fold-pooled-cross-validation)** below. The original LOOO
> instructions follow after it and are unchanged.

## POOLED run (5-fold pooled cross-validation)

Pooled leave-patient-out over the 5 organs above: every organ appears in train **and** test of
every fold, so this measures in-distribution generalization (contrast with LOOO, which holds a
whole organ out). Splits: `cross_organ_splits5_nocp/POOLED/` (5 folds, leakage-safe 10-gene panel
= the genes measured in all 5 cohorts). Frozen UNI+CONCH features (1536-dim). Same five models.

**Seeds vs. folds (both, not either):** the run is **5-fold** CV repeated over **3 seeds** →
**3 × 5 = 15 fold-runs per model**. Seeds and folds are different axes:
- *5 folds* = the CV splits (folds 0–4). **One training command runs all 5 folds internally** and
  writes `results_kfold.json → pearson_mean` = the PCC averaged over the 5 folds. You never pass
  `--folds` for POOLED.
- *3 seeds* = repeat the whole 5-fold run with `--seed 1`, `2`, `3` for error bars.

So each `.sbatch` array task = **one seed** (which then loops the 5 folds): `pooled5nocp_mist.sbatch`
is `--array=0-2` (3 seeds); `pooled5nocp_baselines.sbatch` is `--array=0-11` (4 models × 3 seeds).
**Final table cell = mean ± std over the 3 seeds** of each seed's 5-fold `pearson_mean`.

**Cohorts needed:** `CCRCC  IDC  LUNG  PRAD  SKCM` (note: **not** COAD/PAAD). Get the data bundle
from the **`pooled5-data`** GitHub Release — see [DATA.md](DATA.md) — then:
```bash
export DATA_ROOT=~/pooled5_data/dataset
export EMBED_ROOT=~/pooled5_data/embed_dataroot
```

### On SLURM (any non-Volta GPU — A100/A40/L40s/H200/RTX all fine; **v100 fails under CUDA 13**)
```bash
sbatch pooled5nocp_mist.sbatch                 # MIST (ours), 3 seeds
sbatch baselines/pooled5nocp_baselines.sbatch  # ST-Net, Hist2ST, BLEEP, STEM (4 x 3 seeds)
```
Both are `--requeue`-safe and skip already-finished folds, so a preemption just resumes.

### Without a cluster (GPU if present, else CPU)
MIST (POOLED auto-uses folds 0–4, so no `--folds`):
```bash
PYTHONPATH=$PWD python train.py --regime POOLED --version V3 --components 111 --seed 1 \
  --feature_encoder uni_conch --splits_root cross_organ_splits5_nocp \
  --source_dataroot "$DATA_ROOT" --embed_dataroot "$EMBED_ROOT" \
  --save_root results_pooled5nocp_mist --epochs 100 --patience 20 --device cuda
```
Baselines — set `MODEL` to `stnet`, `hist2st`, `bleep`, then `stem`:
```bash
for MODEL in stnet hist2st bleep stem; do
  PYTHONPATH=$PWD python baselines/baseline_spatial.py --model $MODEL --regime POOLED --seed 1 \
    --feature_encoder uni_conch --splits_root cross_organ_splits5_nocp \
    --source_dataroot "$DATA_ROOT" --embed_dataroot "$EMBED_ROOT" \
    --save_root baselines/results_pooled5nocp_baselines --epochs 100 --patience 20 --device 0
done
```
Repeat with `--seed 2` and `--seed 3` for the 3-seed averages. Results land in
`results_pooled5nocp_mist/POOLED_C111_seed*/` and
`baselines/results_pooled5nocp_baselines/POOLED_<model>_seed*/` (per-fold JSON + `results_kfold.json`;
each seed's `pearson_mean` is the pooled PCC).

### Regenerate the POOLED splits (optional)
```bash
python make_splits_5nocp.py --regimes A --n_folds 5 --n_genes 50 \
    --gene_panel_mode union_topk --check_availability \
    --hest_root "$DATA_ROOT" --out_root cross_organ_splits5_nocp
```

---

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
