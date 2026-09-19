# 5-Organ Cross-Organ POOLED + LOOO (self-contained test cases)

> **This `POOLED` branch adds the 5-fold POOLED run** on the no-COAD/PAAD organ set
> **CCRCC** (kidney), **IDC** (breast), **LUNG** (lung), **PRAD** (prostate), **SKCM** (skin).
> See **[POOLED run](#pooled-run-5-fold-pooled-cross-validation)** below. The original LOOO
> instructions follow after it and are unchanged.

## POOLED run (5-fold pooled cross-validation)

Pooled leave-patient-out over the 5 organs above: every organ appears in train **and** test of
every fold, so this measures in-distribution generalization (contrast with LOOO, which holds a
whole organ out). Splits: `cross_organ_splits5_nocp/POOLED/` (5 folds, leakage-safe 10-gene panel
= the genes measured in all 5 cohorts). Frozen UNI+CONCH features (1536-dim). Same five models
(`ST-Net=stnet`, `Hist2ST=hist2st`, `BLEEP=bleep`, `STEM=stem` via `baselines/baseline_spatial.py`;
**`MIST`** (ours) via `train.py`).

### 0. Prerequisites
- **Python 3.10+** and a **GPU** (any non-Volta card — see the GPU note below — or CPU, slower).
- Everything to run is in this directory: a vendored subset of `stflow/` is bundled, so **no
  external STFlow repo is needed**. Only the HEST data lives outside (step 2).

### 1. Get the code
```bash
git clone https://github.com/koushikchandra/STFilm.git
cd STFilm && git checkout POOLED
cd MorphoST/LOOO
pip install -r requirements.txt      # torch, torch_geometric, scanpy, timm, einops, pandas, scipy, h5py
```

### 2. Get the data  (`pooled5-data` release — **not** `looo5-data`)
The POOLED cohorts are `CCRCC  IDC  LUNG  PRAD  SKCM` (breast + skin, **not** COAD/PAAD), so they
have their own bundle. Full detail in [DATA.md](DATA.md); in short:
```bash
gh release download pooled5-data --repo koushikchandra/STFilm --dir ~/pooled5_data
cd ~/pooled5_data && cat pooled5_data.tar.gz.part-* > pooled5_data.tar.gz && tar xzf pooled5_data.tar.gz
export DATA_ROOT=~/pooled5_data/dataset          # <COHORT>/adata/*.h5ad  + var_50genes.json
export EMBED_ROOT=~/pooled5_data/embed_dataroot  # <COHORT>/uni_conch/fp32/*.h5  (frozen, 1536-dim)
```
Verify before running (both must list files):
```bash
ls $DATA_ROOT/IDC/adata/*.h5ad ; ls $EMBED_ROOT/SKCM/uni_conch/fp32/*.h5
```
No `gh`? Download the parts from the [release page](https://github.com/koushikchandra/STFilm/releases/tag/pooled5-data)
and run the same `cat … > pooled5_data.tar.gz && tar xzf …`.

### 3. Seeds vs. folds (you run BOTH, not either)
The run is **5-fold** CV repeated over **3 seeds** → **3 × 5 = 15 fold-runs per model**:
- *5 folds* = the CV splits (folds 0–4). **One training command runs all 5 folds internally** and
  writes `results_kfold.json → pearson_mean` = the PCC averaged over the 5 folds. You never pass
  `--folds` for POOLED (it defaults to `0 1 2 3 4`).
- *3 seeds* = repeat the whole 5-fold run with `--seed 1`, `2`, `3` for error bars.

Each `.sbatch` **array task = one seed** (which then loops the 5 folds): `pooled5nocp_mist.sbatch`
is `--array=0-2` (3 seeds); `pooled5nocp_baselines.sbatch` is `--array=0-11` (4 models × 3 seeds =
12 tasks). **Final table cell = mean ± std over the 3 seeds** of each seed's 5-fold `pearson_mean`.

### 4a. Run on SLURM
```bash
sbatch pooled5nocp_mist.sbatch                 # MIST (ours), 3 seeds
sbatch baselines/pooled5nocp_baselines.sbatch  # ST-Net, Hist2ST, BLEEP, STEM (4 x 3 seeds)
```
Both are `--requeue`-safe and **skip already-finished folds**, so a preemption/resubmit just resumes.
`DATA_ROOT`/`EMBED_ROOT` are read from the environment (defaults point two levels up); export them, or
edit the two lines at the top of each `.sbatch`.

> **⚠ Cluster-specific `#SBATCH` headers.** The two `.sbatch` files are tuned for **our** nova cluster:
> `--partition=scavenger --account=weile-lab --qos=scavenger` and an `--exclude=…` list of Volta
> (v100) nodes. **On any other cluster, edit those lines** — set your own partition/account/qos and
> **remove the `--exclude` line** (or replace it with your own way of avoiding v100 / pre-sm_75 GPUs).
> Keep `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=64G`.

### 4b. Run without a cluster (GPU if present, else CPU)
MIST (no `--folds` — POOLED auto-runs folds 0–4):
```bash
PYTHONPATH=$PWD python train.py --regime POOLED --version V3 --components 111 --seed 1 \
  --feature_encoder uni_conch --splits_root cross_organ_splits5_nocp \
  --source_dataroot "$DATA_ROOT" --embed_dataroot "$EMBED_ROOT" \
  --save_root results_pooled5nocp_mist --epochs 100 --patience 20 --device cuda
```
Baselines — loop the four models:
```bash
for MODEL in stnet hist2st bleep stem; do
  PYTHONPATH=$PWD python baselines/baseline_spatial.py --model $MODEL --regime POOLED --seed 1 \
    --feature_encoder uni_conch --splits_root cross_organ_splits5_nocp \
    --source_dataroot "$DATA_ROOT" --embed_dataroot "$EMBED_ROOT" \
    --save_root baselines/results_pooled5nocp_baselines --epochs 100 --patience 20 --device 0
done
```
Then repeat everything with `--seed 2` and `--seed 3`. (`--device cuda`/`--device 0` = GPU; use
`--device cpu`/omit CUDA to force CPU.)

### 5. Read & aggregate the results
Output tree (one dir per seed; MIST tag is `C111`):
```
results_pooled5nocp_mist/POOLED_C111_seed{1,2,3}/
    fold_{0..4}_results.json     # per-fold metrics
    results_kfold.json           # -> "pearson_mean" = PCC averaged over the 5 folds (that seed)
baselines/results_pooled5nocp_baselines/POOLED_{stnet,hist2st,bleep,stem}_seed{1,2,3}/  (same layout)
```
The reported POOLED PCC for a model = **mean ± std over the 3 seeds** of `pearson_mean`:
```bash
# one model, all 3 seeds -> mean and std
python - <<'PY'
import json, glob, numpy as np
for tag in ["results_pooled5nocp_mist/POOLED_C111",
            "baselines/results_pooled5nocp_baselines/POOLED_bleep"]:
    v=[json.load(open(f))["pearson_mean"] for f in sorted(glob.glob(f"{tag}_seed*/results_kfold.json"))]
    if v: print(f"{tag.split('/')[-1]:20s} PCC = {np.mean(v):.3f} ± {np.std(v):.3f}  (n={len(v)} seeds)")
PY
```
`results_kfold.json` also carries `mse_mean` / `mae_mean` if you need the error metrics.

### Runtime & resources
~4–6 h wall-clock per array task on one modern GPU for the heavier models (BLEEP/STEM/MIST); ST-Net
is faster. 15 tasks total; they can all run in parallel if you have the GPUs, or serially otherwise
(resume-safe). Peak GPU memory is small (MIST is 4.88M params; features are precomputed) — a single
16 GB card is plenty.

### GPU compatibility note
Runs on **any CUDA GPU with compute capability ≥ sm_75** — A100, A40, L40s, H200, RTX all work. It
does **not** need an A100. **Avoid v100 / older Volta (sm_70)**: with a CUDA-13 / torch-2.12 build
they fail at launch with `cudaErrorNoKernelImage`. CPU also works (slower); the trainers auto-detect
the device.

### Troubleshooting
- **`EMBED_ROOT` empty / `FileNotFoundError` on a `.h5`** → you have expression but not the frozen
  embeddings; re-download the bundle (step 2) — it contains `embed_dataroot/`.
- **`cudaErrorNoKernelImage` / "no kernel image is available"** → you're on a v100; move to a
  newer GPU or run on CPU.
- **`ModuleNotFoundError: stflow`** → run from **this** directory with `PYTHONPATH=$PWD` (the
  vendored `stflow/` shim is local; don't `pip install` the upstream STFlow).
- **A fold errored midway** → just resubmit/rerun the same command; finished folds are skipped and
  it continues from where it stopped.

### Regenerate the POOLED splits (optional — already included)
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
