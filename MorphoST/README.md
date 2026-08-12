# MorphoST — a Simple E(2)-Invariant Local–Global Transformer for Spatial Transcriptomics

MorphoST predicts spatially-resolved gene expression from H&E histology by **direct regression** —
no flow matching, no ZINB prior, no frame averaging. It is **E(2)-invariant by construction**
(coordinates enter only through pairwise distances, never absolute positions) and fuses three cheap
sources of context per layer: distance-biased **local kNN attention**, **global self-attention**, and
a **global slide token** (attention pool).

**Headline result (HEST-1k, per-cohort k-fold, top-50 HVG, Pearson, 3 seeds, grouped to 8 organs):**
MorphoST-V3 (**4.8M params**) reaches **0.427**, matching/edging the far more complex flow-matching
state-of-the-art STFlow (**0.425**, reproduced on the identical UNI pipeline) and beating every
lightweight baseline on all 8 organs. See the paper in [`paper/`](paper/main.tex).

> "Morphology-**aware**, not morphology-**conditioned**": V3 fuses local + global morphology context
> but uses **no** explicit FiLM/adaLN conditioning stage — the ablation (V4/V5) shows conditioning is
> redundant once that context is present.

## The staged model (`morphost.py`)

`--version` selects which context components are on. **V3 is the recommended / default model.**

| Version | Adds | Local | Global | Slide token | Morph-cond | Gate |
|---|---|:-:|:-:|:-:|:-:|:-:|
| V0 | UNI → MLP (image-only floor) | | | | | |
| V1 | + global self-attention | | ✓ | | | |
| V2 | + local kNN distance-biased attn | ✓ | ✓ | | | |
| **V3** | **+ global slide token  ← use this** | **✓** | **✓** | **✓** | | |
| V4 | + morphology conditioning (adaLN) | ✓ | ✓ | ✓ | ✓ | |
| V5 | + local/global adaptive gate | ✓ | ✓ | ✓ | ✓ | ✓ |

V3 config: `dim=256`, `n_layers=4`, `n_heads=4`, `k=8` neighbours, 16 RBF bases.
**Loss:** `MSE + 0.5·(1 − mean per-gene Pearson)` (metric-aligned).

## Requirements

Python 3.10+, `torch`, `numpy`, `pandas`, `scipy`, and the **STFlow package on `PYTHONPATH`** (used
only for its low-level H5/H5AD IO helpers — none of its model code):

```bash
pip install torch numpy pandas scipy
# make STFlow's IO helpers importable (repo sibling of this dir):
export PYTHONPATH=../STFlow
```

## Expected data layout (resolved by convention)

- UNI features: `<embed_dataroot>/<cohort>/uni_v1_official/fp32/<sample_id>.h5`
- expression:   `<source_dataroot>/<cohort>/adata/<sample_id>.h5ad`
- per-cohort protocol: `<source_dataroot>/<cohort>/splits/{train,test}_<i>.csv` + `var_50genes.json`
- cross-organ protocol: `<splits_root>/<regime>/splits/{train,test}_<fold>.csv` + `genes_<fold>.json`

## How to run V3

### A) Per-cohort HEST (STFlow's protocol → the paper's Table 1)

Train + evaluate V3 on one cohort (within-organ k-fold CV), one seed:

```bash
PYTHONPATH=../STFlow python train_hest.py \
    --version V3 --seed 1 --cohort PRAD \
    --source_dataroot ../dataset --embed_dataroot ../embed_dataroot \
    --feature_encoder uni_v1_official --save_root results_hest --device cuda
```

`--cohort all` runs all 10 HEST cohorts (CCRCC, COAD, READ, HCC, IDC, LYMPH_IDC, LUNG, PAAD, PRAD,
SKCM). Repeat for `--seed {1,2,3}`. Writes `results_hest/<COHORT>_V3_seed<s>/results_kfold.json`
(`pearson_mean` per cohort). SLURM: `sbatch morphost_hest.sbatch` (10 cohorts × 3 seeds).

Build the 8-organ paper table (mean±sd over seeds, MorphoST vs baselines):

```bash
python aggregate_hest_organ.py      # console table
python emit_table1.py               # LaTeX body for paper/main.tex (Table 1)
```

### B) Cross-organ transfer (LOOO / POOLED)

Leave-one-organ-out (train on 7 organs, test on the held-out one) or pooled leave-patient-out:

```bash
PYTHONPATH=../STFlow python train.py \
    --version V3 --regime LOOO --seed 1 \
    --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
    --embed_dataroot ../embed_dataroot --save_root results_morphost --device cuda
```

`--regime POOLED` for the pooled leave-patient-out setting. SLURM: `sbatch morphost.sbatch`
(the V0–V5 ablation array) or `sbatch morphost_v3.sbatch`.

## Reproducing the ablation

`morphost.sbatch` sweeps `--version V0..V5` (LOOO, seed 1) — reproduces the "local + global is
sufficient; conditioning (V4/V5) does not help" result. `morphost_recover.sbatch` holds the recovery
experiments that confirm the negative morphology-conditioning result.

## Files

| File | Purpose |
|---|---|
| `morphost.py` | model (`MorphoST`, `MorphoBlock`, local/global attention, RBF), `morphost_loss` |
| `data.py` | UNI feature / expression loading (per-cohort and cross-organ) |
| `train_hest.py` | **per-cohort HEST** train/eval (Table 1) |
| `train.py` | **cross-organ** LOOO/POOLED train/eval |
| `aggregate_hest.py` / `aggregate_hest_organ.py` | 10-cohort / 8-organ result tables |
| `emit_table1.py` | emits the LaTeX body of paper Table 1 from results |
| `paper/` | WACV 2027 paper (`main.tex`, compiles with `tectonic main.tex`) |

## Citation

See [`paper/main.tex`](paper/main.tex): *"MorphoST: A Simple E(2)-Invariant Local–Global Transformer
Matches Flow Matching for Histology-to-Expression Prediction."*
