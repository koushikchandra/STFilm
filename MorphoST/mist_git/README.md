# MIST: Multi-Scale Spatial Context for Molecular Prediction from Histology

Reference implementation of **MIST**, a MultI-context Spatial Transformer that predicts
spatial gene expression from frozen H&E patch features and spot coordinates. On top of a
standard global self-attention backbone, MIST adds two context pathways applied in every
layer:

- a **local** pathway — content-based attention over each spot's spatial *k*-nearest
  neighbors (coordinates enter only through pairwise distances, so this pathway is invariant
  to translation, rotation, and reflection of the coordinate frame), and
- a **slide** pathway — an attention-pooled whole-slide summary broadcast to every spot.

The model predicts all spots of a slide in a single forward pass.

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate     # or conda
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch 2.x on a CUDA GPU (CPU works but is slow).

The code is **self-contained**: `stflow_utils.py` vendors the only external IO helpers
needed (HDF5 / AnnData readers), so no additional project packages are required.

---

## 2. Expected data layout

MIST consumes frozen patch embeddings, log-normalized expression, split CSVs, and gene
panels, resolved by convention (not config):

```
<embed_dataroot>/<COHORT>/<feature_encoder>/fp32/<sample_id>.h5     # spot features + coords + barcodes
<source_dataroot>/<COHORT>/adata/<sample_id>.h5ad                   # expression (AnnData)
<source_dataroot>/<COHORT>/<gene_list>.json                        # e.g. var_50genes.json -> {"genes": [...]}
<source_dataroot>/<COHORT>/splits/train_<i>.csv, test_<i>.csv      # intra-cohort folds
```

- Each `.h5` holds `embeddings`, `coords`, and `barcode` datasets for the spots of one slide.
- Split CSVs have columns `sample_id, patches_path, expr_path`; fold count is inferred as
  `len(splits)//2`.
- Gene-panel JSON is `{"genes": [...]}`. Common panels: `var_10genes`, `var_50genes` (HVG),
  `deg_50genes`, `hmhvg_50genes`, `var_100genes`, `var_200genes`.
- For cross-organ runs, a `splits_root/<REGIME>/` directory holds `train_<fold>.csv`,
  `test_<fold>.csv`, and `genes_<fold>.json` (leakage-safe per-fold panels).

Features are dataset-agnostic; the pipeline follows the public
[HEST-1k](https://huggingface.co/datasets/MahmoodLab/hest) benchmark conventions. Extract
patch features (e.g. with frozen UNI and/or CONCH encoders) into the layout above before
training.

---

## 3. Training

### Intra-cohort (`train_hest.py`)
Train and test within one cohort, over its k-fold splits.

```bash
python train_hest.py \
  --cohort SKCM --version V3 \
  --feature_encoder uni_conch \
  --gene_list var_50genes.json \
  --source_dataroot /path/to/dataset \
  --embed_dataroot  /path/to/embed_dataroot \
  --save_root results_intra --seed 1
```

### Cross-organ (`train.py`) — pooled or leave-one-organ-out
```bash
# Pooled multi-organ (held-out slides)
python train.py --regime POOLED --version V3 --components 111 \
  --feature_encoder uni_conch \
  --splits_root /path/to/cross_organ_splits \
  --source_dataroot /path/to/dataset \
  --embed_dataroot  /path/to/embed_dataroot \
  --save_root results_pooled --seed 1

# Leave-one-organ-out (train on all but one organ, test on the held-out organ)
python train.py --regime LOOO --version V3 --components 111 \
  --feature_encoder uni_conch \
  --splits_root /path/to/cross_organ_splits \
  --source_dataroot /path/to/dataset \
  --embed_dataroot  /path/to/embed_dataroot \
  --save_root results_looo --seed 1
```

**Full MIST** uses `--version V3` (equivalently `--components 111`, the L/G/S bit mask
`[Local, Global, Slide]`). Reported models use `--dim 256 --n_layers 4 --n_heads 4 --k 8
--num_rbf 16 --lr 1e-3 --weight_decay 0.01 --epochs 100 --patience 20 --corr_weight 0.5`,
averaged over seeds 1, 2, 3.

---

## 4. Ablations

`morphost_context.py` exposes the full **L/G/S factorial** and the neighborhood control,
driven by `--config` in the context trainers (`train_context.py`,
`train_context_pooled.py`):

| `--config`      | Local | Global | Slide | Neighbors    |
|-----------------|:-----:|:------:|:-----:|--------------|
| `global`        |       |   ✓    |       | —            |
| `local`         |   ✓   |        |       | spatial kNN  |
| `slide`         |       |        |   ✓   | —            |
| `global_local`  |   ✓   |   ✓    |       | spatial kNN  |
| `global_slide`  |       |   ✓    |   ✓   | —            |
| `local_slide`   |   ✓   |        |   ✓   | spatial kNN  |
| `full`          |   ✓   |   ✓    |   ✓   | spatial kNN  |
| `global_random` |   ✓   |   ✓    |       | **random-k** |
| `full_random`   |   ✓   |   ✓    |   ✓   | **random-k** |

```bash
python train_context_pooled.py --regime POOLED --config full --seed 1 \
  --feature_encoder uni_conch \
  --splits_root /path/to/cross_organ_splits \
  --source_dataroot /path/to/dataset --embed_dataroot /path/to/embed_dataroot \
  --save_root results_ablation
```

The `full` vs `full_random` comparison isolates the value of **spatially defined**
neighborhoods; the single-stream configs isolate each pathway's contribution.

Equivalently, `train_hest.py --components <LGS>` runs the same factorial intra-cohort
(e.g. `--components 010` = global-only backbone, `--components 111` = full model).

---

## 5. Outputs

Each run writes, under `<save_root>/<TAG>/`:

- `fold_<f>_results.json` — per-fold metrics (per-gene and mean Pearson, MSE, MAE, ...),
- `results_kfold.json` — aggregate across folds,
- `test_predictions.npz` — predicted vs. measured expression and coordinates,
- `fold_<f>/best_model.pt` — the selected checkpoint (best inner-validation PCC).

---

## 6. Files

| File | Role |
|------|------|
| `morphost.py` | MIST model (local kNN + global + slide-pool layer, kNN graph, loss) |
| `morphost_context.py` | L/G/S factorial + random-neighbor variants (`CONFIGS`) |
| `morphost_random.py` | random-k neighbor graph (ablation control) |
| `morphost_count.py`, `morphost_noslide.py` | count-head / no-slide variants |
| `train_hest.py` | intra-cohort trainer |
| `train.py` | cross-organ (POOLED / LOOO) trainer |
| `train_context.py`, `train_context_pooled.py` | ablation trainers |
| `data.py` | feature/expression/split loading |
| `stflow_utils.py` | vendored HDF5 / AnnData IO helpers (self-contained) |
| `evaluation.py` | metrics, prediction saving, train/val split |

---

## 7. Citation

If you use this code, please cite the accompanying paper (see the submission).
