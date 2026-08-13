# MorphoST — a Simple $E(2)$-Invariant Local–Global Transformer for Spatial Transcriptomics

MorphoST predicts spatially-resolved gene expression from H&E histology by **direct regression** —
no flow matching, no ZINB prior, no frame averaging. It is **$E(2)$-invariant by construction**
(coordinates enter only through pairwise distances) and fuses three cheap sources of context per
layer: distance-biased **local kNN attention**, **global self-attention**, and a **global slide
token**. `V3` (local + global + slide token) is the recommended model — see [`paper/`](paper/main.tex).

**Headline (HEST-1k, per-cohort $k$-fold, top-50 HVG, Pearson, 3 seeds, nested validation, 8 organs):**
MorphoST-V3 (**4.75M params**) reaches **0.389** average and wins **6 of 8** organs against every
feature-matched baseline.

---

## 1. Setup

```bash
# Python 3.10+; STFlow must be importable (only its low-level H5/H5AD IO helpers are used)
pip install torch numpy pandas scipy scikit-learn
export PYTHONPATH=../STFlow          # run all commands from inside MorphoST/
```

## 2. Data layout (resolved by convention)

| What | Path |
|---|---|
| UNI features | `<embed_dataroot>/<cohort>/uni_v1_official/fp32/<sample_id>.h5` |
| expression | `<source_dataroot>/<cohort>/adata/<sample_id>.h5ad` |
| per-cohort splits | `<source_dataroot>/<cohort>/splits/{train,test}_<i>.csv` + `var_50genes.json` |
| cross-organ splits | `<splits_root>/<regime>/splits/{train,test}_<fold>.csv` + `genes_<fold>.json` |

The repo expects `../dataset` (expression + splits), `../embed_dataroot` (UNI features), and
`../cross_organ_splits8` (the leave-one-organ-out / pooled splits) as siblings of this directory.

## 3. Evaluation protocol (important)

All **publishable** numbers use **nested validation**: each outer training fold is split into an inner
15% validation set (slide-level; or a disjoint **spot-level** split for single-training-slide cohorts
— COAD/HCC/LUNG/SKCM), the checkpoint is selected on that inner set, and the outer test fold is scored
**once**. Result directories that begin with `results_corrected_` follow this protocol. Older
`results_hest*` / `results_morphost*` dirs used test-fold selection and are exploratory only — do not
report them. See [`EXPERIMENTS_WACV.md`](EXPERIMENTS_WACV.md).

---

## 4. Quick start (one cohort / one regime, single GPU)

```bash
# Per-cohort HEST (Table 1): train + eval V3 on one cohort, one seed
PYTHONPATH=../STFlow python train_hest.py --version V3 --seed 1 --cohort PRAD --corr_weight 0.5 \
    --source_dataroot ../dataset --embed_dataroot ../embed_dataroot \
    --save_root results_corrected_hest_corr --device cuda

# Cross-organ transfer (LOOO): full model = components 111
PYTHONPATH=../STFlow python train.py --regime LOOO --version V3 --components 111 --seed 1 \
    --splits_root ../cross_organ_splits8 --source_dataroot ../dataset \
    --embed_dataroot ../embed_dataroot --save_root results_corrected_factorial --device cuda
```

`--cohort all` runs all 10 HEST cohorts; `--regime POOLED` for pooled leave-sample-out.

---

## 5. Reproduce the paper (SLURM)

Each paper table maps to one array job (all a100-pinned; edit the partition/account header for your
cluster). Every script is **fold-level idempotent** — rerun to fill only what is missing.

| Paper artifact | Command | Output |
|---|---|---|
| **Table 1–2** main + 3-metric | `sbatch morphost_corrected_hest.sbatch` (MorphoST V3, MSE & MSE+PCC) and `sbatch corrected_baselines_hest.sbatch` (7 baselines) | `results_corrected_hest_{corr,mse}`, `results_corrected_hest_baselines` |
| **Ablation** (factorial L/G/S) | `sbatch morphost_factorial.sbatch` | `results_corrected_factorial` (`LOOO_C000..C111`) |
| **Loss control** | `morphost_corrected_hest.sbatch` (MSE arm) + `sbatch loss_control_baselines.sbatch` | `results_corrected_hest_baselines_corr` |
| **STImage** 2nd benchmark | `sbatch morphost_corrected_stimage.sbatch` | `results_corrected_stimage` |
| **Geometry** appendix | `sbatch morphost_geometry.sbatch` | `results_corrected_geometry` |
| **Sensitivity** appendix | `sbatch morphost_sensitivity.sbatch` | `results_corrected_sensitivity` |
| **Efficiency** (latency/mem/invariance) | `sbatch morphost_benchmark.sbatch` (or `python benchmark_model.py`) | `benchmark_morphost_a100.json` |

Single-slide cohorts (COAD/HCC/LUNG/SKCM) are handled by the same scripts via the spot-level inner
split; `rerun_smallcohort_*.sbatch` are targeted reruns for just those four.

## 6. Build the tables and figures

```bash
python emit_final.py            # LaTeX cells for Tables 1, 2, ablation, loss control (paste into paper/main.tex)
python analyze_corrected.py results_corrected_hest_corr results_corrected_hest_mse \
       ../results_corrected_hest_baselines ../results_corrected_hest_baselines_corr
python downstream_domain.py results_corrected_hest_corr --k 6      # spatial-domain recovery (ARI/NMI)
python plot_expression_maps.py results_corrected_hest_corr/CCRCC_V3_seed1/fold_0/test_predictions.npz \
       --output paper/expression_map_example.png
```

## 7. Model versions (`morphost.py`, `--version`)

| Version | Local kNN | Global | Slide token | Note |
|:--|:-:|:-:|:-:|:--|
| V0 | | | | image-only MLP floor |
| V1 | | ✓ | | + global attention |
| V2 | ✓ | ✓ | | + local kNN attention |
| **V3** | **✓** | **✓** | **✓** | **recommended / paper model** |

Config: `dim=256`, `n_layers=4`, `n_heads=4`, `k=8`, 16 RBF bases. Loss: `MSE + 0.5·(1 − mean per-gene Pearson)`.
The factorial ablation (`--components LGS`, e.g. `011`) toggles each pathway independently.

## 8. Files

| File | Purpose |
|---|---|
| `morphost.py` | model (`MorphoST`, local/global attention, RBF), `morphost_loss` |
| `data.py` | UNI feature / expression loading (incl. single-slide spot fallback) |
| `evaluation.py` | metrics, `train_val_split` (nested + spot-level fallback), prediction IO |
| `train_hest.py` / `train.py` | per-cohort HEST / cross-organ trainers |
| `emit_final.py`, `aggregate_hest_organ.py`, `analyze_corrected.py` | result aggregation → tables |
| `downstream_domain.py` | spatial-domain-recovery ARI/NMI |
| `benchmark_model.py` | latency / memory / invariance benchmark |
| `paper/` | WACV paper (`tectonic paper/main.tex`) |

## 9. Status / caveats

- The **STFlow** column of Table 1 requires a corrected STFlow reproduction (its training loop patched
  for inner-validation) and is not yet included.
- The pooled cross-organ split is sample-disjoint but not verified patient-disjoint for cohorts lacking
  patient metadata. See [`EXPERIMENTS_WACV.md`](EXPERIMENTS_WACV.md) for the full experiment audit.
