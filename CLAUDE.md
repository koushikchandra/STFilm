# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for predicting **spatial transcriptomics (gene expression) from H&E histology images**. The active research thread is adding **FiLM / adaLN conditioning** (especially categorical metadata: organ, sequencing platform, cancer subtype) to accepted H&E→spatial-transcriptomics models so a single model generalizes across organs/platforms instead of being trained per-dataset. See `FiLM_ideas.md` for the full design space, target papers (Stem, STFlow, STAMP, HistoPrism), and the validity rule (a FiLM conditioner is only usable if available at *inference* — histology embeddings and metadata are; gene expression is the target and is not).

The top-level directory is **not** a git repo; the `STFlow/` subdirectory is a separate (shallow) clone of the upstream STFlow repo that has been locally modified.

## Layout

- `STFlow/` — STFlow flow-matching model, cloned and modified to add FiLM. This is where the core model lives.
- `make_cross_organ_splits.py` — generates the cross-organ evaluation splits (described below) in STFlow's split format.
- `fetch_genomics.py` — scrapes ICLR 2025/2026 accepted papers for genomics terms via the OpenReview API; emits `iclr_genomics_papers.{csv,md}` and `_genomics_full.pkl`.
- `FiLM_ideas.md` — the running design doc and concrete diffs for the FiLM work.

## Commands

Install the STFlow package (editable):
```
cd STFlow && pip install -e .
```
Dependencies are not pinned by setup.py. Needs: `torch`, `torch_geometric`, `timm`, `einops`, `scanpy`, `pandas`, `numpy`, `scipy`, `huggingface_hub`, `wandb`, `openreview` (for `fetch_genomics.py`). `timm` in particular has had to be installed manually.

**Stage 1 — extract patch features** (run once per encoder; writes `.h5` embeddings to `embed_dataroot`):
```
python STFlow/stflow/app/hest/benchmark.py \
    --datasets all --encoders uni_v1_official \
    --weights_root /path/to/weights_root \
    --source_dataroot /path/to/source_dataroot \
    --embed_dataroot /path/to/embed_dataroot --batch_size 128
```

**Stage 2 — train STFlow** (consumes the extracted features):
```
python STFlow/stflow/app/flow/train.py \
    --datasets LUNG --feature_encoder uni_v1_official \
    --source_dataroot /path/to/source_dataroot \
    --embed_dataroot /path/to/embed_dataroot \
    --batch_size 2 --n_layers 4 --n_sample_steps 5
```
`--datasets all` expands to the 10 HEST-bench cohorts. Per-dataset k-fold is driven by the CSV files in `source_dataroot/<dataset>/splits/`; `train.py` reads `train_{i}.csv`/`test_{i}.csv` and infers fold count as `len(splits)//2`.

**Generate cross-organ splits** (for the generalization experiments):
```
python make_cross_organ_splits.py --hest_root /path/to/hest_root --out_root /path/to/out
```
Emits `LOOO/` (leave-one-organ-out, tests transfer) and `POOLED/` (pooled leave-patient-out, tests metadata benefit) plus leakage-safe `genes_<fold>.json` panels. See the module docstring for the regimes and the gene-panel leakage rule.

There is no test suite, linter, or build step.

## Expected data layout

`benchmark.py` / `train.py` resolve paths by convention, not config:
- features: `embed_dataroot/<dataset>/<feature_encoder>/fp32/<sample_id>.h5`
- expression: `source_dataroot/<dataset>/adata/<sample_id>.h5ad`
- gene panel: `source_dataroot/<dataset>/<gene_list>` (default `var_50genes.json`, 50 genes)
- splits: `source_dataroot/<dataset>/splits/train_<i>.csv`, `test_<i>.csv` (columns `sample_id, patches_path, expr_path`)

The 10 HEST-bench cohorts (grouped to organ in `make_cross_organ_splits.py` to prevent tissue leakage): CCRCC(kidney), COAD+READ(colorectal), HCC(liver), IDC+LYMPH_IDC(breast), LUNG, PAAD(pancreas), PRAD(prostate), SKCM(skin).

## Model architecture (STFlow + local FiLM additions)

Flow-matching generative model. `Interpolant` (`flow/interpolant.py`) linearly interpolates a ZINB/gaussian prior sample toward the true expression; the `Denoiser` predicts the clean expression and inference integrates the velocity field over `n_sample_steps`.

The denoiser backbone is a **custom geometric `SpatialTransformer`** (`model/transformer.py`), NOT a DiT. It builds a kNN graph over spot coordinates and uses **frame averaging (`model/fa.py`) for SE(2)-equivariance** over the 2D spatial layout. This equivariance is load-bearing for the FiLM work:

- FiLM/adaLN was **added locally** (upstream STFlow has none). `TransformerBlock` has an `adaLN-Zero` head; the `SpatialTransformer` threads a `cond` conditioner; `MetadataEmbedder` (in `denoiser.py`) turns categorical metadata into that conditioner, with **index 0 reserved as a null token** (zero-initialized) for classifier-free guidance and unseen categories.
- **Equivariance guardrail:** FiLM may only modulate the *invariant scalar token stream* (`token_embs`) using *invariant* conditioners (image embedding, timestep, metadata). It must NOT touch the frame/edge geometric features. The current code respects this by gating only the geometric attention's *output* and applying full shift/scale only on the scalar MLP path. Preserve this when editing — breaking it breaks equivariance.
- adaLN is zero-initialized so every block starts as identity (conditioning dormant at step 0, training-stable).
- The model side already accepts a `meta` dict; wiring metadata through the dataset loader and training loop is the outstanding work.

## Known upstream STFlow bugs (worked around, not upstream-fixed)

- `GeneUpdate.__init__` originally didn't accept the `non_negative` kwarg that `TransformerBlock` passes it — would `TypeError` on instantiation. Locally accepted as a no-op kwarg.
- `MLPAttnEdgeAggregation` hardcodes `+50` (the gene-panel size) in its attention input dim, so the model only works with `n_genes == 50`. The default panel is 50, so this holds — but any change to panel size requires fixing this.
