# Recent-baseline FiLM experiment log

## Phase 1: Gene-DML and FEAST

- Submitted: 2026-08-12
- Slurm job: `12002817`
- Grid: 2 models × 3 modes (`none`, `desc`, `local`) × 2 regimes (`LOOO`, `POOLED`) × 3 seeds
- Total tasks: 36
- Script: `recent_film_grid.sbatch`
- Adapter: `recent_film_baselines.py`
- Output: `results_recent_film/{REGIME}_{model}-{film}_seed{seed}/`
- Inputs: frozen UNI v1 features, `cross_organ_splits8`, training-only 50-gene panels, log1p targets
- Result schema: identical fold JSON and `results_kfold.json` format used by earlier FiLM baselines

### Fidelity

- Gene-DML adaptation preserves target, neighborhood, and global pathways with learned token fusion
  and auxiliary pathway regression.
- FEAST imports and trains the official `model/feast.py` attention architecture; only its data adapter,
  1024→hidden projection, and FiLM preconditioning are external.
- These are feature-matched COAST adaptations, not claims of reproducing the papers' native-dataset scores.

## Phase 2: STevs and FLAG

Pending faithful adapters. STevs requires Product-of-Experts probabilistic fusion and a negative-binomial
decoder. FLAG requires its graph-conditioned diffusion loop and, for the complete model, Geneformer or
scGPT embeddings. They must not be replaced by generic regression approximations.

## HyperST

Blocked because no official code release was found as of 2026-08-12.

## Fixes (2026-08-12)

- **Merge bug (all tasks reported FAILED):** `merge_fold_results` expected a `pearson_corrs` key the
  adapter never produced (`KeyError`), crashing *after* per-fold training. Replaced with a self-contained
  kfold merge (pearson mean/std, spearman, mse) matching the other baselines. Gene-DML fold results were
  intact; regenerated its `results_kfold.json` locally.
- **FEAST CUDA OOM:** FEAST's full O(N^2) attention was evaluated on uncapped test slides (~13 GiB on a
  large pooled slide). Added a deterministic per-slide eval cap (`--max_spots`, keyed by slide_id so
  none/desc/local score identical spots -> fair FiLM comparison). Deleted FEAST fold results and
  resubmitted FEAST-only, a100-pinned (job 12003352, array 18-35).
