# EXPERIMENT_LOG.md

Reproducibility + methods record for the **FiLM-conditioning study on STFlow**
(H&E histology → spatial gene-expression prediction).

**Goal of the study:** take STFlow (ICML 2025 flow-matching model) as the baseline and measure,
in clean matched-seed ablations, whether adding **FiLM / adaLN-Zero conditioning** improves
gene-expression prediction — first plain FiLM (Stage 2), then categorical **organ metadata**
FiLM for cross-organ generalization (Stage 3).

Encoder used throughout: **`resnet50_trunc`** (1024-d, ImageNet timm, non-gated) — chosen so no
gated-HuggingFace login (UNI/GigaPath) is needed. The V0/V1/V2 deltas are encoder-agnostic.

The three FiLM variants (one `--film` flag, see Stage 0):
- **V0** `--film none`  — upstream STFlow baseline: histology+time injected only at the input layer.
- **V1** `--film context` — FiLM re-injects img+time via adaLN at **every** layer.
- **V2** `--film meta`   — V1 **plus** a categorical organ embedding added to the conditioner.

---

## Stage 0 — Add a FiLM on/off switch (code)

Made V0/V1/V2 selectable so ablations are clean and seed-matched. FiLM was added *locally*;
upstream STFlow has none.

- **`STFlow/stflow/app/flow/train.py`** — added `--film {none,context,meta}` (default `context`);
  persisted into each per-dataset `config.json`. Added `resnet50_trunc:1024` to the feature_dim map.
- **`STFlow/stflow/model/denoiser.py`** — `Denoiser.inference` reads `self.film_mode`
  (`= getattr(config,"film","context")`): `none` → `cond=None`; else `cond = img+time`; and if
  `meta_categories` is set and `meta` is passed, `cond += MetadataEmbedder(meta)`.
  `MetadataEmbedder` reserves **index 0 as a zero-init null token** (CFG / unseen category).
- **`STFlow/stflow/model/transformer.py`** — `TransformerBlock` adaLN-Zero head; the `c is None`
  branch is the V0 path; adaLN is **zero-initialized** so every block starts as identity.

**Equivariance guardrail (do not break):** FiLM may only modulate the *invariant scalar token
stream* using *invariant* conditioners (image embedding, timestep, metadata). It must NOT touch the
frame/edge geometric features (frame-averaging gives SE(2)-equivariance). Current code gates only
the geometric attention's *output* and applies shift/scale only on the scalar MLP path.

---

## Stage 1 — Data acquisition + feature extraction

- **Cohorts (6 organs):** `LUNG` (lung), `HCC` (liver), `CCRCC` (kidney), `PAAD` (pancreas),
  `PRAD` (prostate), `SKCM` (skin). Downloaded from the **public** `MahmoodLab/hest-bench` via
  `snapshot_download(allow_patterns=[...])` into `dataset/`.
- **Feature extraction — `extract_features.py`** (working-dir script): builds `resnet50_trunc`
  inline (avoids `encoder_wrappers`, which imports `transformers`), reuses STFlow's `embed_tiles`
  + `H5TileDataset`, and **skips** `benchmark.py`'s `__main__` (which would re-download the full
  ~42 GB + gated UNI/GigaPath and run an unneeded random-forest probe).
  Output: `embed_dataroot/<cohort>/resnet50_trunc/fp32/<sample>.h5`
  with `{embeddings[N,1024], coords, barcodes}`.

### Environment fixes to run upstream STFlow in this conda env (all import guards / casts)
- `app/flow/train.py`: optional `wandb` import; `resnet50_trunc:1024` in feature_dim map.
- `utils/utils.py`: optional `mygene`.   `data/normalize_utils.py`: optional `scprep`.
- `flow/noise.py`: **pure-torch ZINB fallback** (`zinb_prior`) when `scvi` is absent — matches
  scvi's (total_count, logits) NB + Bernoulli zero-inflation gate.
- `app/flow/test.py`: cast per-gene `pearson_corr` to `float` (numpy float32 isn't JSON-serializable).

---

## Stage 2 — V0 vs V1 ablation (does FiLM help at all?)

**Protocol.** For each cohort, train twice with identical seed/splits: `--film none` (V0) and
`--film context` (V1). All other hyperparameters at STFlow defaults (50-gene panel, 4 layers, ZINB
prior, flow-matching). Repeat for **seeds 1, 2, 3**. Metric: `pearson_mean` over k-folds from
`results_kfold.json`.

**Run scripts:** `run_ablation2.sh` (main sweep), `run_ablation2_resume.sh` (seed-3 resume after a
node failure — see "Incident" below). Results under `results_ablation/<exp_code>/<cohort>/`.
Aggregator: **`aggregate_ablation.py`** (globs `results_ablation/*/*/config.json` +
`results_kfold.json`, groups by (cohort, film, seed), reports V0/V1 mean±std and delta).

### Result (3 seeds, higher Pearson = better)

| Cohort | Organ | V0 (no FiLM) | V1 (FiLM) | Mean Δ |
|--------|-------|--------------|-----------|--------|
| PRAD  | prostate | 0.2229 ± 0.022 | 0.3089 ± 0.051 | **+0.086 ± 0.068** |
| HCC   | liver    | 0.0187 ± 0.001 | 0.0967 ± 0.007 | **+0.078 ± 0.007** |
| PAAD  | pancreas | 0.3497 ± 0.007 | 0.4250 ± 0.004 | **+0.075 ± 0.010** |
| CCRCC | kidney   | 0.0995 ± 0.001 | 0.1672 ± 0.027 | **+0.068 ± 0.027** |
| LUNG  | lung     | 0.5369 ± 0.014 | 0.5672 ± 0.003 | **+0.030 ± 0.012** |
| SKCM  | skin     | 0.5565 ± 0.008 | 0.5734 ± 0.007 | **+0.017 ± 0.008** |

**FiLM helped 6/6 cohorts; mean Δ = +0.059 ± 0.026.** Gains largest on hard/low-baseline organs
(prostate, liver, pancreas), smallest on easy high-baseline organs (skin, lung). → V1 is carried
forward as the **baseline** for Stage 3.

### Incident: GPU/fabric-manager failure (recovered)
Mid-sweep, node `nova22-amp-7` threw **CUDA error 802 "system not yet initialized"** — root cause
was the `nvidia-fabricmanager` service in a `failed` state (A100-SXM4 needs it; requires root, not
our code). Recovery: built `gpu_ready.py` (real readiness test — actually allocates a CUDA tensor,
not just `torch.cuda.is_available()`) and `auto_resume.sh` (polls `gpu_ready.py` every 60 s, then
launches the resume). Resumed seed-3 on a healthy node (`nova22-amp-6`) via `run_ablation2_resume.sh`.

---

## Stage 3 — Metadata FiLM (V2) cross-organ generalization

**Question.** Treating V1 (FiLM, histology-only) as baseline: does also telling a **single shared
multi-organ model** which organ each patch is from (via FiLM) help? Two regimes:

- **POOLED** — leave-patient-out, all 6 organs seen in training (5 folds). Tests the *metadata
  benefit* (organ id is always a known token at test).
- **LOOO** — leave-one-organ-out (6 folds: train on 5 organs, test on the held-out 6th). Tests
  *transfer* to an unseen organ; the held-out organ gets the **null token (id 0)** at test.

**Organ vocab:** `0 = null`, then `kidney=1, liver=2, lung=3, pancreas=4, prostate=5, skin=6`
→ `meta_categories = {"organ": 6}` (MetadataEmbedder builds `nn.Embedding(7, hidden)`).

### Code added/changed
- **`STFlow/stflow/model/transformer.py` — `+n_genes` fix.** `MLPAttnEdgeAggregation` hardcoded
  `+50` (the gene-panel size) in its attention input dim, so the model only worked at `n_genes==50`.
  Added an `n_genes` constructor param, replaced both `+50` with `+n_genes`, and threaded it from
  `TransformerBlock` (already fed `n_genes=config.n_genes` by `SpatialTransformer`). Harmless at 50;
  required because cross-organ panels are smaller (see below).
- **`train_cross_organ.py`** (new, working-dir): self-contained cross-organ trainer. Reuses STFlow
  `Denoiser`/`Interpolant`/`SPData`. Loads the `cross_organ_splits/{POOLED,LOOO}` CSVs; derives the
  organ id from each row's cohort prefix (`patches_path.split("/")[0]` → `COHORT_TO_GROUP` → vocab);
  threads a `{"organ": LongTensor}` meta dict through a custom batcher → `model(..., meta=...)`.
  - **CFG meta-dropout** (`--meta_dropout 0.1`): in V2 training, randomly replace organ id with the
    null token 10% of the time so the model can run unconditionally (needed for LOOO's null token).
  - Per-fold eval (`evaluate`) does the flow-matching integration and computes Pearson; LOOO eval
    forces the null organ for the unseen test organ (`force_null_organ=True`).
- **`run_stage3.sh`** (new): sweep `seed ∈ {1,2,3} × regime ∈ {POOLED,LOOO} × film ∈ {context,meta}`
  = 12 matched runs → `results_cross_organ/<REGIME>_<film>_seed<seed>/`. Log: `stage3.log`.

### Splits — `cross_organ_splits/` (from `make_cross_organ_splits.py`)
`POOLED/` (train_0..4 / test_0..4) and `LOOO/` (train_<organ> / test_<organ>), each with a
`genes_<fold>.json` panel and the cohort→organ grouping that prevents tissue leakage.

### Data fix: gene-panel narrowing (important)
The cross-organ union panels (39 genes) included genes **absent from some samples' `adata`** —
and availability varies **per sample**, not just per cohort (e.g. CCRCC sample `INT24` lacks genes
that `INT1` has; SKCM/LUNG/PAAD are ~540-gene HEST panels missing many union genes). `load_adata`'s
`adata[:, genes]` then `KeyError`s. **Fix:** narrowed each fold's panel to genes present in *every*
participating train+test sample (panel selection by availability is not expression leakage). Result:
**POOLED → 19 genes; LOOO → 19 (kidney/liver/prostate/skin), 17 (lung), 12 (pancreas)**. Originals
backed up as `cross_organ_splits/**/genes_*.json.orig`. The `+n_genes` fix makes these sizes work.

### Smoke tests (all passed before the sweep)
POOLED-meta, LOOO-meta (null-token path), and POOLED-context — each trained 1 epoch and evaluated
cleanly on GPU.

### Results — STATUS: all 3 seeds complete (12/12 configs)
Aggregation rule (matched, like Stage 2): compare V1 vs V2 on the **same folds/seeds**; report
`pearson_mean` and the V2−V1 delta.

**Seed-1 POOLED — all organs seen (metadata-benefit test):**

| Fold | V1 (context) | V2 (meta) | Δ |
|------|--------------|-----------|---|
| 0 | 0.7258 | 0.7303 | +0.0046 |
| 1 | 0.7297 | 0.7445 | +0.0148 |
| 2 | 0.5749 | 0.5937 | +0.0188 |
| 3 | 0.2460 | 0.2747 | +0.0287 |
| 4 | 0.2170 | 0.2197 | +0.0028 |
| **mean** | **0.4987** | **0.5126** | **+0.0139** |

**Seed-1 LOOO — held-out organ unseen, null token at test (transfer test):**

| Fold (held-out organ) | V1 (context) | V2 (meta) | Δ |
|------|--------------|-----------|---|
| kidney   | 0.0338 | 0.0264 | −0.0074 |
| liver    | 0.0856 | 0.0635 | −0.0222 |
| lung     | 0.6611 | 0.6554 | −0.0057 |
| pancreas | 0.5103 | 0.4979 | −0.0124 |
| prostate | 0.0527 | 0.0610 | +0.0083 |
| skin     | 0.6547 | 0.6510 | −0.0037 |
| **mean** | **0.3331** | **0.3259** | **−0.0072** |

**Interpretation (seed 1).** *In-distribution organ metadata helps; it does not transfer to unseen
organs.* POOLED: V2 > V1 on **5/5** folds (+0.014), larger on harder low-baseline folds. LOOO:
V2 < V1 on **5/6** folds (−0.007) — V2 learns to lean on the organ signal, but at test the held-out
organ has only the null token, so it gains nothing (slightly less). The honest narrative is that
metadata FiLM is a **benefit when the organ is in-distribution, not a generalization tool for unseen
organs**. Effects are small and LOOO test sets are tiny (lung/liver/skin = 2 samples).

**Final 3-seed summary (pearson_mean, V2−V1 delta):**

| Regime | Seed | V1 (context) | V2 (meta) | Δ (V2−V1) |
|--------|------|--------------|-----------|-----------|
| POOLED | 1 | 0.4987 | 0.5126 | +0.0139 |
| POOLED | 2 | 0.5062 | 0.5136 | +0.0074 |
| POOLED | 3 | 0.5013 | 0.5097 | +0.0084 |
| **POOLED** | **mean** | **0.5021 ± 0.0031** | **0.5120 ± 0.0017** | **+0.0099 ± 0.0029** |
| LOOO | 1 | 0.3331 | 0.3259 | −0.0072 |
| LOOO | 2 | 0.3252 | 0.3351 | +0.0099 |
| LOOO | 3 | 0.3336 | 0.3217 | −0.0119 |
| **LOOO** | **mean** | **0.3306 ± 0.0038** | **0.3275 ± 0.0056** | **−0.0031 ± 0.0094** |

**Cross-seed conclusion.** POOLED: V2 > V1 on **all 3 seeds**, Δ = **+0.0099 ± 0.0029** (Δ std ≪ mean
→ a robust in-distribution gain from organ metadata). LOOO: Δ = **−0.0031 ± 0.0094**, sign flips across
seeds → **statistically indistinguishable from zero**, confirming the seed-1 read that metadata FiLM
does **not** transfer to unseen organs (held-out organ falls back to the null token at test).

## Stage 3b — V3 (film=desc): continuous histology descriptor instead of a categorical organ tag

**Motivation.** Stage 3's negative LOOO result is *architectural*: a categorical organ token has nothing
for an unseen organ — it collapses to the zero null token, i.e. no conditioning at all (a "null-token
cliff"). V3 replaces it with a **continuous slide-level descriptor**: the masked mean-pool of the patch
embeddings, projected to hidden size and added to the per-layer conditioner. It is computed *inside*
`Denoiser.inference` from the image features already passed in (no loader change), so it is **always
available at inference, including for unseen organs** — it degrades gracefully instead of going null.

**Code.** `denoiser.py`: `film_mode=="desc"` adds `self.desc_proj = nn.Linear(feature_dim, hidden)`;
`inference` computes `desc = masked_mean(raw_img); cond = features + desc_proj(desc)`. `train_cross_organ.py`:
`--film desc` (no meta, no CFG dropout). Same matched seeds/splits/panels as V1/V2.

**Final 3-seed results (pearson_mean), V1 (context) vs V2 (meta) vs V3 (desc):**

| Regime | V1 (context) | V2 (meta) | V3 (desc) | V3−V2 | V3−V1 |
|--------|--------------|-----------|-----------|-------|-------|
| POOLED (in-distribution) | 0.5021 ± 0.0031 | 0.5120 ± 0.0017 | **0.5147 ± 0.0018** | +0.0028 ± 0.0032 | +0.0126 ± 0.0031 |
| LOOO (transfer, unseen organ) | 0.3306 ± 0.0038 | 0.3275 ± 0.0056 | **0.3383 ± 0.0014** | **+0.0108 ± 0.0053** | **+0.0077 ± 0.0042** |

Per-seed LOOO V3−V2: +0.0142, +0.0033, +0.0150 (positive on **all 3 seeds**). Per-seed LOOO V3−V1:
+0.0070, +0.0132, +0.0030 (positive on **all 3 seeds**). Per-organ, the gains concentrate on the
**low-baseline organs where the null cliff hurt most** (e.g. liver V3−V2 ≈ +0.05–0.06 each seed; lung,
an easy high-Pearson organ, ≈ flat).

**Conclusion.** Where the categorical tag V2 *failed* to transfer (V2−V1 = −0.0031 ± 0.0094, a
sign-flipping null), the continuous descriptor V3 **reverses it into a consistent positive transfer
gain** (V3−V2 = +0.0108 ± 0.0053, V3−V1 = +0.0077 ± 0.0042; Δ std < mean for both). V3 is also the most
stable model (lowest LOOO std, 0.0014) and **loses nothing in-distribution** (POOLED: ties/edges V2,
beats V1). This is the intended contribution: *a histology-derived continuous tissue descriptor matches
categorical metadata when the organ is seen and outperforms it on unseen organs.*

### Stage 3 caveats
- **Directional proof-of-concept, not a benchmark.** Only 6 single-cohort organs; some LOOO test
  sets are tiny (lung/liver/skin have 2 samples each) → noisy transfer signal.
- Smaller gene panels (12–19) than Stage 2 (50) → **Stage 3 numbers are NOT comparable to Stage 2's**;
  only within-stage deltas are meaningful.

## Stage 3c — UNI-encoder replication (credibility)

Re-ran the full V1/V2/V3 × POOLED/LOOO × 3-seed sweep on **UNI (`uni_v1_official`, ViT-L/16)** features
instead of resnet50_trunc, matched seeds/splits/panels. Purpose: show the conditioner story is not an
artifact of a weak ImageNet encoder. Features extracted with `extract_features_uni.py`; sweep run
2-GPU-sharded via `run_uni_shard.sh` (one shard per A100, `OMP_NUM_THREADS=5` each to avoid CPU
oversubscription on the shared node; per-fold resume added to `train_cross_organ.py`).

**Final 3-seed results (pearson_mean), UNI:**

| Regime | V1 (context) | V2 (meta) | V3 (desc) | V3−V2 | V3−V1 |
|--------|--------------|-----------|-----------|-------|-------|
| POOLED (in-distribution) | 0.5233 ± 0.0019 | 0.5283 ± 0.0012 | **0.5343 ± 0.0038** | +0.0060 | +0.0111 |
| LOOO (transfer, unseen organ) | 0.3659 ± 0.0014 | 0.3686 ± 0.0028 | **0.3745 ± 0.0004** | +0.0059 | +0.0087 |

**Head-to-head transfer (LOOO Δ vs V1 context), resnet vs UNI:**

| | resnet50_trunc | UNI |
|---|---|---|
| V2 (meta) − V1 | **−0.0031** | +0.0027 |
| V3 (desc) − V1 | **+0.0077** | **+0.0087** |

**Conclusion.** The headline result replicates with a SOTA encoder: **V3 (continuous descriptor) is the
best conditioner in all four cells** (both encoders × both regimes), its transfer gain is consistent
(+0.0077 resnet, +0.0087 UNI, UNI std 0.0004), and **V3 beats V2 on transfer regardless of encoder**
(+0.0108 resnet, +0.0059 UNI). One encoder-dependent nuance: the categorical token V2 *hurts* transfer
with resnet (−0.0031) but is roughly neutral with UNI (+0.0027) — so the robust, encoder-independent
claim is "the categorical organ tag gives unreliable/negligible transfer benefit while the continuous
descriptor reliably improves transfer (and wins in-distribution)," rather than "V2 always causes
negative transfer." The two-encoder replication strengthens the contribution.

---

## Reproduce

```bash
# 0. install STFlow (editable) + deps (torch, torch_geometric, timm, einops, scanpy, ...)
cd STFlow && pip install -e . && cd ..

# 1. extract resnet50_trunc features for the 6 cohorts  (writes embed_dataroot/)
PYTHONPATH=STFlow python extract_features.py

# 2. Stage 2 ablation (V0 vs V1), 3 seeds, then aggregate
PYTHONPATH=STFlow bash run_ablation2.sh
PYTHONPATH=STFlow python aggregate_ablation.py

# 3. Stage 3 cross-organ splits + V1-vs-V2 sweep (POOLED + LOOO, 3 seeds)
python make_cross_organ_splits.py --hest_root dataset --out_root cross_organ_splits
#    (then narrow each genes_<fold>.json to per-fold sample availability — see Stage 3 data fix)
PYTHONPATH=STFlow bash run_stage3.sh        # -> results_cross_organ/, log: stage3.log
```

Single Stage 3 run, e.g. V2 POOLED seed 1:
```bash
PYTHONPATH=STFlow python train_cross_organ.py --regime POOLED --film meta --seed 1 --device 0
```

## Key files
| File | Role |
|------|------|
| `STFlow/stflow/model/denoiser.py` | FiLM switch + `MetadataEmbedder` (null token at idx 0) |
| `STFlow/stflow/model/transformer.py` | adaLN-Zero block; `+n_genes` fix |
| `STFlow/stflow/app/flow/train.py` / `test.py` | `--film` flag; Stage 2 train/eval |
| `extract_features.py` | Stage 1 resnet50_trunc feature extraction |
| `run_ablation2.sh`, `aggregate_ablation.py` | Stage 2 sweep + aggregation |
| `make_cross_organ_splits.py` | LOOO + POOLED splits, leakage-safe panels |
| `train_cross_organ.py`, `run_stage3.sh` | Stage 3 cross-organ trainer + sweep |
| `gpu_ready.py`, `auto_resume.sh` | GPU-readiness test + auto-resume watcher |
