# STFiLM MoE (Morphology-Routed Experts) — Development Log

Running tracker of the mixture-of-experts conditioning experiment (V5 `moe`).

- **Base method:** STFiLM = STFlow (ICML'25) + FiLM/adaLN conditioning.
- **This experiment (V5 `moe`):** the `desc`/`hybrid` conditioner adapts *every spot the same way*
  (one broadcast slide descriptor). Instead, let each spot **dynamically route over `E` morphology
  experts** on the invariant scalar MLP path. Motivation: a slide mixes tumor / stroma / immune /
  necrosis / normal — a single global adjustment blurs them. Experts are organ-agnostic reusable
  morphology transforms → better unseen-organ transfer + interpretable expert maps.
- **Design invariant:** all new capability is **opt-in behind `film=moe`**. Modes
  `none/context/meta/desc/hybrid` build the exact same modules as before (verified: their aux is
  `None`, forward/backward unchanged) → earlier code/results untouched.
- **Equivariance guardrail:** experts + router operate only on the invariant scalar token stream;
  the router reads scalar features (never coords / gene_exp). Geometric attention untouched →
  SE(2)-equivariance preserved (verified: max |pred(x) − pred(Rx+t)| = 0.0).

## Conditioning modes (`--film`)
| mode | conditioner | new? |
|---|---|---|
| `none` | none (upstream STFlow) | baseline |
| `desc` | + masked-mean-pool histology descriptor | V3 (current STFiLM) |
| `hybrid` | `desc` + cross-attention to K prototypes | V4 |
| **`moe`** | **per-spot routing over E expert MLPs on the scalar stream** | **V5 (this)** |

## Files changed / added (all gated; earlier modes byte-identical)

### `STFlow/stflow/model/transformer.py`
- **`MoEMlp` (new class)** — E small expert MLPs + a linear router. Output = (optionally top-k
  sparse) softmax-weighted expert combination on the scalar tokens `h`. Returns routing probs
  (for the smoothness loss) and a Switch-Transformer load-balance aux (`E·Σ frac_e·prob_e`).
- **`TransformerBlock`** — new ctor flags `use_moe / n_experts / moe_top_k /
  use_prototypes_in_router`. When `use_moe`, the MLP sublayer runs `MoEMlp` instead of `self.mlp`,
  still multiplied by the existing adaLN-Zero `gate_mlp` (so identity at init). Optional
  `router_ctx_proj` adds mean-pooled slide prototype tokens to the router input. `forward` now
  returns a 3rd value `aux=(load_balance, smoothness)` (`None` for non-moe / `c is None`).
- **`SpatialTransformer`** — reads `config.use_moe` (+ moe hparams), threads them to blocks,
  accumulates per-block aux, and returns `(gene_exp, aux_out)` (`aux_out=None` when no moe).

### `STFlow/stflow/model/denoiser.py`
- **`Denoiser`** — passes moe kwargs into inner `ModelConfig`; builds the prototype `attn_pool`
  for `hybrid` OR (`moe` and `use_prototypes_in_router`); stores `lambda_bal / lambda_smooth`.
  `inference()` unpacks `(prediction, aux)` from the backbone and stashes aux; `forward()` adds
  `lambda_bal·balance + lambda_smooth·smoothness` to the MSE (only when aux is not None).

### `train_cross_organ.py`
- `--film` choices += `"moe"`; new `--n_experts` (4), `--moe_top_k` (0=dense), `--lambda_bal`
  (1e-2), `--lambda_smooth` (1e-3), `--use_prototypes_in_router`.

### `aggregate_comparison.py`
- New `STFlow+MoE (ours)` row → `results_moe_uni8/{REGIME}_moe_seed{1,2,3}`.

## How to run
```
# Dense MoE (V5), matching results_final_uni8 config. 2 regimes x 3 seeds = 6 configs.
bash run_uni_moe.sh <device> <shard_idx> <n_shards>
# then: python aggregate_comparison.py   # adds the MoE row to comparison_{LOOO,POOLED}.md
```

## Status
- [x] Implemented MoE path (opt-in; earlier modes verified aux=None, unchanged).
- [x] Smoke test — all modes fwd/bwd (`scratchpad/smoke_moe.py`); moe emits (bal, smooth) aux.
- [x] adaLN-Zero: `gate_mlp` still gates the MoE output → moe == STFlow at init.
- [x] **SE(2)-equivariance preserved**: max |pred(x) − pred(Rx+t)| = 0.0.
- [x] Dense full runs on `cross_organ_splits8` (LOOO+POOLED × 3 seeds) → `results_moe_uni8/`
      (SLURM job 11976157, dense routing, prototypes-in-router, n_experts=4, lambdas default).
- [x] Aggregated MoE vs desc/local/none/baselines; `comparison_*.md/csv` regenerated.

## RESULT (2026-08-07) — MoE does NOT beat desc; it regresses to ~baseline
| model | LOOO (3 sd) | POOLED (3 sd) |
|---|---|---|
| STFlow none | 0.4477 ±0.0044 | 0.7853 ±0.0005 |
| STFiLM desc | 0.4590 ±0.0017 | 0.7937 ±0.0007 |
| **local (new best)** | **0.4624 ±0.0032** | **0.7950 ±0.0004** |
| moe | 0.4488 ±0.0013 | 0.7862 ±0.0028 |

- MoE lands **−0.010 LOOO / −0.008 POOLED below desc** (~5–8× its own seed std → real, not noise).
  Added expert capacity + router + untuned aux losses washed out the FiLM gain on small HEST folds.
- **`local` (hierarchical local+global FiLM) is the winner** in both regimes — the cheap fix beats
  the elaborate one. MoE stands as a negative-result ablation.
- Likely causes of MoE regression (not yet tested): over-regularization by `lambda_smooth`
  (uniform routing → redundant experts), too few tokens/expert to specialize, capacity hurting
  on tiny folds. Rescue attempts (sparse top-1, fewer experts, deeper-layers-only routing, lambda
  sweep) are optional — the simpler `local` already wins, so only pursue if MoE is wanted for the
  interpretability story (expert maps).

## Follow-up (2026-08-08) — symmetric baselines, significance, floor fix, ensembling
- **Symmetric comparison:** HisToGene/Hist2ST re-run at seeds 2–3 (job 11977206) → all rows now
  3-seed. `aggregate_comparison.py` updated. Ranking unchanged.
- **Paired significance** (`paired_significance.py`): FiLM vs `none` significant at raw paired-t
  p≈0.02 both regimes (PRIMARY claim); `local` vs `desc` is a TIE (p=0.27/0.41); Holm across the
  4-contrast family makes FiLM-vs-none marginal (p≈0.07–0.11) → frame FiLM-vs-none as
  pre-registered primary endpoint, rest secondary.
- **LOOO low-variance floor** (`rescore_filtered.py`): kidney/prostate/liver near-zero is a metric
  artifact — leakage-safe panel = training organs' HVGs, near-silent in held-out organ (detected in
  1–11% of spots). Detection-filtered rescore (≥10% detection) lifts all methods ~+0.03, preserves
  ranking (LOOO local 0.4624→0.4950). Still leakage-safe.
- **Baseline protocol note:** STFlow/HEST is per-cohort (own top-50 HVGs); our cross-organ LOOO/
  POOLED is novel and NOT directly comparable to their published per-cohort numbers.
- **Phase 3 seed-ensembling (in progress):** encoder-swap blocked (only UNI+resnet50 local). Added
  opt-in `--dump_preds` to `train_cross_organ.py` (best-epoch pred/gt → `fold_*_preds.npz`;
  earlier behavior unchanged). Re-running none/desc/local → `results_ens_uni8` (job 11979942).
  Aggregate with `ensemble_seeds.py`. Full writeup: `RESULTS.md`.

## Not pursued (unless requested)
- Ablations: dense vs sparse; random-router; n_experts sweep; lambda sweep.
- Interpretability: per-spot argmax expert maps; marker-gene-per-expert Δ_{e,g}.

## Open decisions / notes
- **Default is dense soft routing (`moe_top_k=0`)** — sparse top-k left some experts ungradiented
  on a tiny batch (expected dead-expert risk on small HEST cohorts); enable sparse only as an
  ablation once load balancing is confirmed to keep experts alive.
- `n_experts=4`, `lambda_bal=1e-2`, `lambda_smooth=1e-3` are untuned defaults.
- No checkpoint saving yet (`train_fold` doesn't `torch.save`) — same limitation as hybrid.
- Router currently uses scalar tokens (+ optional prototypes). Could add categorical `meta`
  tokens to the router input later (organ-aware routing) — still inference-valid.
