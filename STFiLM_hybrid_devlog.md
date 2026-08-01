# STFiLM Hybrid (FiLM + Cross-Attention) — Development Log

Running tracker of the hybrid conditioning experiment: what changed, where, why, how to run,
and status. Keep this updated as the work proceeds.

- **Base method:** STFiLM = STFlow (ICML'25) + FiLM/adaLN metadata conditioning.
- **This experiment (V4 `hybrid`):** keep FiLM/adaLN, and *add* a cross-attention path so each
  spot can attend to `K` learned prototype tokens pooled from the slide's histology. Motivation:
  affine FiLM is low-capacity and the `desc` conditioner is a single mean-pool; cross-attention +
  attention-pooled prototypes is more expressive. See conversation rationale (Options A/B/C → chose
  **Option A: hybrid**, so it is still legitimately "FiLM").
- **Design invariant:** all new capability is **opt-in behind `film=hybrid`**. Modes
  `none/context/meta/desc` build the exact same modules as before → earlier code/results unchanged.
- **Equivariance guardrail:** cross-attention operates only on the invariant scalar token stream
  using invariant conditioning tokens (pooled from image embeddings). It never touches
  coords/frames/edges → SE(2)-equivariance preserved (verified, see Status).

## Framework / stack
- PyTorch + torch_geometric + einops + timm (STFlow's stack). No new dependencies.
- Model code: `STFlow/stflow/model/` (`transformer.py`, `denoiser.py`, `config.py`).
- Trainer: `train_cross_organ.py` (our cross-organ LOOO/POOLED harness, not upstream train.py).
- Metric: `stflow.app.flow.test.metric_func` (`pearson_mean`).
- Run env: base conda (`miniconda3`), `PYTHONPATH=STFlow`.

## Conditioning modes (`--film`)
| mode | conditioner | new in this work? |
|---|---|---|
| `none` | none (upstream STFlow) | baseline |
| `context` | img+time re-injected via adaLN each layer | V1 |
| `meta` | context + categorical organ token (null idx 0) | V2 |
| `desc` | context + masked-mean-pool histology descriptor | V3 (current STFiLM) |
| **`hybrid`** | **`desc` FiLM path + cross-attention to K prototype tokens** | **V4 (this)** |

## Files changed / added

### `STFlow/stflow/model/transformer.py`
- **`CrossAttention` (new class)** — per-cell multi-head cross-attention: query = scalar token
  `[N, d]`, keys/values = per-cell conditioning tokens `[N, K, d]`. Pure scalar-stream op.
- **`TransformerBlock`** — new `cross_attn=False` ctor flag. When True, builds `self.cross_attn`
  and a separate adaLN-Zero gate `self.adaLN_xattn` (own gate → keeps the existing 4-chunk `adaLN`
  untouched). `forward(..., cond_tokens=None)`: if enabled, adds
  `token_embs += gate_xattn * cross_attn(token_embs, cond_tokens)` between the geometric-attn path
  and the MLP path. When `cross_attn=False`, block is byte-identical to before.
- **`SpatialTransformer`** — reads `config.cross_attn` (default False), passes to each block, and
  zero-inits `adaLN_xattn[-1]` (identity at init). `forward(..., cond_tokens=None)`: gathers
  `cond_tokens[batch_idx]` → `[N, K, d]` (batch-safe) and threads to blocks.

### `STFlow/stflow/model/denoiser.py`
- **`AttentionPool` (new class)** — learned-query attention pool: patch embeddings
  `[B, n, feature_dim]` → `K` prototype tokens `[B, K, d_model]`, masking padding spots. Richer
  than the single mean-pool descriptor.
- **`Denoiser`** — set `self.film_mode` before backbone; pass `cross_attn=(film=='hybrid')` into the
  inner `ModelConfig`. `desc_proj` now built for `desc` **and** `hybrid`. New `self.attn_pool` built
  only for `hybrid`. `inference()` builds `cond_tokens = attn_pool(raw_img, pad_mask)` for hybrid and
  passes to the backbone. `none/desc` paths unchanged (cond_tokens stays None).

### `STFlow/stflow/model/config.py`
- No edit needed — `ModelConfig(**kwargs)` already absorbs the new `cross_attn` kwarg.

### `train_cross_organ.py`
- `--film` choices += `"hybrid"`. New `--n_proto` (default 8) = number of prototype tokens.

## How to run
```
# Hybrid (V4), LOOO, matching the existing final config
PYTHONPATH=STFlow python train_cross_organ.py \
    --regime LOOO --film hybrid --seed 1 \
    --feature_encoder uni_v1_official \
    --splits_root cross_organ_splits8 \
    --source_dataroot dataset --embed_dataroot embed_dataroot \
    --save_root results_hybrid_uni8 --n_proto 8 --device 0
# repeat for --regime POOLED and seeds 2,3; compare vs desc/none via aggregate_comparison.py
```
To fold into the comparison table: add a `hybrid` row to `MODELS` in `aggregate_comparison.py`
pointing at `results_hybrid_uni8/{REGIME}_hybrid_seed{1,2,3}`.

## Status
- [x] Implemented hybrid path (opt-in, earlier code preserved).
- [x] Smoke test — all modes run fwd/bwd (`scratchpad/smoke_hybrid.py`).
- [x] adaLN-Zero: cross-attn gate zero-init → `hybrid` == `desc` at init.
- [x] Learnability: over 4 Adam steps the gate opens (|w| 0→82→138) and `attn_pool` + `cross_attn`
      receive gradient; loss 1.17→0.39 (`scratchpad/smoke_train.py`).
- [x] **SE(2)-equivariance preserved**: max |pred(coords) − pred(R·coords+t)| = 0.0.
- [~] Full runs on `cross_organ_splits8` (LOOO+POOLED × 3 seeds) — **LAUNCHED** SLURM job
      `11784857` (array 0-1, 2 GPUs, 6 configs), scripts `hybrid_sweep.sbatch` + `run_uni_hybrid.sh`,
      output `uni8_hybrid_s{0,1}_*.log`, results → `results_hybrid_uni8/`.
- [ ] Aggregate hybrid vs desc/none/baselines; update `comparison_*.md/csv`.

## Open decisions / notes
- `n_proto=8` is an untuned default; sweep {4, 8, 16}.
- Hybrid currently uses the `desc` descriptor + prototypes (no categorical meta); could add
  `meta` tokens to `cond_tokens` later.
- No checkpoint saving yet (`train_fold` doesn't `torch.save`) — add it so inference-time sweeps
  (CFG, sample steps) are free rather than requiring reruns.
- Cross-attn grad is small right after the gate opens (expected; downstream of a just-opened gate).
