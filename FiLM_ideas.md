# FiLM Conditioning Ideas for Transcriptomics + Histopathology Models

Goal: extend recent venue-accepted H&E→spatial-transcriptomics methods with **FiLM (Feature-wise Linear Modulation; Perez et al. 2018)** conditioning, as a publishable methods contribution at a top ML venue.

## FiLM mechanism
```
γ, β = MLP(c)          # conditioner c → per-channel scale/shift, each shape [C]
h_out = γ ⊙ h + β      # modulate features h
```
Design choice every time: **what is the conditioner `c`, and what features `h` does it modulate?**

**Core validity rule:** a FiLM conditioner is only usable if available at inference.
- Histology embeddings — available (green)
- Categorical metadata (organ, platform, cancer subtype) — available (green)
- Gene expression — NOT available, it is the prediction target → pretraining-only (red)

## Target papers (all accepted)

| Paper | Venue | Link |
|-------|-------|------|
| **Stem** — Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images | ICLR 2025 (Poster) | https://openreview.net/forum?id=FtjLUHyZAO |
| **STFlow** — Scalable Generation of Spatial Transcriptomics from Histology via Whole-Slide Flow Matching | ICML 2025 (Poster) | https://icml.cc/virtual/2025/poster/45412 · arXiv 2506.05361 |
| **STAMP** — Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology | ICLR 2026 (Poster) | https://openreview.net/forum?id=uVXO6gzVzj |
| **HistoPrism** — Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction | ICLR 2026 (Poster) | https://openreview.net/forum?id=6dTHxb9JuA |

## Per-paper FiLM insertion ideas

1. **Stem (strongest fit).** Denoiser already uses adaptive group norm on timestep `t` — that *is* FiLM on `t`. Extend the conditioner to `c = [histology_embed ; t_embed ; organ ; platform]` and emit `γ,β` per denoiser block. Replaces concat-style conditioning with multiplicative modulation. Low-risk, principled.

2. **STFlow (equally strong).** Velocity field `v_θ(x_t, t, context)`. The slide encoder already emits a per-spot histology+spatial context vector → ideal FiLM conditioner per flow block. Change *how* the existing context enters (FiLM vs. current fusion), not what signal is used.

3. **HistoPrism (most novel angle).** For a transformer, FiLM = **adaptive LayerNorm (adaLN, DiT-style / adaLN-Zero).** Condition on categorical metadata (cancer type / organ / platform) → directly targets their pan-cancer generalization claim. Cheap to add, plausibly improves the headline metric.

4. **STAMP (watch inference asymmetry).** FiLM the image encoder with a gene-expression embedding **during pretraining only** (genes unavailable at deployment). Safe variants: image→gene FiLM, or genes-as-FiLM in the SSL phase only. Do not build an inference path that needs the gene signal.

## Most promising underexplored axis
Use **categorical metadata** (sequencing platform Visium vs. Xenium, organ, cancer subtype) as the FiLM conditioner so that **one** model generalizes across platforms/organs instead of per-dataset training. FiLM is purpose-built for low-dimensional categorical conditioning. Small architectural delta, defensible contribution.

## Stem code analysis (github.com/SichenZhu/Stem, `Stem/models.py`)

**Key finding: Stem already IS a FiLM model.** It is a DiT (Diffusion Transformer) and uses **adaLN-Zero**, which is FiLM. So "adding FiLM to Stem" is not itself a contribution — FiLM is the conditioning mechanism. The defensible move is adding a **new conditioner** (categorical metadata) into the existing FiLM path.

Concretely, in `Stem/models.py`:
- `modulate(x, shift, scale)` (lines 8-9) is the FiLM op: `x * (1 + scale) + shift`.
- `DiTBlock.forward(x, c)` (lines 109-113) runs FiLM via `adaLN_modulation(c)` → shift/scale/gate for both attention and MLP sublayers.
- The conditioner is built in `StemModel.forward` (line 210) as **`c = t + y`**, where `t` = timestep embedding and `y` = histology foundation-model embedding (512-d, pooled H&E patch embedding) projected by `label_embed` (lines 158-162).
- adaLN is zero-initialized (`initialize_weights`, lines 188-195) → each block starts at identity, so any new term added to `c` begins as a near-no-op (training-stable).
- Call site: `gaussian_diffusion.py:279` invokes `model(x, t, **model_kwargs)`. Today `model_kwargs = {'y': ...}`. **Adding a `meta` key threads through with ZERO changes to the diffusion code** — only `StemModel.forward` + data prep change.

### Concrete diff — add a categorical-metadata FiLM conditioner

Patch `Stem/models.py`:

```python
# --- NEW: place near GeneJointEmbedding (after line ~84) ---
class MetadataEmbedder(nn.Module):
    """FiLM conditioner from categorical metadata (organ, platform, cancer subtype).
    Reserve index 0 as a 'null' token for classifier-free guidance / dropout."""
    def __init__(self, num_categories: dict, hidden_size):
        super().__init__()
        # +1 per field for the null/unconditional token at index 0
        self.embeds = nn.ModuleDict({
            name: nn.Embedding(n + 1, hidden_size) for name, n in num_categories.items()
        })

    def forward(self, meta: dict):
        # meta: {field_name: LongTensor (N,)}, 0 = null
        out = 0
        for name, emb in self.embeds.items():
            out = out + emb(meta[name])
        return out                                   # (N, hidden_size)
```

```python
# --- StemModel.__init__ : add arg + build embedder ---
    def __init__(self,
        input_size=200, hidden_size=1152, depth=28, num_heads=16,
        mlp_ratio=4.0, label_size=512, learn_sigma=True,
        meta_categories=None,                        # NEW e.g. {"organ": 18, "platform": 2}
    ):
        ...
        self.label_embed = nn.Sequential(...)        # unchanged
        self.meta_embed = (MetadataEmbedder(meta_categories, hidden_size)
                           if meta_categories else None)   # NEW
```

```python
# --- StemModel.initialize_weights : keep metadata near-identity at start ---
        if self.meta_embed is not None:              # NEW
            for emb in self.meta_embed.embeds.values():
                nn.init.normal_(emb.weight, std=0.02)
                nn.init.constant_(emb.weight[0], 0)  # null token = 0 (true no-op)
```

```python
# --- StemModel.forward : thread meta into the FiLM conditioner ---
    def forward(self, x, t, y, meta=None):           # NEW arg
        x = self.gene_joint_embed(x)
        t = self.time_embed(t)
        y = self.label_embed(y)
        c = t + y                                    # existing FiLM conditioner
        if self.meta_embed is not None and meta is not None:
            c = c + self.meta_embed(meta)            # NEW: metadata-conditioned FiLM
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return x
```

**Threading it through (no diffusion-code edits):** in the training/sampling loop, add the metadata tensors to `model_kwargs`, e.g. `model_kwargs = {"y": hist_emb, "meta": {"organ": organ_id, "platform": plat_id}}`. `gaussian_diffusion.py` passes `**model_kwargs` straight to `forward`.

**Classifier-free guidance:** randomly set metadata fields to the null index 0 during training (e.g. p=0.1); at sampling, interpolate between null-conditioned and metadata-conditioned outputs to control how strongly platform/organ steers the prediction.

### What this buys (the paper hook)
One Stem model that **generalizes across platforms (Visium↔Xenium) and organs** via metadata-conditioned FiLM, instead of per-dataset training — plus CFG as a knob to study how strongly tissue/platform priors shape inferred expression. Small, additive code change on top of an accepted method; the novelty is the conditioning axis, not the FiLM layer.

### Open design choice
Additive fusion `c = t + y + m` matches the original DiT pattern but is a weak fusion. An ablation worth running: replace addition with a small MLP over `concat([t, y, m])`, or give metadata its own adaLN scale (separate modulation head) — measure whether richer fusion beats addition.

## STFlow code analysis (github.com/Graph-and-Geometric-Learning/STFlow, ICML 2025 Spotlight)

**Key finding: the OPPOSITE of Stem — STFlow has NO FiLM, so there is real room to add it.** STFlow is a flow-matching model whose denoiser is a custom *geometric* `SpatialTransformer` (frame-averaging / SE(2)-equivariant graph attention), not a DiT. It conditions **additively, once, at the input** and never re-injects the condition.

Concretely, in `stflow/model/denoiser.py` (`Denoiser.inference`, lines 82-84):
```python
img_features = self.image_transform(img_features)          # per-spot histology feats -> d_model
time_emb     = self.fourier_proj(t_steps)[:, None].expand(...)
features     = img_features + time_emb                       # <-- ONLY conditioning, injected once
prediction   = self.backbone(gene_exp=noisy_exp, features=features, coords=coords)
```
`features` becomes the initial `token_embs`. Inside `stflow/model/transformer.py`:
- `TransformerBlock.forward` (lines 162-169) is plain pre-norm-ish residual: `token_embs += attn(...)`, `token_embs += mlp(...)`. **No adaptive modulation.** After the input, time/image condition is only carried forward implicitly through residuals.
- `MLPAttnEdgeAggregation` (lines 42-127) is the equivariant geometric attention; `gene_exp` enters as neighbor differences and frame/edge features carry the geometry.

So the condition is injected exactly once and decays through depth — the classic case where **per-layer FiLM re-injection helps** (this is precisely what DiT/adaLN does and what STFlow lacks).

### EQUIVARIANCE GUARDRAIL (important)
STFlow's whole selling point is SE(2)/frame-averaging equivariance + cell-cell interaction. FiLM is safe **only if it modulates the invariant scalar token stream** (`token_embs`), conditioned on invariant scalars (image embedding, time, categorical metadata). **Do NOT** let FiLM touch the frame/edge geometric features (`frame_feats`, `radial_coords`) or you break equivariance. The diff below modulates only `token_embs`, so equivariance is preserved.

### Concrete diff — add adaLN-style FiLM (per-layer re-injection) + metadata

Patch `stflow/model/transformer.py`:
```python
def modulate(x, shift, scale):                       # NEW (same as Stem)
    return x * (1 + scale) + shift

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_edge_model, n_genes, n_heads=1, activation="gelu",
                 attn_drop=0., proj_drop=0., gene_exp_non_negative=True, mlp_ratio=4.0):
        super().__init__()
        self.attn = MLPAttnEdgeAggregation(...)       # unchanged
        self.mlp  = Mlp(...) or SwiGLUPacked(...)      # unchanged
        self.gene_updater = GeneUpdate(...)           # unchanged
        # NEW: FiLM conditioner heads + non-affine norms on the scalar token stream
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, c):   # NEW arg c
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)         # c: [N, d_model]
        h = modulate(self.norm1(token_embs), sh1, sc1)
        token_embs = token_embs + g1 * self.attn(gene_exp, h, coords, neighbor_indices)
        h = modulate(self.norm2(token_embs), sh2, sc2)
        token_embs = token_embs + g2 * self.mlp(h)
        gene_exp = self.gene_updater(token_embs)
        return gene_exp, token_embs
```
```python
class SpatialTransformer(nn.Module):
    def forward(self, gene_exp, features, coords, cond):                    # NEW arg cond
        ...                                                                 # existing pad-mask/flatten
        cond = cond[~pad_mask]                                              # NEW: flatten like features
        all_gene_exp = []
        for blk in self.blks:
            gene_exp, features = blk(gene_exp, features, coords, nearest_indices, cond)  # pass cond
            all_gene_exp.append(gene_exp)
        ...
```
adaLN-Zero init (in `SpatialTransformer.__init__`, after building `self.blks`):
```python
        for blk in self.blks:                                              # NEW: start each block ~identity
            nn.init.constant_(blk.adaLN[-1].weight, 0)
            nn.init.constant_(blk.adaLN[-1].bias, 0)
```

Patch `stflow/model/denoiser.py` — separate the init token stream from the modulation signal, and add metadata:
```python
    def __init__(self, config):
        ...
        self.fourier_proj    = TimestepEmbedder(config.hidden_dim)
        self.image_transform = nn.Linear(config.feature_dim, config.hidden_dim)
        # NEW: categorical metadata embedder (organ/platform/subtype); idx 0 = null for CFG
        self.meta_embed = (MetadataEmbedder(config.meta_categories, config.hidden_dim)
                           if getattr(config, "meta_categories", None) else None)

    def inference(self, noisy_exp, img_features, coords, t_steps, meta=None, predict=False):
        img_features = self.image_transform(img_features)                  # init token stream
        time_emb     = self.fourier_proj(t_steps)[:, None].expand(*img_features.shape[:2], -1)
        cond = img_features + time_emb                                     # FiLM signal (was the input)
        if self.meta_embed is not None and meta is not None:
            cond = cond + self.meta_embed(meta)[:, None]                   # broadcast per spot
        prediction = self.backbone(
            gene_exp=noisy_exp, features=img_features, coords=coords, cond=cond,  # token init = image only
        )
        return prediction
```
(`MetadataEmbedder` = the same small class from the Stem section.)

### Two contributions this unlocks for STFlow
1. **FiLM itself is novel here** (STFlow has none): per-layer re-injection of the histology+time condition via adaLN, vs. STFlow's inject-once-at-input. Direct ablation: original additive input vs. adaLN re-injection.
2. **Metadata-conditioned generalization** (same axis as Stem): one model across platforms (Visium↔Xenium)/organs, with CFG via the null token.

### Stem vs STFlow — the contrast (use this in the paper framing)
| | Stem (ICLR'25) | STFlow (ICML'25) |
|--|--|--|
| Backbone | DiT | geometric (frame-averaging) graph transformer |
| FiLM today | **already present** (adaLN-Zero) | **absent** (additive, input-only) |
| Condition re-injected per layer? | yes | **no** |
| FiLM contribution | add new *conditioner* (metadata) | add FiLM *mechanism* + metadata |
| Equivariance constraint | none | **must modulate scalar tokens only** |

STFlow is the stronger target if the goal is "introduce FiLM"; Stem is the target if the goal is "introduce a new conditioning signal into an existing FiLM model." The metadata-generalization story works for both.

## Split generator — `make_cross_organ_splits.py` (DONE, verified on a synthetic fixture)

Emits both regimes in STFlow's split format from the per-cohort HEST-bench data.
Grounded in the real layout: each cohort dir has `splits/{train,test}_i.csv`
(columns `sample_id, patches_path, expr_path`), a `var_50genes.json`, `.h5ad` expr,
`.h5` tiles.

Key behaviors (all unit-checked):
- Regroups 10 cohorts into 8 organ groups (colorectal={COAD,READ}, breast={IDC,LYMPH_IDC}).
- **Regime B (LOOO):** `test = held-out group`, `train = rest`; shared gene panel built
  ONLY from training cohorts (leakage-safe — verified the held-out organ's unique genes
  are excluded).
- **Regime A (pooled leave-patient-out):** every organ in every fold; asserts no
  same-patient leakage across the boundary.
- Rewrites paths to `<COHORT>/<orig_path>` so a pooled run resolves all cohorts from one
  `--hest_root` (run STFlow with `source_dataroot=hest_root`, `gene_list=genes_<fold>.json`).
- Patient grouping needs HEST master metadata (`--hest_meta`, cols `id`/`patient`);
  without it, falls back to sample-level grouping with a warning (a real leakage caveat).
- `--gene_panel_mode`: `union_topk` (default, lightweight, leakage-safe), `intersection`,
  or `hvg` (expression-driven, needs scanpy + downloaded h5ad).

Run:
```bash
python3 make_cross_organ_splits.py \
  --hest_root /path/to/hest-bench --out_root ./cross_organ_splits \
  --regimes AB --n_folds 5 --n_genes 50 \
  --gene_panel_mode union_topk \
  --hest_meta /path/to/HEST_v1_metadata.csv   # optional but needed for patient-safe folds
# -> cross_organ_splits/{LOOO,POOLED}/splits/*.csv + genes_*.json + manifest.json
```

## IMPLEMENTED — STFlow adaLN+metadata FiLM (in `./STFlow/`, verified)

Cloned `Graph-and-Geometric-Learning/STFlow` into `./STFlow/` and applied the FiLM diff to
`stflow/model/transformer.py` and `stflow/model/denoiser.py`. Verified with a real
forward+backward smoke test (tiny tensors, padding, with/without metadata, CFG null token).

What was added:
- `transformer.py`: `modulate()`; `TransformerBlock` now has `norm_mlp` + a `adaLN`
  head (`Linear(d_model, 4*d_model)`) producing `gate_msa, shift_mlp, scale_mlp, gate_mlp`;
  `forward(..., c)` gates the geometric attention's OUTPUT (input untouched → equivariance
  preserved) and applies full FiLM on the MLP path. `SpatialTransformer` zero-inits every
  block's adaLN (adaLN-Zero) and threads `cond` (flattened by the same pad-mask as features).
- `denoiser.py`: new `MetadataEmbedder` (per-field `nn.Embedding`, index 0 = null token,
  zero-inited); `Denoiser` builds it from `config.meta_categories`; `inference`/`forward`
  take `meta` and build `cond = img+time(+meta)`, token init left unchanged.

Verified behaviors (post-init, i.e. once adaLN moves off zero in step 1):
- conditioning changes the output; organ+platform embeddings receive gradient; unused
  null token (idx 0) gets zero gradient. At exact init adaLN-Zero makes conditioning a
  no-op (same property as Stem/DiT) — expected, not a bug.

Two upstream issues found & handled (FLAG for user):
1. `GeneUpdate.__init__` did NOT accept `non_negative`, yet `TransformerBlock` passes it
   → upstream would `TypeError` on instantiation. Worked around by accepting the kwarg
   (stored, not yet applied to output). Decide: apply softplus/relu (targets are log1p,
   non-negative) or drop the flag.
2. `MLPAttnEdgeAggregation` hardcodes `+50` (== n_genes) in `mlp_attn` in_features, so the
   model only works with **n_genes == 50**. Our shared panel default is 50 → consistent,
   but any other panel size needs this changed to `+n_genes`.

### Still open before training
- Download HEST-bench (`MahmoodLab/hest-bench`) + supply patient metadata for Regime A.
- Run `make_cross_organ_splits.py`; optionally `--gene_panel_mode hvg` once data is local.
- Wire the emitted `genes_<fold>.json` + pooled `source_dataroot` into STFlow's flow
  trainer (`stflow/app/flow/train.py`) and the dataset loader: emit per-spot/per-slide
  `meta = {"organ": id, "platform": id}` tensors and pass them into `Denoiser.forward`.
  (Model side already accepts `meta`; remaining work is the data/training loop.)
- Build the organ/platform id vocab (Regime B: held-out organ → null id 0).
- Add CFG: randomly null out `meta` fields during training (p~0.1); guidance sweep at sampling.

## Related files in this directory
- `make_cross_organ_splits.py` — cross-organ HEST-1k split generator (this section)
- `iclr_genomics_papers.csv` / `iclr_genomics_papers.md` — full ICLR 2025+2026 genomics scan
