# Running STFiLM-`local` (our best variant)

`local` is the best-performing STFiLM conditioner in all our experiments. It conditions the
STFlow flow-matching backbone with **both** a global slide descriptor **and** a per-spot local
descriptor (i.e. `local` = `local + desc`), applied via equivariance-preserving adaLN-Zero FiLM.

**Headline results (UNI features, 3 seeds, `pearson_mean`):**

| Regime | single-model | 3-seed ensemble |
|---|---|---|
| LOOO (leave-one-organ-out, transfer) | 0.4624 | 0.4750 |
| POOLED (leave-patient-out) | 0.7950 | 0.8021 |

It is the top method on HEST cross-organ, improves **every** one of 11 clinical biomarkers over the
unconditioned backbone, and adds only ~0.26M parameters.

---

## What `local` does

For each transformer block, a zero-initialized adaLN-Zero head produces per-channel shift/scale
from an **inference-available, SE(2)-invariant** conditioner:

```
cond = image_features + timestep
     + desc_proj( global masked-mean of UNI features )      # per-slide  (broadcast)
     + local_proj( per-spot kNN-neighbourhood mean )        # per-spot   (spatially varying)
```

Both terms are histology-derived (available for unseen organs) and are means (permutation- and
rigid-motion invariant), so the SE(2)-equivariant backbone is preserved. Zero-init means STFiLM
== STFlow at initialization; conditioning grows during training.

---

## Prerequisites (data layout)

Resolved by convention (see `CLAUDE.md`):

- UNI patch features: `embed_dataroot/<cohort>/uni_v1_official/fp32/<sample_id>.h5`
- expression: `dataset/<cohort>/adata/<sample_id>.h5ad`
- leakage-safe cross-organ splits + 50-gene panels: `cross_organ_splits8/{LOOO,POOLED}/`
  (built by `make_cross_organ_splits.py`)

If you only have raw HEST patches, extract UNI features first with
`STFlow/stflow/app/hest/benchmark.py` (or `extract_gigapath_only.py` for GigaPath).

---

## Train one run

```bash
# LOOO transfer, seed 1
PYTHONPATH=STFlow python train_cross_organ.py \
    --regime LOOO --film local --seed 1 \
    --feature_encoder uni_v1_official \
    --splits_root cross_organ_splits8 \
    --source_dataroot dataset \
    --embed_dataroot embed_dataroot \
    --save_root results_local_uni8 \
    --lr 1e-3 --epochs 100 --dump_preds --device 0
```

Swap `--regime POOLED` for the pooled leave-patient-out regime. Results are written to
`results_local_uni8/<regime>_local_seed<seed>/` (per-fold `*_results.json`, `results_kfold.json`,
and `*_preds.npz` when `--dump_preds` is set, for seed ensembling).

Key flags:

- `--film local` — the variant (global + per-spot conditioning). Others: `none` (= STFlow),
  `desc`, `hybrid`, `moe`, `localg` (gated fusion).
- `--dump_preds` — save best-epoch predictions so seeds can be ensembled.
- Optional (opt-in, off by default): `--lr_schedule cosine --warmup 10 --patience 40`,
  `--optimizer adamw --weight_decay 0.01`, `--corr_weight 0.5`.

## Full sweep (both regimes × 3 seeds)

```bash
for REGIME in LOOO POOLED; do
  for SEED in 1 2 3; do
    PYTHONPATH=STFlow python train_cross_organ.py \
      --regime $REGIME --film local --seed $SEED \
      --feature_encoder uni_v1_official --splits_root cross_organ_splits8 \
      --source_dataroot dataset --embed_dataroot embed_dataroot \
      --save_root results_local_uni8 --lr 1e-3 --epochs 100 --dump_preds --device 0
  done
done
```

(On SLURM, `run_uni_local.sh` / `spatial_sweep.sbatch`-style array jobs shard this across GPUs.)

## Aggregate & compare

```bash
python aggregate_comparison.py     # -> comparison_LOOO.md / comparison_POOLED.md (all baselines + ours)
python ensemble_seeds.py           # 3-seed ensemble (needs *_preds.npz)
python biomarker_eval.py           # per-biomarker PCC (local beats none 11/11)
python paired_significance.py      # local vs none/desc significance
```

---

## Notes / gotchas

- The gene panel is fixed at **50 genes** (STFlow's `MLPAttnEdgeAggregation` hardcodes `+50`).
- LOOO scores on kidney/liver/prostate are near-zero **by metric artifact** (leakage-safe panels
  are near-silent in those held-out organs), not model failure — see `rescore_filtered.py`.
- `local` beats `desc` only slightly (within seed noise); we report it as best-performing, not
  significantly superior to `desc`.
