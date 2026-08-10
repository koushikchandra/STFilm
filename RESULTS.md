# STFiLM — Cross-Organ Results Summary

Consolidated results for FiLM/adaLN conditioning added to STFlow, evaluated on the
leakage-safe cross-organ HEST splits (`cross_organ_splits8`), UNI features, `pearson_mean`,
3 seeds. Two regimes: **LOOO** (leave-one-organ-out → transfer) and **POOLED**
(leave-patient-out, all organs seen → metadata benefit).

## 1. Headline

- **FiLM conditioning significantly beats unconditioned STFlow** — this is the primary, defensible result.
- `local` (per-spot kNN descriptor + global) has the best mean; `desc` (global descriptor)
  is **statistically tied** with it. Pick `local` as the headline variant (best mean + per-spot
  story); `desc` is the simpler tie.
- **MoE regresses to ~baseline** — negative-result ablation.
- All variants preserve SE(2)-equivariance (verified max |pred(x)−pred(Rx+t)| = 0).

## 2. Symmetric comparison (all 3-seed; probes deterministic)

### LOOO — pearson_mean
| model | raw | detection-filtered¹ |
|---|---|---|
| HisToGene | 0.4379 | 0.4612 |
| Hist2ST | 0.4087 | 0.4438 |
| STFlow `none` | 0.4477 | 0.4789 |
| STFiLM `desc` | 0.4590 | 0.4910 |
| **STFiLM `local`** | **0.4624** | **0.4950** |
| STFlow+MoE | 0.4488 | 0.4811 |

### POOLED — pearson_mean
| model | raw |
|---|---|
| HisToGene | 0.7766 |
| Hist2ST | 0.7689 |
| STFlow `none` | 0.7853 |
| STFiLM `desc` | 0.7937 |
| **STFiLM `local`** | **0.7950** |
| STFlow+MoE | 0.7862 |

¹ Detection-filtered = re-score over only genes expressed in ≥10% of the held-out organ's
spots (see §4). Leakage-safe (target-side filter). Ranking is unchanged; the FiLM gap widens
slightly. Baselines were made 3-seed on 2026-08-08 (HisToGene/Hist2ST seeds 2–3, job 11977206).

## 3. Paired significance (per-fold, seed-averaged)

Primary contrast = **FiLM vs `none`** (pre-registered), reported at raw paired-t.
Secondary contrasts reported Holm-corrected across the 4-contrast family.

| contrast | LOOO Δ / raw t-p / Holm | POOLED Δ / raw t-p / Holm |
|---|---|---|
| `desc` vs `none` (FiLM benefit) | +0.0113 / **0.037** / 0.111 | +0.0084 / **0.018** / 0.073 |
| `local` vs `none` (FiLM benefit) | +0.0147 / **0.020** / 0.078 | +0.0097 / **0.021** / 0.073 |
| `local` vs `desc` | +0.0034 / 0.27 / 0.27 | +0.0013 / 0.41 / 0.41 |
| `local` vs HisToGene | +0.0245 / 0.095 / 0.19 | +0.0183 / 0.051 / 0.10 |

- **FiLM vs `none`: significant at raw p ≈ 0.02** in both regimes (primary claim).
- POOLED has only 5 folds → Wilcoxon floor is p=0.0625 (5/0 sweeps); paired-t p≈0.02 is the
  informative test there.
- Under Holm across all 4 contrasts, FiLM-vs-none is **marginal (p≈0.07–0.11)** — report as
  secondary/exploratory if not treating FiLM-vs-none as the pre-registered primary endpoint.
- `local` vs `desc` is a **tie** (p=0.27–0.41). Do not claim `local` > `desc`.

Script: `paired_significance.py`.

## 4. The LOOO low-variance floor (important)

Three organs sit near zero in raw LOOO — kidney (0.15), prostate (0.13), liver (0.04) — and
drag the mean down. This is **NOT a model failure**; it is a metric artifact:

- Sample count rules out "too little data": kidney/prostate have the **most** held-out samples
  (24, 23) and fail; lung/skin have only 2 and succeed (0.68/0.78).
- Gene panels 100% overlap the training organs → not an unseen-gene problem.
- **Cause:** the leakage-safe LOOO panel is built from *training* organs' HVGs, which are
  **near-silent in the held-out organ**. Median gene variance is 10–200× lower and genes are
  detected in only 1–11% of spots (prostate: 1.2%, liver: 5.8%, kidney: 11.4%) vs 40–74% for
  the signal organs. Pearson against a ~constant target is structurally ~0.
- **Fix:** detection-filtered scoring (§2, col 2) — keep only genes expressed in ≥10% of the
  held-out organ's spots. Lifts every method ~+0.03, preserves ranking, makes the number
  meaningful. Still leakage-safe (filter uses the target's own distribution, never training).
- Liver stays low even filtered (0.07, 5 genes) and prostate rests on 2 genes — genuinely thin,
  not fully rescued.

Scripts: `rescore_filtered.py` (diagnostic + filtered table).

## 5. Baseline paper protocol vs ours (do not conflate)

- **STFlow / HEST-bench (baselines):** evaluate **each organ separately** — patient-level
  k-fold *within* one cohort, scored on **that cohort's own top-50 HVGs**. No cross-organ
  transfer. Kidney is scored on kidney's own (variable, expressed) genes → healthy Pearson.
- **Ours (`cross_organ_splits8`):** LOOO/POOLED cross-organ, scored on a leakage-safe panel
  from *other* organs → the floor in §4. **This is our novel regime; our numbers are NOT
  directly comparable to STFlow's published per-cohort numbers** (different genes, different task).

## 6. Phase 3 — seed ensembling (complete)

Averaging the 3 seeds' predictions (`--dump_preds` → `results_ens_uni8`, job 11979942;
`ensemble_seeds.py`) adds ~+0.012 PCC and preserves the ranking. The `single` column reproduces
the original per-seed numbers exactly, validating the dump pipeline.

| film | LOOO single→ens | LOOO ens+filter | POOLED single→ens |
|---|---|---|---|
| none | 0.4477→0.4645 | 0.4996 | 0.7853→0.7937 |
| desc | 0.4590→0.4691 | 0.5022 | 0.7937→0.8011 |
| **local** | **0.4624→0.4750** | **0.5096** | **0.7950→0.8021** |

Encoder-swap lever stays blocked (only UNI + resnet50 weights local).

## 7. Biomarker-level evaluation (STFlow Table-2 style)

Our leakage-safe panels are 11 clinically relevant markers (MKI67/Ki-67, CD8A/CD3E immune
infiltration, KIT/TNFRSF17 drug targets, ACTA2 stroma, …). Per-gene PCC (mean over folds/regimes/
3 seeds, `biomarker_eval.py`): **FiLM improves every biomarker** — `local` beats `none` on
**11/11** (mean Δ +0.012), `desc` on 11/11 (+0.010); largest gains on immune markers
(CD3E/CD8A/CXCR4/CCR7, +0.015–0.016). Biomarker means: none 0.5775, desc 0.5872, local 0.5898.

## Reproduce
```
python aggregate_comparison.py      # symmetric 3-seed comparison tables
python paired_significance.py       # raw + Holm-corrected paired tests
python rescore_filtered.py          # low-variance-floor diagnostic + filtered LOOO
python ensemble_seeds.py            # (after job 11979942) seed-ensemble numbers
```
