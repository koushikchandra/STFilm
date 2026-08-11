# STFiLM performance-improvement plan (12 levers)

The menu of ways to push STFiLM past its current results, with honest EV/cost and live status.
Two distinct goals:

- **(A) Turn the HEST *tie* into a clean *win*** — cheap recipe/conditioner/inference tweaks,
  each ~+0.005–0.02. Can run on UNI now. **Will not break the ~0.46 LOOO wall.**
- **(B) Raise the *ceiling* (esp. LOOO ~0.46)** — only a **stronger encoder** realistically moves this.

Current headline (UNI, `local`, 3-seed): **LOOO 0.4750 / POOLED 0.8021** (ensemble);
single-model **0.4624 / 0.7950**.

## The 12 levers

| # | Lever | Group | Est. gain | Cost | Status |
|---|---|---|---|---|---|
| 1 | Cosine LR + longer training | recipe | +0.01–0.02 | cheap | ✅ ran — **no help** for STFiLM |
| 2 | Classifier-free guidance at inference (sweep scale w>1) | inference | +0.005–0.015 | cheap | ⬜ not run |
| 3 | Correlation loss (1−Pearson) + AdamW/wd + EMA | recipe/loss | +0.005–0.015 | cheap | ✅ ran (`plus`) — **no help** |
| 4 | Seed ensembling as headline (0.475/0.802) | inference | +0.01 | free | ✅ have |
| 5 | Richer local conditioner (multi-scale kNN, attention-pooled, learned) | conditioner | +0.005–0.015 | medium | 🔶 `localg` gated-fusion ≈ **tie** |
| 6 | + Platform metadata (Visium/Xenium) conditioning | conditioner | cross-platform | cheap | ⬜ not run |
| 7 | Morphology-conditioned ZINB prior | conditioner | +? | medium | ⬜ not run |
| 8 | Backbone hyperparams: n_layers 4→6/8, hidden 128→256, k, dropout | backbone | +0.005–0.01 | cheap–med | ✅ **WIN** (`bigcap` 6/256): LOOO +0.0100, POOLED +0.0032 |
| 9 | More sample steps S=5→10/16 | inference | ±small | cheap | ⬜ not run |
| 10 | **GigaPath encoder** (re-extract → rerun STFiLM) | **encoder** | **+0.02–0.05 (ceiling)** | expensive | ❌ **NEGATIVE** — GigaPath < UNI (−0.03 LOOO) |
| 11 | UNI + resnet50 concat (both on disk) | encoder | +? small | cheap | ⬜ not run (resnet50 is the weak encoder → ablation; deprioritized after #10 negative) |
| 12 | 5 seeds (tightens significance, not mean) | reporting | p-value | cheap | ⬜ not run |

## What we've learned so far

- **#8 (capacity, `bigcap` n_layers=6/hidden=256) → WIN.** LOOO 0.4724 vs 0.4624 (+0.0100),
  POOLED 0.7982 vs 0.7950 (+0.0032), 3 seeds, every seed above baseline. First lever to move the
  ceiling; takes STFiLM from a LOOO tie with TRIPLEX (0.462) to a clear lead (0.472). Cost ~3× params
  (8.1M) / ~3× train time. **This is the lever to build on.**
- **#10 (GigaPath encoder) → NEGATIVE.** 3-seed means below UNI on every arm: LOOO none 0.417/desc
  0.437/local 0.433 vs UNI 0.448/0.459/0.462; POOLED ~flat. The ViT-giant encoder *lowers* the ceiling.
  Silver lining: FiLM still helps *within* GigaPath (none→local +0.02), so the conditioning benefit is
  encoder-independent — a robustness datapoint for the paper. **"Stronger encoder" thesis falsified;
  UNI stays.** (Also kills #11.)
- **#1, #3 ran → no help.** Recipe/loss tweaks do **not** move the flow-matching model. The
  cheap-batch (A) thesis is weak for STFiLM specifically (they *did* help CoMRA).
- **#5 (`localg` gated fusion) → tie** with `local` (LOOO 0.4632 vs 0.4624, POOLED 0.7937 vs
  0.7950, 3 seeds). No gain from learning to blend global vs per-spot descriptor.
- **#10 (GigaPath) → running**; early signal *discouraging* — unconditioned `none` LOOO = 0.419,
  slightly **below** UNI's ~0.44. The `desc`/`local` arms are the ones that matter; verdict pending.
- **Diagnostic that reframes everything:** LOOO is suppressed by ~3 organs (kidney/liver/prostate)
  where leakage-safe panel genes are near-silent (variance 10–200× lower, expressed in 1–11% of
  spots). Pearson vs a near-constant target is noise no matter the model. See
  `rescore_filtered.py` / memory `looo-low-variance-floor`. Model-side gains only pay off on the 5
  signal-bearing organs (skin/colorectal/lung/pancreas/breast).

## Recommended next (as of 2026-08-10)

Ranked by expected value given what we now know:

1. **#8 — backbone capacity bump** (n_layers 4→6/8, hidden 128→256). Untried, cheap, and it
   targets model capacity rather than the recipe (which we've shown is a dead end). Best cheap swing.
2. **#2 — classifier-free guidance at inference.** We already train with a null-token / CFG-ready
   path; only needs an eval that applies guidance scale w>1. Zero retraining, pure inference lever.
3. **#7 — morphology-conditioned ZINB prior.** Higher-risk/higher-reward; the only lever aimed at
   the *floor organs*, which is where the real LOOO deficit lives.

Deprioritized: #11 (resnet concat, weak encoder → small-EV ablation), #12 (reporting only),
#6/#9 (marginal). #10 (GigaPath) is the true ceiling lever but early data is not encouraging —
let it finish before investing in more encoders.
