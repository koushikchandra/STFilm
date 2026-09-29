# MorphoST autonomous run — state & decision plan

**Window:** 2026-08-11 03:38 → **08:38 CDT** (5 hours). Run MorphoST autonomously; do not ask.

## Reference bar (HEST LOOO, Pearson)
STFlow 0.448 · STFiLM-local 0.462 · STFiLM-local+capacity 0.472

## Decision tree (applied each wake, idempotent — check before launching)
After the **V0–V5 ablation (seed1 LOOO)** completes, take `best` = max over versions:

- **best ≥ 0.46** → INVEST HARD. Launch best version: LOOO seeds 2,3 + POOLED seeds 1–3. Flag "consider rebuilding paper around MorphoST."
- **0.44 ≤ best < 0.46** → INVEST. Launch best version: LOOO seeds 2,3 + POOLED seeds 1–3.
- **0.40 ≤ best < 0.44** → TUNE. Launch variants of best version: (a) dim=384/n_layers=6, (b) k=12, (c) corr_weight=0.2. One seed LOOO each.
- **best < 0.40** → DIAGNOSE/TUNE hard. Launch dim=384/n_layers=6 + corr_weight=0.0 (pure MSE) + lr=5e-4; check for under/overfit in logs.

## Guardrails
- GPU cap = 5 (account). bigcap2 (3) + jupyter also run; launch only into free slots.
- Stop launching NEW multi-hour jobs after **~07:30** (1h before deadline); use the last hour to
  aggregate + write findings.
- Every result/action appended to the LOG below.
- Idempotency: before submitting a config, check its results dir / squeue name; skip if present.

## LOG
- 03:38 — launched V0–V5 ablation (array 11993205), seed1 LOOO. bigcap2 (8/384) also running.
- 06:44 — ABLATION DONE (seed1 LOOO): V0 0.4566 · V1 0.4573 · V2 0.4540 · V3 **0.4639** · V4 0.4288 · V5 0.4345.
  best=V3=0.4639 (≥0.46). Two surprises: (1) V0 plain MLP already beats STFlow (0.457>0.448);
  (2) the NOVEL parts regress — morph-cond (V4) and gate (V5) HURT; V3 (local+global+slide token,
  no morph-cond) wins. Decision: INVEST in V3. Launched morpho_v3 (11993799): LOOO seeds 2,3 +
  POOLED seeds 1–3. NOTE for user: the morphology-conditioning novelty does not help here — needs
  investigation before it can anchor the paper.
- bigcap2 (8/384): LOOO seed1 5/8, seed2 3/8, seed3 5/8 — still running.
- 07:35 — V3 invest in progress: LOOO seed2 5/8, seed3 6/8 (running); POOLED seeds queued (GPU cap full: bigcap2 x3 + V3 x2). bigcap2 LOOO ~6/8/seed, POOLED not started. Past 07:30 guardrail -> no new launches; monitoring only.
- 10:08 — WINDOW ENDED (past 08:38). FINAL:
  * MorphoST V3 LOOO 3-seed = 0.4597 +/- 0.0056 (0.4639/0.4633/0.4518); POOLED 3-seed = 0.7924 +/- 0.0012.
    Verdict: MorphoST V3 ~ ties STFiLM-local (0.462/0.795), slightly BELOW on both. Beats STFlow (0.448)
    but does NOT beat STFiLM. Novel morph-cond/gate (V4/V5) regressed. => keep STFiLM as main method;
    MorphoST is a competitive-but-not-better, and NOT-more-novel, alternative. Do not rebuild paper around it.
  * bigcap2 (8/384) LOOO 3-seed = 0.4735 vs bigcap (6/256) 0.4724 => +0.0011, capacity PLATEAUED.
    bigcap (6/256) is the sweet spot. POOLED bigcap2 still trickling (1-2/5), not needed.
  Autonomous window closed; no further launches.

## EXTENDED WINDOW (user: "continue running")
Objective: recover the morphology-conditioning NOVELTY (V4/V5 regressed). Test 2 hypotheses on seed1 LOOO:
  H1 local desc redundant w/ local kNN attn -> global-only conditioning
  H2 conditioner overfits small folds       -> cond_dropout=0.3
Bar to beat: V3=0.4639 (seed1). If a conditioned variant > V3, the novelty is real -> scale it (seeds/POOLED).
- launched morpho_rec (see squeue): R1 V4/global, R2 V4/both/dp.3, R3 V5/global/dp.3, R4 V5/both/dp.3.
- 10:48 — recovery partial: R2 (V4 both+dropout0.3)=0.4347 DONE — barely above plain V4 (0.4288), still FAR
  below V3 (0.4639). H2 (overfit/dropout) looks FALSE. R1 (V4 global-only) 7/8, R3/R4 queued. bigcap2 POOLED
  seed3 4/5, others 2/5. No new launches; awaiting R1/R3/R4.

## NEW THREAD — "is FiLM general?" (FiLM_baselines/, user request)
Add STFiLM-style FiLM (none/desc/local) to ALL baselines to test if conditioning helps every backbone
or is STFlow-specific. Launched film_base array 11994673 (HEST LOOO seed1): histogene/hist2st/triplex/
stnet x {none,desc,local} + bleep x {none,desc}. TRIPLEX included per user.
Compare each backbone's desc/local vs its own none. Decision after landing:
  - If FiLM helps most backbones -> strong "conditioning is general" result for the paper.
  - If it only helps some (e.g. helped STFlow, not MorphoST) -> nuanced finding worth reporting.
- 11:47 — RECOVERY CONCLUSIVE: R1 0.4329 · R2 0.4347 · R3 0.4492 · R4 0.4380 — ALL below V3 (0.4639).
  FiLM genuinely does NOT help the MorphoST direct-regressor backbone; neither redundancy (H1) nor
  regularization (H2) recovers it. MorphoST thread CLOSED: V3 (FiLM-free) is its best, ties STFiLM-local
  but no novelty. Keep STFiLM as main method. Interesting contrast for paper: FiLM helps STFlow
  (generative flow-matching, +0.016) but hurts MorphoST (direct regressor) -> FiLM_baselines (running)
  will tell if this generalizes to the other direct-regressor baselines.
- 12:01 — FiLM_baselines partial: HISTOGENE done -> none 0.4359, desc 0.4512 (+0.015), local 0.4486 (+0.013):
  FiLM HELPS histogene (a direct-regressor transformer!). So MorphoST may be the exception, not the rule.
  hist2st mid-run; triplex/stnet/bleep queued. bigcap2 POOLED: seed3=0.7969 done, seeds1,2 at 4/5.
- ~13:xx — FiLM_baselines DONE (HEST LOOO seed1). "IS FiLM GENERAL?" -> CONDITIONAL, and it's a clean story:
    stnet:  none .3587 desc .3408(-.018) local .4004(+.042)   -> local HELPS big (had NO context)
    histog: none .4359 desc .4512(+.015) local .4486(+.013)   -> HELPS
    hist2st:none .4111 desc .4182(+.007) local .4254(+.014)   -> HELPS
    triplex:none .4623 desc .4617(-.001) local .4589(-.003)   -> NEUTRAL (already fuses spot/neigh/global)
    bleep:  none .4227 desc .4187(-.004)                       -> slightly hurts (retrieval)
  PATTERN: FiLM helps backbones LACKING explicit global+local context (stnet/histogene/hist2st/STFlow);
  redundant/neutral for those that ALREADY fuse multi-resolution context (TRIPLEX, MorphoST-V3). This
  EXPLAINS MorphoST's FiLM failure (V3 already has local+global+slide token) — not a bug, redundancy.

## 3-SEED HARDENING + 2 NEW BACKBONES (user: "test with all 3 seeds")
Goal: put error bars on the "FiLM helps" side and defend against "FiLM only rescues weak models" by
adding a STRONG context-poor backbone. Added 2 per-spot (context-free) backbones to baseline_spatial.py:
  deepspace  — DeepSpaCE per-spot MLP (weak, 2nd context-free regressor)
  mlpprobe   — strong deep residual-MLP UNI probe (modern/competitive, still context-free) [key defense]
Both + FiLM wrappers (DeepSpaCEFiLM, MLPProbeFiLM) in film_baselines.py; smoke-tested OK.
Launched film_baselines_3seed.sbatch (array 11995717, 0-59) = 6 reg backbones x {none,desc,local} +
bleep x {none,desc}, seeds 1,2,3, HEST LOOO. Fold-level idempotency skips already-done seed1 originals.
Prediction: deepspace + mlpprobe both show FiLM gains (context-poor) -> 5-6 "helps" cases w/ error bars.

## PER-COHORT HEST TABLE + a100 fix (check-in)
Built MorphoST-V3 per-cohort HEST table (train_hest.py) + feature-matched baselines (hest_baselines.py)
+ vanilla STFlow (hest_stflow.sbatch, --film none). Key infra fix: scavenger tasks CRASHED on v100
(CUDA 'no kernel image', torch lacks sm_70) -> pinned all sbatch to --gres=gpu:a100:1. Grids progressing:
morpho_hest 26/30, hest_base 160/210, film_scav 57/60, hest_stflow 0/30 (training on a100/h200, slow).
Result so far: MorphoST-V3 8-organ avg 0.427 ~ STFlow published 0.428; wins all 8 organs vs feature-matched
baselines. Paper drafted in MorphoST/paper/ (WACV 2027). Caveat logged: MorphoST corr-aware loss vs baselines' MSE.

## FINAL 3-SEED (check-in): 3/4 grids complete
morpho_hest 30/30, hest_base 210/210, film_scav 60/60 DONE (full 3 seeds). hest_stflow 0/30 (long pole,
resubmitted; STFlow ~100ep/fold + scavenger preemption). Table 1 updated to FINAL 3-seed mean±sd for
MorphoST-V3 + all baselines (published STFlow column retained until ours lands). MorphoST-V3 8-organ avg
0.427, wins ALL 8 organs vs feature-matched; on par w/ STFlow published 0.428.
FiLM 3-seed (60/60): FiLM HELPS 5/7 backbones (stnet +.038 p<1e-4, deepspace +.015 p<1e-4, histogene
+.014 p=2e-4, mlpprobe +.006 p=.03, hist2st +.006 p=.07); NEUTRAL triplex (-.002); HURTS bleep (-.009).
mlpprobe (strong probe) now SIGNIFICANTLY helped at 3 seeds -> defends 'not just weak models'.

## STFlow REPRODUCTION COMPLETE — in-house column swapped into Table 1
Found the 'missing' STFlow results: train.py nests them under <cohort>_STFlow_seed<s>/<exp>::<ts>/<cohort>/results_kfold.json
(not the flat path my glob used). All 30 (3 seeds) present on NORMAL partition. Our reproduction MATCHES published
within ±0.01/organ (e.g. IDC .588 vs .587, LUNG .603 vs .610, READ .245 vs .240). OUR STFlow 8-organ avg = 0.425.
=> MorphoST-V3 (0.427) EDGES our STFlow (0.425); 4-4 per-organ split; wins all 8 vs the 6 lightweight baselines.
Aggregator (aggregate_hest_organ.py cohort_seed_mean) + emit_table1.py updated to read nested STFlow; Table 1 in
paper now uses our in-house STFlow (mean±sd), dagger/footnote removed. Cancelled lingering duplicate stflow tasks.
ALL FOUR GRIDS DONE. Paper Table 1 final.
