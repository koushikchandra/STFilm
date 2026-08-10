# STFiLM — WACV 2026 paper (Overleaf)

## Files
- `main.tex` — full paper draft: abstract, intro, related work, method, experiments, results, discussion, conclusion. All result tables are populated with our real 3-seed numbers.
- `references.bib` — bibliography. **Verify author lists / venues / page numbers before submission** (some entries are best-effort).

## How to set up on Overleaf
1. On Overleaf: **New Project → Templates → search "WACV 2026"** (or the official WACV author kit). This provides `wacv.sty` and `ieeenat_fullname.bst`, which `main.tex` expects.
2. In that project, **replace the template's `main.tex`** with this `main.tex`, and **upload `references.bib`**.
3. Compile (pdfLaTeX). `\documentclass[review]{wacv}` gives the anonymized review format with line numbers. For the camera-ready, switch `[review]` → `[final]` (or the kit's `\wacvfinalcopy`), and fill in `\author{...}` and `\wacvPaperID`.

If the WACV style file is unavailable, the paper also compiles against the CVPR kit with minimal changes (swap `wacv` → `cvpr` documentclass and the `.bst`), since WACV and CVPR share the same LaTeX kit.

## Numbers in the tables (source of truth)
All numbers trace to files in the repo root:
- Main comparison (Table 1): `comparison_LOOO.csv`, `comparison_POOLED.csv` (`aggregate_comparison.py`), plus `hybrid`/`meta` overall means computed from `results_hybrid_uni8/` and `results_cross_organ_uni8/`.
- Significance (Table 2): `paired_significance.py`.
- Metric floor + detection-filtered numbers (Table 3, text): `rescore_filtered.py`.
- Full prose writeup with all details: `../RESULTS.md`.

## TODO before submission
- [x] Biomarker downstream table (Table 4) — `biomarker_eval.py`; FiLM improves 11/11 markers.
- [x] Seed-ensembling numbers folded into the results text — `ensemble_seeds.py`.
- [ ] Add a qualitative figure (per-organ bars; predicted-vs-true expression maps).
- [ ] Add STImage-1K4M results (second benchmark) — the one remaining STFlow eval we don't run.
- [ ] Verify all `references.bib` entries.
