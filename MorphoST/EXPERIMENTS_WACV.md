# WACV 2027 experiment audit

## Protocol warning

The legacy `results_hest*`, `results_morphost*`, `results_hest_baselines`, and cross-organ baseline
directories used the outer test fold for epoch selection. They are exploratory results and must not
be reported as final test performance.

Publishable reruns use an inner slide-level validation split (15% of each outer training fold), save
the validation-selected checkpoint, evaluate the outer test fold once, and export predictions. Their
result directories begin with `results_corrected_`.

## Submitted corrected experiments

| Job | Output | Purpose |
|---|---|---|
| `morphost_corrected_hest.sbatch` | `results_corrected_hest_{mse,corr}` | HEST main result and loss control |
| `corrected_baselines_hest.sbatch` | `results_corrected_hest_baselines` | seven feature-matched baselines |
| `loss_control_baselines.sbatch` | `results_corrected_hest_baselines_corr` | shared PCC-aware loss for strong baselines |
| `morphost_factorial.sbatch` | `results_corrected_factorial` | all 8 local/global/slide combinations |
| `morphost_corrected_stimage.sbatch` | `results_corrected_stimage` | independent STImage benchmark |
| `morphost_geometry.sbatch` | `results_corrected_geometry` | RBF distance, distance-free kNN, absolute XY |
| `morphost_sensitivity.sbatch` | `results_corrected_sensitivity` | k, depth, width, and RBF sensitivity |
| `morphost_benchmark.sbatch` | `benchmark_morphost_a100.json` | latency, peak memory, and invariance |

## Analysis commands

```bash
cd MorphoST
python analyze_corrected.py results_corrected_hest_corr results_corrected_hest_mse \
  results_corrected_hest_baselines results_corrected_hest_baselines_corr

python analyze_corrected.py results_corrected_factorial \
  --compare LOOO_C111 LOOO_C000 --output_dir analysis_factorial

python plot_expression_maps.py \
  results_corrected_hest_corr/CCRCC_V3_seed1/fold_0/test_predictions.npz \
  --output paper/expression_map_example.png
```

## Completed efficiency result

On an NVIDIA A100 80GB PCIe, MorphoST has 4,751,942 parameters. Synthetic-slide inference used
approximately 0.0094 s / 335 MiB at 3,000 spots, 0.0213 s / 842 MiB at 5,000 spots, and 0.0647 s /
3,183 MiB at 10,000 spots. Coordinate-transform deviations were below `6e-5`; permutation deviation
was below `6e-7`. These are model-only synthetic benchmarks and must be labeled as such.

## Outstanding prerequisites

- Corrected STFlow reruns require modifying its training loop to use an inner validation split and
  to export predictions; do not compare corrected MorphoST against legacy test-selected STFlow.
- Larger shared gene panels are currently blocked by inconsistent gene identifiers/coverage across
  cohorts. A trial availability-filtered build retained only 13 genes in some LOOO folds. Harmonize
  Ensembl identifiers and duplicated symbols before attempting 250/1,000-gene cross-organ panels.
- Patient metadata was not supplied to `make_cross_organ_splits.py`; the existing pooled splits group
  by sample ID. Obtain the HEST metadata and rebuild splits using the patient column before describing
  pooled evaluation as patient-disjoint.

## Single-slide-cohort fix (2026-08-12)

The nested-validation correction crashed on COAD/HCC/LUNG/SKCM: these cohorts have folds whose outer
training set is a SINGLE slide, so a slide-level inner-validation holdout is impossible
(`train_val_split` raised "At least two training slides are required"). 24 MorphoST + 84 baseline
tasks failed for exactly these 4 cohorts; the other 6 were unaffected.

Fix (`MorphoST/evaluation.py`): when a training fold has <2 slides, fall back to an inductive
spot-level split of the one slide -- train/val get complementary, non-overlapping spot subsets
(85%/15%), tagged via `_spot_*` columns and honored by both loaders (`MorphoST/data.py`,
`baseline_spatial.py`). No spot is both trained on and validated on. Symmetric across MorphoST and all
baselines. Reruns: rerun_smallcohort_{morphost,baselines,lossctrl}.sbatch (arrays 12002529/30/31).
