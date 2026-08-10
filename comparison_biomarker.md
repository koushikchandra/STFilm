### Biomarker-level Pearson (cross-organ folds × 3 seeds × both regimes)
```
Per-biomarker Pearson (mean over all cross-organ folds x 3 seeds x both regimes)

gene       ST-Net  HisToG  Hist2S   BLEEP  TRIPLE    none    desc   local  Δ(loc-none)  annotation
--------------------------------------------------------------------------------------------------
MKI67       0.523   0.651   0.631   0.624   0.666   0.651   0.652   0.659       +0.008  Ki-67 proliferation index (prognostic; grading)
CD3E        0.417   0.607   0.587   0.599   0.637   0.624   0.640   0.640       +0.016  pan T-cell (immune infiltration)
ACTA2       0.479   0.587   0.569   0.613   0.619   0.613   0.620   0.621       +0.008  myofibroblast / stroma (aSMA)
GPR183      0.348   0.609   0.587   0.579   0.604   0.603   0.607   0.609       +0.006  immune cell positioning (EBI2)
CXCR4       0.403   0.592   0.573   0.542   0.597   0.589   0.598   0.604       +0.016  chemokine receptor (metastasis)
CD8A        0.374   0.571   0.553   0.555   0.601   0.588   0.595   0.603       +0.015  cytotoxic T-cell (immune infiltration; immunotherapy)
GZMK        0.343   0.541   0.517   0.544   0.569   0.560   0.575   0.575       +0.015  cytotoxic granzyme-K T-cell
CCR7        0.335   0.542   0.520   0.532   0.563   0.550   0.566   0.566       +0.016  naive/central-memory T-cell homing
CD79A       0.364   0.537   0.509   0.530   0.569   0.541   0.561   0.557       +0.016  B-cell
TNFRSF17    0.267   0.518   0.505   0.511   0.535   0.521   0.530   0.531       +0.009  BCMA (myeloma target)
KIT         0.309   0.505   0.487   0.491   0.514   0.512   0.517   0.522       +0.011  c-Kit (GIST/melanoma drug target)

Mean over biomarkers:
  ST-Net         0.3784
  HisToGene      0.5691
  Hist2ST        0.5489
  BLEEP          0.5564
  TRIPLEX        0.5886
  STFlow none    0.5775
  STFiLM desc    0.5872
  STFiLM local   0.5898

STFiLM desc beats none on 11/11 biomarkers (mean Δ = +0.0097)

STFiLM local beats none on 11/11 biomarkers (mean Δ = +0.0123)
```
