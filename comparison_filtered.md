### Variance/detection-filtered LOOO rescoring (leakage-safe; drop genes <10% detected in held-out organ)
```
Detection filter: keep genes expressed in >= 10% of held-out spots

Genes kept per organ:
  kidney       8/11 kept  (dropped: ['CCR7', 'KIT', 'TNFRSF17'])
  prostate     2/11 kept  (dropped: ['CCR7', 'CD3E', 'CD79A', 'GZMK', 'MKI67', 'CD8A', 'GPR183', 'KIT', 'TNFRSF17'])
  liver        5/11 kept  (dropped: ['CCR7', 'CD79A', 'GZMK', 'CD8A', 'KIT', 'TNFRSF17'])
  lung        11/11 kept  (dropped: [])
  skin        11/11 kept  (dropped: [])
  colorectal  11/11 kept  (dropped: [])
  breast      10/10 kept  (dropped: [])
  pancreas     8/ 8 kept  (dropped: [])

model            LOOO raw  LOOO filt   per-organ filtered (kept-gene mean)
ST-Net             0.3595     0.3877   kid=0.12 pro=0.25 liv=0.09 lun=0.64 ski=0.69 col=0.54 bre=0.29 pan=0.49
HisToGene          0.4379     0.4612   kid=0.11 pro=0.20 liv=0.14 lun=0.69 ski=0.76 col=0.74 bre=0.54 pan=0.51
Hist2ST            0.4087     0.4438   kid=0.13 pro=0.32 liv=0.12 lun=0.59 ski=0.72 col=0.71 bre=0.49 pan=0.47
BLEEP              0.4231     0.4524   kid=0.13 pro=0.31 liv=0.05 lun=0.63 ski=0.75 col=0.65 bre=0.55 pan=0.55
TRIPLEX            0.4623     0.5003   kid=0.15 pro=0.37 liv=0.11 lun=0.70 ski=0.78 col=0.74 bre=0.58 pan=0.56
STFlow none        0.4477     0.4789   kid=0.14 pro=0.31 liv=0.09 lun=0.67 ski=0.77 col=0.74 bre=0.56 pan=0.55
STFiLM desc        0.4590     0.4910   kid=0.17 pro=0.33 liv=0.09 lun=0.69 ski=0.78 col=0.73 bre=0.56 pan=0.58
STFiLM local       0.4624     0.4950   kid=0.19 pro=0.34 liv=0.07 lun=0.68 ski=0.79 col=0.74 bre=0.57 pan=0.59
STFlow MoE         0.4488     0.4811   kid=0.16 pro=0.32 liv=0.07 lun=0.67 ski=0.78 col=0.73 bre=0.55 pan=0.57
```
