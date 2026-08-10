### Model size / conditioning overhead (UNI, 50 genes, hidden 128, 4 layers)

**Accepted baselines** (params from STFlow Tab.~6):

| model | #params (M) |
|---|---|
| ST-Net | 0.051 |
| BLEEP | 0.670 |
| HisToGene | 149.046 |
| TRIPLEX | 13.767 |
| STFlow | 1.147 |

**Our conditioner variants** (STFlow backbone + FiLM):

| variant | #params (M) | Δ vs none (params) |
|---|---|---|
| none | 1.412 | — |
| meta | 1.413 | +1,152 |
| desc | 1.543 | +131,200 |
| hybrid | 2.074 | +662,016 |
| local | 1.674 | +262,400 |
| moe | 3.005 | +1,593,360 |

The recommended conditioner, \textsc{local}, adds only 262,400 parameters (+18.6\%) over the 1.412M STFlow backbone, for a 1.674M model --- 1--2 orders of magnitude smaller than TRIPLEX (13.8M) and HisToGene (149M). \textsc{meta} is essentially free (+1,152); only \textsc{moe} inflates the model (3.005M), consistent with its lack of benefit.
