### SE(2)-invariance check: max |pred(x) - pred(Rx+t)| under a random rotation ($137^\circ$) + translation, per conditioner (random weights).

| conditioner | max abs deviation | pred scale | invariant? |
|---|---|---|---|
| none | 1.30e-06 | 0.240 | yes |
| meta | 0.00e+00 | 0.235 | yes |
| desc | 0.00e+00 | 0.230 | yes |
| hybrid | 0.00e+00 | 0.232 | yes |
| local | 0.00e+00 | 0.232 | yes |
| moe | 0.00e+00 | 0.231 | yes |

All conditioner variants leave the prediction invariant to SE(2) transforms of the spot layout (deviation at floating-point noise level, $\ll$ the $O(1)$ prediction scale), confirming FiLM modulates only the invariant scalar stream and never the frame-averaged geometric features.
