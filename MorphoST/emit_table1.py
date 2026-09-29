"""Emit the LaTeX body of paper Table 1 (8-organ, mean with subscript std over seeds), so the table
regenerates from results as the grids finish. Cell format $mean_{std}$ (STFlow Table-1 convention).
Bold = row max across the method columns. Prints only the tabular rows (paste into main.tex).

STFlow column: uses our in-house per-cohort reproduction (results_hest_stflow) once available; until
then falls back to the published grouped means (flagged with a dagger).
"""
import numpy as np
import aggregate_hest_organ as A

# published STFlow Table-1 (HEST) grouped to 8 organs (Breast=IDC+LYMPH, Colorectal=COAD+READ)
STFLOW_PUB = {"Breast": 0.446, "Prostate": 0.421, "Pancreas": 0.507, "Skin": 0.704,
              "Colorectal": 0.283, "Kidney": 0.332, "Liver": 0.124, "Lung": 0.610}
# display columns in paper order
COLS = ["stnet", "histogene", "hist2st", "bleep", "triplex", "STFlow", "MorphoST-V3"]
NAME = {"stnet": "ST-Net", "histogene": "HisToGene", "hist2st": "Hist2ST", "bleep": "BLEEP",
        "triplex": "TRIPLEX", "STFlow": "STFlow", "MorphoST-V3": "\\morph{} (V3)"}


USE_INHOUSE_STFLOW = True  # our reproduction (results_hest_stflow) is complete -> use it, not STFLOW_PUB


def cell(m, organ, cohorts):
    if m == "STFlow" and not USE_INHOUSE_STFLOW:
        mu = STFLOW_PUB.get(organ, float("nan"))
        return mu, None
    root, tag = A.roots[m]
    mu, sd, n = A.organ_cell(root, cohorts, tag)
    return mu, (sd if n > 1 else None)


def fmt(mu, sd, bold):
    if mu != mu:
        return "--"
    body = f"{mu:.3f}" + (f"_{{{sd:.3f}}}" if sd is not None else "")
    s = f"${body}$"
    return f"\\best{{{s}}}" if bold else s


means = {m: [] for m in COLS}
lines = []
for organ, cohorts in A.ORGANS:
    vals = {m: cell(m, organ, cohorts)[0] for m in COLS}
    best_m = max((m for m in COLS if vals[m] == vals[m]), key=lambda m: vals[m])
    cells = []
    for m in COLS:
        mu, sd = cell(m, organ, cohorts)
        if mu == mu:
            means[m].append(mu)
        cells.append(fmt(mu, sd, m == best_m))
    lines.append(f"{organ:11s} & " + " & ".join(cells) + " \\\\")

# average row
avg = {m: (np.mean(means[m]) if means[m] else float("nan")) for m in COLS}
best_avg = max((m for m in COLS if avg[m] == avg[m]), key=lambda m: avg[m])
acells = [(f"\\best{{${avg[m]:.3f}$}}" if m == best_avg else f"${avg[m]:.3f}$") if avg[m] == avg[m]
          else "--" for m in COLS]

print(" & ".join(["Organ"] + [NAME[m] for m in COLS]) + " \\\\")
print("\\midrule")
for ln in lines:
    print(ln)
print("\\midrule")
print("\\textbf{Average} & " + " & ".join(acells) + " \\\\")
# completeness note (as a LaTeX comment)
def seeds(m):
    root, tag = A.roots[m]
    return [A.organ_cell(root, c, tag)[2] for _, c in A.ORGANS]
print("% seeds/organ: " + "; ".join(f"{m}:{seeds(m)}" for m in COLS if m != "STFlow"))
