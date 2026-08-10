"""Generate the STFlow-Table-1-style comprehensive per-organ table (LaTeX table*) from the
aggregated CSVs. Methods are columns, organs are rows, the two benchmarks (HEST, STImage) are
stacked blocks sharing method columns, with an Average row per block. Best per row is bold, the
strongest feature-matched baseline per row is underlined. Writes paper/tab_perorgan.tex."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

# (csv model label -> short column header). Order = table column order.
METHOD_COLS = [
    ("ST-Net", "ST-Net"), ("HisToGene", "HisToG."), ("Hist2ST", "Hist2ST"),
    ("BLEEP", "BLEEP"), ("TRIPLEX", "TRIPLEX"),
    ("STFlow (ICML'25)", "STFlow"), ("STFiLM-local (ours)", "STFiLM"),
]
BASELINES = {"ST-Net", "HisToGene", "Hist2ST", "BLEEP", "TRIPLEX"}
BENCHES = [
    ("HEST", "comparison_LOOO.csv",
     ["kidney", "liver", "lung", "pancreas", "prostate", "skin", "breast", "colorectal"]),
    ("STImage-1K4M", "comparison_stimage_LOOO.csv",
     ["breast", "kidney", "liver", "pancreas", "skin", "prostate", "brain", "heart"]),
]


def fmt(v):
    return f"{v:.3f}" if pd.notna(v) else "--"


def load(csv):
    df = pd.read_csv(os.path.join(ROOT, csv))
    return {r["model"]: r for _, r in df.iterrows()}


def row_cells(d, organ):
    vals = {}
    for label, short in METHOD_COLS:
        r = d.get(label)
        vals[short] = (r[organ] if (r is not None and organ in r and pd.notna(r[organ])) else np.nan)
    present = {s: v for s, v in vals.items() if pd.notna(v)}
    best = max(present, key=present.get) if present else None
    base = {s: v for s, v in vals.items() if pd.notna(v) and
            next(l for l, sh in METHOD_COLS if sh == s) in BASELINES}
    bestbase = max(base, key=base.get) if base else None
    out = []
    for _, short in METHOD_COLS:
        c = fmt(vals[short])
        if short == best and pd.notna(vals[short]):
            c = f"\\textbf{{{c}}}"
        elif short == bestbase and pd.notna(vals[short]):
            c = f"\\underline{{{c}}}"
        out.append(c)
    return out


def main():
    ncol = len(METHOD_COLS)
    lines = [
        "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{l" + "c" * ncol + "}", "\\toprule",
        "Held-out organ & " + " & ".join(sh for _, sh in METHOD_COLS) + " \\\\",
    ]
    for bench, csv, organs in BENCHES:
        path = os.path.join(ROOT, csv)
        lines.append("\\midrule")
        if not os.path.isfile(path):
            lines.append("\\multicolumn{%d}{c}{\\emph{%s: pending}} \\\\" % (ncol + 1, bench))
            continue
        d = load(csv)
        lines.append("\\multicolumn{%d}{l}{\\textbf{%s} (leave-one-organ-out, PCC)} \\\\" % (ncol + 1, bench))
        lines.append("\\midrule")
        for organ in organs:
            lines.append(f"{organ.capitalize()} & " + " & ".join(row_cells(d, organ)) + " \\\\")
        lines.append("\\cmidrule(lr){1-%d}" % (ncol + 1))
        lines.append("Average & " + " & ".join(row_cells(d, "OVERALL")) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Per-organ cross-organ transfer (leave-one-organ-out), Pearson correlation, "
        "3-seed mean, on both benchmarks. Best per row in \\textbf{bold}, strongest feature-matched "
        "baseline \\underline{underlined}. All methods share identical UNI features, leakage-safe "
        "splits, 50-gene panels, and metric. Shaded low-variance organs form the metric floor "
        "(Sec.~\\ref{sec:floor}).}",
        "\\label{tab:perorgan}", "\\end{table*}",
    ]
    out = os.path.join(ROOT, "paper", "tab_perorgan.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
