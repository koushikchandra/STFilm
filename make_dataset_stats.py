"""T7 — dataset statistics for both benchmarks (STFlow Table-7/8 style). Derives per-organ slide
counts and average spots-per-slide directly from the cross-organ split CSVs + the embedding h5s,
so the numbers match exactly what the models were trained/evaluated on. Emits comparison_datasets.md."""
import os, glob, json
import numpy as np
import pandas as pd
from stflow.hest_utils.file_utils import read_assets_from_h5

ROOT = os.path.dirname(os.path.abspath(__file__))

# HEST cross-organ: organ -> contributing HEST cohorts + platform (from HEST-bench)
HEST_META = {
    "breast":     (["IDC", "LYMPH_IDC"], "Xenium/Visium"),
    "prostate":   (["PRAD"],             "Visium"),
    "pancreas":   (["PAAD"],             "Xenium"),
    "skin":       (["SKCM"],             "Xenium"),
    "colorectal": (["COAD", "READ"],     "Visium"),
    "kidney":     (["CCRCC"],            "Visium"),
    "liver":      (["HCC"],              "Visium"),
    "lung":       (["LUNG"],             "Xenium"),
}


def avg_spots(embed_root, feature, sample_rows):
    """Mean #spots across the given (cohort, sid) embedding h5 files."""
    ns = []
    for cohort, sid in sample_rows:
        h5 = os.path.join(ROOT, embed_root, cohort, feature, "fp32", f"{sid}.h5")
        if os.path.isfile(h5):
            dd, _ = read_assets_from_h5(h5)
            ns.append(len(dd["barcodes"].flatten()))
    return (np.mean(ns), np.sum(ns)) if ns else (float("nan"), 0)


def hest_rows():
    rows = []
    for organ, (cohorts, plat) in HEST_META.items():
        # each organ's LOOO test CSV lists all its slides
        df = pd.read_csv(os.path.join(ROOT, "cross_organ_splits8", "LOOO", "splits", f"test_{organ}.csv"))
        samples = [(r["patches_path"].split("/")[0], r["sample_id"]) for _, r in df.iterrows()]
        mean_s, tot_s = avg_spots("embed_dataroot", "uni_v1_official", samples)
        rows.append([organ, "+".join(cohorts), plat, len(df), int(tot_s), round(mean_s)])
    return rows


def stimage_rows():
    man = pd.read_csv(os.path.join(ROOT, "stimage", "manifest.csv"))
    rows = []
    for organ, g in man.groupby("organ"):
        rows.append([organ, g["tissue"].iloc[0], g["tech"].iloc[0], len(g),
                     int(g["spot_num"].sum()), round(g["spot_num"].mean())])
    return rows


def write_table(f, title, header, rows):
    f.write(f"\n**{title}**\n\n")
    f.write("| " + " | ".join(header) + " |\n")
    f.write("| " + " | ".join("---" for _ in header) + " |\n")
    for r in rows:
        f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    tot_slides = sum(r[3] for r in rows); tot_spots = sum(r[4] for r in rows)
    f.write(f"| **total** | | | **{tot_slides}** | **{tot_spots}** | |\n")


def main():
    out = os.path.join(ROOT, "comparison_datasets.md")
    with open(out, "w") as f:
        f.write("### Dataset statistics (both cross-organ benchmarks; 50-gene panels, UNI features)\n")
        write_table(f, "HEST cross-organ (8 organ groups)",
                    ["organ", "HEST cohorts", "platform", "#slides", "#spots", "avg spots"],
                    hest_rows())
        write_table(f, "STImage-1K4M selection (8 organs)",
                    ["organ", "tissue", "platform", "#slides", "#spots", "avg spots"],
                    stimage_rows())
    print(open(out).read())
    print("wrote comparison_datasets.md")


if __name__ == "__main__":
    main()
