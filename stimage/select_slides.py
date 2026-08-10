"""Select a leakage-safe cross-organ subset of STImage-1K4M that mirrors our HEST organs.

Strategy: human Visium only (single platform -> consistent spot geometry, avoids a
platform confound), organs overlapping HEST plus a couple of new ones for breadth, capped
per organ, with a spot-count band to avoid tiny/giant slides. Emits a reviewable manifest.
"""
import argparse, os
import pandas as pd

# organs to include; keys are canonical (our) organ names, values are STImage `tissue` labels
ORGAN_MAP = {
    "breast":   ["breast"],
    "kidney":   ["kidney"],
    "liver":    ["liver"],
    "pancreas": ["pancreas"],
    "skin":     ["skin"],
    "prostate": ["prostate"],
    "brain":    ["brain"],       # new organ (not in HEST) -> breadth
    "heart":    ["heart"],       # new organ
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_organ", type=int, default=10)
    ap.add_argument("--spot_min", type=int, default=500)
    ap.add_argument("--spot_max", type=int, default=8000)
    ap.add_argument("--tech", default="Visium")   # ST | Visium | VisiumHD
    ap.add_argument("--species", default="human")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.meta)
    df = df[df.species == args.species]
    df = df[df.tech.str.lower() == args.tech.lower()]
    df = df[(df.spot_num >= args.spot_min) & (df.spot_num <= args.spot_max)]

    rows = []
    for organ, tissues in ORGAN_MAP.items():
        sub = df[df.tissue.isin(tissues)].copy()
        if sub.empty:
            print(f"[warn] no slides for {organ}")
            continue
        # prefer mid-sized slides (closest to median spot_num) for stable folds, then sample
        med = sub.spot_num.median()
        sub["rank"] = (sub.spot_num - med).abs()
        sub = sub.sort_values("rank").head(args.per_organ)
        for _, r in sub.iterrows():
            rows.append({
                "organ": organ,
                "slide": r.slide,
                "tissue": r.tissue,
                "tech": args.tech,
                "spot_num": int(r.spot_num),
                "gene_num": int(r.gene_num),
                "involve_cancer": r.get("involve_cancer", ""),
                "coord": f"{args.tech}/coord/{r.slide}_coord.csv",
                "count": f"{args.tech}/gene_exp/{r.slide}_count.csv",
                "image": f"{args.tech}/image/{r.slide}.png",
            })
    man = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    man.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(man)} slides across {man.organ.nunique()} organs")
    print(man.groupby("organ").agg(n=("slide", "size"),
                                    spot_med=("spot_num", "median")).round(0))

if __name__ == "__main__":
    main()
