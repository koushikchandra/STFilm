"""
Concatenate UNI (1024-dim) and CONCH (512-dim) embeddings into uni_conch (1536-dim) h5 files.
Run after CONCH extraction is complete for all cohorts.

Usage:
    python concat_uni_conch.py [--cohorts CCRCC COAD ...] [--embed_dataroot embed_dataroot]
"""
import argparse
import h5py
import numpy as np
import os
from pathlib import Path

COHORTS = ["CCRCC", "COAD", "HCC", "IDC", "LUNG", "LYMPH_IDC", "PAAD", "PRAD", "READ", "SKCM"]


def concat_sample(uni_path: str, conch_path: str, out_path: str) -> bool:
    with h5py.File(uni_path, "r") as fu, h5py.File(conch_path, "r") as fc:
        uni_emb   = fu["embeddings"][:]   # (N, 1024)
        conch_emb = fc["embeddings"][:]   # (N, 512)
        barcodes  = fu["barcodes"][:]
        coords    = fu["coords"][:]

        assert uni_emb.shape[0] == conch_emb.shape[0], (
            f"spot count mismatch: {uni_path} {uni_emb.shape[0]} vs {conch_path} {conch_emb.shape[0]}"
        )
        # verify barcodes match
        assert np.array_equal(fu["barcodes"][:], fc["barcodes"][:]), \
            f"barcode mismatch between UNI and CONCH for {Path(uni_path).stem}"

        combined = np.concatenate([uni_emb, conch_emb], axis=1)  # (N, 1536)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as fout:
        fout.create_dataset("embeddings", data=combined, compression="gzip")
        fout.create_dataset("barcodes",   data=barcodes)
        fout.create_dataset("coords",     data=coords)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", nargs="+", default=COHORTS)
    parser.add_argument("--embed_dataroot", default="embed_dataroot")
    args = parser.parse_args()

    total_ok, total_skip, total_err = 0, 0, 0

    for cohort in args.cohorts:
        uni_dir   = Path(args.embed_dataroot) / cohort / "uni_v1_official" / "fp32"
        conch_dir = Path(args.embed_dataroot) / cohort / "conch_v1_official" / "fp32"
        out_dir   = Path(args.embed_dataroot) / cohort / "uni_conch" / "fp32"

        if not uni_dir.exists():
            print(f"[SKIP] {cohort}: no UNI features at {uni_dir}")
            continue
        if not conch_dir.exists():
            print(f"[SKIP] {cohort}: no CONCH features at {conch_dir} — run extract_conch.sbatch first")
            total_skip += 1
            continue

        samples = [p.stem for p in uni_dir.glob("*.h5")]
        print(f"[{cohort}] {len(samples)} samples")

        for sample in samples:
            uni_path   = str(uni_dir   / f"{sample}.h5")
            conch_path = str(conch_dir / f"{sample}.h5")
            out_path   = str(out_dir   / f"{sample}.h5")

            if os.path.exists(out_path):
                total_ok += 1
                continue

            if not os.path.exists(conch_path):
                print(f"  [WARN] missing CONCH for {sample}")
                total_skip += 1
                continue

            try:
                concat_sample(uni_path, conch_path, out_path)
                total_ok += 1
            except Exception as e:
                print(f"  [ERR] {sample}: {e}")
                total_err += 1

    print(f"\nDone: {total_ok} ok, {total_skip} skipped, {total_err} errors")


if __name__ == "__main__":
    main()
