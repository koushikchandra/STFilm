"""Download selected Mus musculus samples from MahmoodLab/hest.

Downloads both st/<ID>.h5ad (expression) and patches/<ID>.h5 (image patches)
into dataset/MOUSE_<ORGAN>/{adata,patches}/ mirroring the hest-bench layout.
"""
import os
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_TOKEN_PATH = os.path.join(os.path.dirname(__file__), ".hf/token")
REPO_ID = "MahmoodLab/hest"
REPO_TYPE = "dataset"

# 7 organs × 59 slides total — Visium / Visium HD only
MOUSE_SELECTION = {
    "MOUSE_BRAIN":  ["TENX88","TENX87","TENX86","TENX85","TENX84",
                     "TENX83","TENX82","TENX80","TENX79","TENX78"],
    "MOUSE_LIVER":  ["NCBI844","NCBI843","NCBI842","NCBI841","NCBI840",
                     "NCBI839","NCBI838","NCBI837","NCBI836","NCBI835"],
    "MOUSE_SKIN":   ["ZEN81","ZEN80","ZEN79","ZEN78","ZEN65",
                     "NCBI689","NCBI688","NCBI687"],
    "MOUSE_HEART":  ["MEND28","MEND27","MEND26","MEND25",
                     "NCBI787","NCBI786","NCBI543","NCBI542"],
    "MOUSE_BOWEL":  ["MEND32","MEND31","MEND30","MEND29",
                     "NCBI729","NCBI728","NCBI686","NCBI685"],
    "MOUSE_KIDNEY": ["TENX45","TENX22","NCBI598","NCBI597","NCBI596"],
    "MOUSE_MUSCLE": ["NCBI796","NCBI795","NCBI794","NCBI652","NCBI651",
                     "NCBI650","NCBI649","NCBI648","NCBI647","NCBI646"],
}


def download_sample(sample_id, organ, dataset_root, token):
    adata_dir   = Path(dataset_root) / organ / "adata"
    patches_dir = Path(dataset_root) / organ / "patches"
    adata_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    adata_dst   = adata_dir   / f"{sample_id}.h5ad"
    patches_dst = patches_dir / f"{sample_id}.h5"

    for dst, hf_path in [(adata_dst, f"st/{sample_id}.h5ad"),
                          (patches_dst, f"patches/{sample_id}.h5")]:
        if dst.exists():
            print(f"  SKIP {dst.name} (exists)")
            continue
        t0 = time.time()
        tmp = hf_hub_download(
            repo_id=REPO_ID, repo_type=REPO_TYPE,
            filename=hf_path,
            local_dir=str(dst.parent.parent / "_hf_cache"),
            token=token,
        )
        # move to final location
        import shutil
        shutil.move(tmp, str(dst))
        elapsed = time.time() - t0
        sz_mb = dst.stat().st_size / 1e6
        print(f"  OK  {dst.name}  {sz_mb:.0f} MB  ({elapsed:.0f}s)")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default="dataset")
    p.add_argument("--organ", default=None,
                   help="Download only this organ key (e.g. MOUSE_BRAIN). Default: all.")
    args = p.parse_args()

    token = open(HF_TOKEN_PATH).read().strip()
    dataset_root = os.path.join(os.path.dirname(__file__), args.dataset_root)

    selection = MOUSE_SELECTION
    if args.organ:
        selection = {args.organ: MOUSE_SELECTION[args.organ]}

    total = sum(len(v) for v in selection.values())
    done = 0
    for organ, ids in selection.items():
        print(f"\n=== {organ} ({len(ids)} slides) ===")
        for sid in ids:
            done += 1
            print(f"[{done}/{total}] {sid}")
            try:
                download_sample(sid, organ, dataset_root, token)
            except Exception as e:
                print(f"  ERROR: {e}")

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
