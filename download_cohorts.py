"""Download additional HEST-bench cohorts (public repo) into dataset/.

Adds breast (IDC + LYMPH_IDC) and colorectal (COAD + READ) to the existing
6-organ set. Each cohort dir carries patches/, adata/, splits/, var_50genes.json
- exactly the layout train_cross_organ.py / extract_features_uni.py expect.
"""
import os
from huggingface_hub import snapshot_download

REPO = "MahmoodLab/hest-bench"
DEST = "dataset"
COHORTS = ["IDC", "LYMPH_IDC", "COAD", "READ"]

os.makedirs(DEST, exist_ok=True)
patterns = [f"{c}/*" for c in COHORTS]
print(f"Downloading {COHORTS} from {REPO} -> {DEST}/")
snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    allow_patterns=patterns,
    local_dir=DEST,
)
for c in COHORTS:
    ad = len([f for f in os.listdir(os.path.join(DEST, c, "adata")) if f.endswith(".h5ad")]) \
        if os.path.isdir(os.path.join(DEST, c, "adata")) else 0
    pq = len([f for f in os.listdir(os.path.join(DEST, c, "patches")) if f.endswith(".h5")]) \
        if os.path.isdir(os.path.join(DEST, c, "patches")) else 0
    print(f"  {c}: {ad} adata, {pq} patches")
print("DOWNLOAD_DONE")
