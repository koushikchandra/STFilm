"""Cross-organ data loading for MorphoST. Reuses STFlow's low-level IO helpers (h5/h5ad readers)
only -- none of STFlow's *model* code -- so the backbone stays independent while consuming the
exact same UNI features, leakage-safe splits, and 50-gene panels as the STFiLM experiments."""
import os
import json
import numpy as np
import scanpy as sc
import torch

from stflow.hest_utils.st_dataset import load_adata
from stflow.hest_utils.file_utils import read_assets_from_h5
from stflow.data.normalize_utils import get_normalize_method


def _load_adata(h5ad, genes, barcodes, normalize_method):
    """load_adata wrapper that strips GRCm38_ prefix from mouse MEND-series samples."""
    import pandas as pd
    adata = sc.read_h5ad(h5ad)
    if len(adata.var_names) > 0 and adata.var_names[0].startswith("GRCm38_"):
        adata.var_names = adata.var_names.str.replace("GRCm38_", "", regex=False)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        available = [g for g in genes if g in adata.var_names]
        adata = adata[:, available]
    if normalize_method is not None:
        adata = normalize_method(adata)
    df = adata.to_df()
    if genes is not None:
        missing = [g for g in genes if g not in df.columns]
        if missing:
            df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing)], axis=1)
        df = df[genes]
    return df

# organ groupings (mirror make_cross_organ_splits.py / train_cross_organ.py)
COHORT_TO_GROUP = {
    "CCRCC": "kidney", "COAD": "colorectal", "READ": "colorectal", "HCC": "liver",
    "IDC": "breast", "LYMPH_IDC": "breast", "LUNG": "lung", "PAAD": "pancreas",
    "PRAD": "prostate", "SKCM": "skin",
}


def load_sample(row, feature_encoder, embed_dataroot, source_dataroot, gene_list,
                normalize_method="log1p", device="cuda", cohort=None):
    """Return (feats[N,1024], coords[N,2], expr[N,G]) tensors on `device` for one slide.

    In the cross-organ splits the CSV `patches_path` is prefixed with the cohort ("PRAD/patches/..").
    In the STFlow-style per-cohort splits it is not ("patches/.."), so the caller passes `cohort`."""
    cohort = cohort if cohort is not None else row["patches_path"].split("/")[0]
    sid = row["sample_id"]
    h5 = os.path.join(embed_dataroot, cohort, feature_encoder, f"fp32/{sid}.h5")
    h5ad = os.path.join(source_dataroot, cohort, f"adata/{sid}.h5ad")
    d, _ = read_assets_from_h5(h5)
    barcodes = d["barcodes"].flatten().astype(str).tolist()
    coords = d["coords"]
    feats = d["embeddings"]
    norm = normalize_method if callable(normalize_method) else get_normalize_method(normalize_method)
    expr = _load_adata(h5ad, genes=gene_list, barcodes=barcodes, normalize_method=norm).values
    from evaluation import spot_role_index  # single-slide inner-val fallback (no-op for full slides)
    idx = spot_role_index(row, len(feats))
    if idx is not None:
        feats, coords, expr = feats[idx], coords[idx], expr[idx]
    return (torch.from_numpy(feats).float().to(device),
            torch.from_numpy(coords).float().to(device),
            torch.from_numpy(expr).float().to(device))


def load_gene_list(splits_root, regime, fold):
    """Leakage-safe per-fold gene panel (genes_<fold>.json)."""
    path = os.path.join(splits_root, regime, f"genes_{fold}.json") if regime else \
           os.path.join(splits_root, f"genes_{fold}.json")
    if os.path.isfile(path):
        return json.load(open(path))["genes"]
    raise FileNotFoundError(f"no gene panel {path}")
