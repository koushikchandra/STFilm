"""Minimal, self-contained data helpers vendored for the MIST code release.

These three functions are all that ``data.py`` needs to read HEST-1k style inputs:
  - ``read_assets_from_h5``   : read spot features/coords/barcodes from an .h5 file
  - ``load_adata``            : read + (optionally) normalize expression from an .h5ad
  - ``get_normalize_method``  : resolve a normalization name (e.g. "log1p") to a callable

Only pip dependencies are used (h5py, numpy, scanpy). No project-specific paths.
"""
import h5py
import numpy as np
import scanpy as sc


# --------------------------------------------------------------------------- #
# HDF5 reading
# --------------------------------------------------------------------------- #
def read_assets_from_h5(h5_path, keys=None, skip_attrs=False, skip_assets=False):
    """Read datasets (and optionally attributes) from an HDF5 file into dicts."""
    assets, attrs = {}, {}
    with h5py.File(h5_path, "r") as f:
        if keys is None:
            keys = list(f.keys())
        for key in keys:
            if not skip_assets:
                assets[key] = f[key][:]
            if not skip_attrs and f[key].attrs is not None:
                attrs[key] = dict(f[key].attrs)
    return assets, attrs


# --------------------------------------------------------------------------- #
# Expression normalization
# --------------------------------------------------------------------------- #
def identity(adata):
    return adata.copy()


def log1p(adata):
    process_data = adata.copy()
    sc.pp.log1p(process_data)
    return process_data


def get_normalize_method(normalize_method, **kwargs):
    """Resolve a normalization name to a callable ``adata -> adata``."""
    if normalize_method is None:
        return None
    if normalize_method == "log1p":
        return log1p
    if normalize_method in ("identity", "raw", "none"):
        return identity
    raise ValueError(f"Unknown normalize method: {normalize_method}")


def normalize_adata(adata):
    """Default normalization: log1p of the per-spot expression."""
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)
    sc.pp.log1p(filtered_adata)
    return filtered_adata


# --------------------------------------------------------------------------- #
# AnnData loading
# --------------------------------------------------------------------------- #
def load_adata(expr_path, genes=None, barcodes=None, normalize_method=normalize_adata):
    """Read an .h5ad, optionally subset barcodes/genes, normalize, return a DataFrame."""
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if genes is not None:
        adata = adata[:, genes]
    if normalize_method is not None:
        adata = normalize_method(adata)
    return adata.to_df()
