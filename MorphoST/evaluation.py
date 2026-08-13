"""Evaluation and split utilities for defensible MorphoST experiments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def spot_partition(n: int, seed: int, val_fraction: float):
    """Deterministic complementary (train_idx, val_idx) over n spots, for the single-slide fallback."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = min(max(1, int(round(n * val_fraction))), n - 1)
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def spot_role_index(row, n: int):
    """If `row` carries a spot-level split tag (single-slide fallback), return the spot index array
    for its role ('train'/'val'); else None (use all spots). `row` is a pandas Series."""
    role = row.get("_spot_role") if hasattr(row, "get") else None
    if role is None or (isinstance(role, float) and np.isnan(role)):
        return None
    train_idx, val_idx = spot_partition(n, int(row["_spot_seed"]), float(row["_spot_frac"]))
    return train_idx if role == "train" else val_idx


def train_val_split(df: pd.DataFrame, seed: int, val_fraction: float = 0.15):
    """Hold out slides from the training fold for model selection.

    With >=2 training slides this is a slide-level (inductive) holdout. With a single training slide
    (small HEST cohorts: COAD/HCC/LUNG/SKCM folds) there is no second slide to hold out, so we fall
    back to an inductive SPOT-level split of that one slide: `train` and `val` receive complementary,
    non-overlapping spot subsets (tagged via `_spot_*` columns, honored by the data loaders). No spot
    is ever both trained on and validated on."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must lie in (0, 1)")
    if len(df) >= 2:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))
        n_val = min(max(1, int(round(len(df) * val_fraction))), len(df) - 1)
        return (df.iloc[order[n_val:]].reset_index(drop=True),
                df.iloc[order[:n_val]].reset_index(drop=True))
    # single-slide fallback: spot-level inductive split of the one training slide
    train_row = df.iloc[[0]].copy()
    val_row = df.iloc[[0]].copy()
    for r, role in ((train_row, "train"), (val_row, "val")):
        r["_spot_role"] = role
        r["_spot_seed"] = seed
        r["_spot_frac"] = val_fraction
    return train_row.reset_index(drop=True), val_row.reset_index(drop=True)


def _safe_corr(x, y, fn):
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(fn(x, y)[0])


def expression_metrics(pred, target, genes, slide_ids=None):
    """Compute macro gene-wise metrics and, when available, slide-level metrics."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2:
        raise ValueError(f"Expected matching [spots, genes] arrays, got {pred.shape} and {target.shape}")
    if len(genes) != pred.shape[1]:
        raise ValueError("Gene list length does not match prediction width")

    pearson = np.array([_safe_corr(pred[:, g], target[:, g], pearsonr)
                        for g in range(pred.shape[1])])
    spearman = np.array([_safe_corr(pred[:, g], target[:, g], spearmanr)
                         for g in range(pred.shape[1])])
    mse_gene = np.mean((pred - target) ** 2, axis=0)
    mae_gene = np.mean(np.abs(pred - target), axis=0)
    pred_centered = pred - pred.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = np.mean(pred_centered * target_centered, axis=0)
    ccc = 2 * covariance / (
        pred.var(axis=0) + target.var(axis=0) + (pred.mean(axis=0) - target.mean(axis=0)) ** 2 + 1e-12
    )

    result = {
        "pearson_mean": float(np.nanmean(pearson)),
        "pearson_median": float(np.nanmedian(pearson)),
        "spearman_mean": float(np.nanmean(spearman)),
        "mse": float(np.mean(mse_gene)),
        "rmse": float(np.sqrt(np.mean(mse_gene))),
        "mae": float(np.mean(mae_gene)),
        "ccc_mean": float(np.nanmean(ccc)),
        "pearson_per_gene": pearson.tolist(),
        "spearman_per_gene": spearman.tolist(),
        "mse_per_gene": mse_gene.tolist(),
        "mae_per_gene": mae_gene.tolist(),
        "ccc_per_gene": ccc.tolist(),
        "genes": list(genes),
        "n_spots": int(pred.shape[0]),
    }

    if slide_ids is not None:
        slide_ids = np.asarray(slide_ids).astype(str)
        if len(slide_ids) != len(pred):
            raise ValueError("slide_ids must contain one entry per spot")
        per_slide = []
        for sid in pd.unique(slide_ids):
            mask = slide_ids == sid
            pcc = [_safe_corr(pred[mask, g], target[mask, g], pearsonr)
                   for g in range(pred.shape[1])]
            per_slide.append({
                "slide_id": sid,
                "pearson_mean": float(np.nanmean(pcc)),
                "mse": float(np.mean((pred[mask] - target[mask]) ** 2)),
                "mae": float(np.mean(np.abs(pred[mask] - target[mask]))),
                "n_spots": int(mask.sum()),
            })
        result["per_slide"] = per_slide
        result["pearson_slide_macro"] = float(np.mean([x["pearson_mean"] for x in per_slide]))
    return result


def save_predictions(path, pred, target, coords, slide_ids, genes):
    """Save compact arrays required for later statistics, maps, and alternative metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pred=np.asarray(pred, np.float32), target=np.asarray(target, np.float32),
                        coords=np.asarray(coords, np.float32), slide_ids=np.asarray(slide_ids).astype(str),
                        genes=np.asarray(genes).astype(str))


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Paired slide-level bootstrap for the mean metric difference a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired_bootstrap expects matching one-dimensional arrays")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    return {
        "mean_difference": float(np.mean(a - b)),
        "ci95": np.quantile(diffs, [0.025, 0.975]).tolist(),
        "p_two_sided": float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))),
        "n_pairs": int(len(a)),
        "n_boot": int(n_boot),
    }


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
