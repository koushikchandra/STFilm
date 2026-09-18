"""Benchmark MorphoST parameter count, inference scaling, memory, and invariance."""
import argparse
import json
import math
import time

import numpy as np
import torch

from morphost import MorphoST


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_size(model, n, feat_dim, device, warmup, repeats):
    feats = torch.randn(n, feat_dim, device=device)
    coords = torch.randn(n, 2, device=device) * 100
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            model(feats, coords)
        synchronize(device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(feats, coords)
            synchronize(device)
            times.append(time.perf_counter() - start)
        memory = (torch.cuda.max_memory_allocated(device) / 2**20
                  if device.type == "cuda" else None)
        return {"n_spots": n, "median_seconds": float(np.median(times)),
                "mean_seconds": float(np.mean(times)), "peak_memory_mib": memory,
                "spots_per_second": float(n / np.median(times)), "oom": False}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"n_spots": n, "oom": True}


@torch.no_grad()
def invariance_checks(model, n, feat_dim, device):
    feats = torch.randn(n, feat_dim, device=device)
    coords = torch.randn(n, 2, device=device) * 100
    base = model(feats, coords)
    theta = 0.731
    rotation = torch.tensor([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]], device=device)
    reflection = torch.tensor([[-1.0, 0.0], [0.0, 1.0]], device=device)
    translation = torch.tensor([37.2, -91.4], device=device)
    perm = torch.randperm(n, device=device)
    inv_perm = torch.argsort(perm)
    tests = {
        "translation_max_abs": (model(feats, coords + translation) - base).abs().max().item(),
        "rotation_max_abs": (model(feats, coords @ rotation.T) - base).abs().max().item(),
        "reflection_max_abs": (model(feats, coords @ reflection.T) - base).abs().max().item(),
        "permutation_max_abs": (model(feats[perm], coords[perm])[inv_perm] - base).abs().max().item(),
    }
    return tests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="benchmark_morphost.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sizes", default="500,1000,3000,5000,10000")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--components", default="111")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--k", type=int, default=8)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MorphoST(dim=args.dim, n_layers=args.layers, n_heads=args.heads, k=args.k,
                     n_genes=50, version="V3", components=args.components,
                     dropout=0.0, attn_dropout=0.0).to(device).eval()
    output = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "invariance": invariance_checks(model, 257, 1024, device),
        "scaling": [benchmark_size(model, n, 1024, device, args.warmup, args.repeats)
                    for n in map(int, args.sizes.split(","))],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
