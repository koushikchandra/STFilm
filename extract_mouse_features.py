"""Extract UNI features from pre-tiled HEST patches for mouse samples.

Reads dataset/MOUSE_<ORGAN>/patches/<ID>.h5 (img key: N×224×224×3)
and writes embed_dataroot/MOUSE_<ORGAN>/uni_v1_official/fp32/<ID>.h5
(keys: barcodes, coords, embeddings) — same format as hest-bench.
"""
import os
import sys
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "STFlow"))
from stflow.hest_utils.encoder import load_encoder

MOUSE_ORGANS = [
    "MOUSE_BRAIN","MOUSE_LIVER","MOUSE_SKIN","MOUSE_HEART",
    "MOUSE_BOWEL","MOUSE_KIDNEY","MOUSE_MUSCLE",
]
ENCODER = "uni_v1_official"
BATCH_SIZE = 128


class PatchDataset(Dataset):
    def __init__(self, h5_path, transform):
        self.h5_path  = h5_path
        self.transform = transform
        with h5py.File(h5_path, "r") as f:
            bc_key = "barcode" if "barcode" in f else "barcodes"
            self.n      = len(f[bc_key])
            self.coords = f["coords"][:]    # load all coords up front

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            img = f["img"][idx]          # (224,224,3) uint8
        img = self.transform(img)
        return img, self.coords[idx]     # no barcode — object dtype breaks collate


def extract(h5_path, out_path, encoder, transform, device, batch_size):
    ds = PatchDataset(h5_path, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_emb, all_coords = [], []
    encoder.eval()
    with torch.no_grad():
        for imgs, coords in loader:
            emb = encoder(imgs.to(device)).cpu().numpy()
            all_emb.append(emb)
            all_coords.append(coords.numpy())

    emb    = np.concatenate(all_emb,    axis=0).astype(np.float32)
    coords = np.concatenate(all_coords, axis=0).astype(np.int64)

    # Read barcodes directly (bypass collate); key varies across HEST samples
    with h5py.File(str(h5_path), "r") as f:
        bc_key = "barcode" if "barcode" in f else "barcodes"
        raw_bc = f[bc_key][:]   # (N,) bytes or object

    bcs = np.array([b if isinstance(b, bytes) else b.encode()
                    for b in raw_bc.ravel()], dtype=object).reshape(-1, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("embeddings", data=emb)
        f.create_dataset("coords",     data=coords)
        dt = h5py.special_dtype(vlen=bytes)
        f.create_dataset("barcodes",   data=bcs, dtype=dt)
    print(f"  -> {out_path}  shape={emb.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root",  default="dataset")
    p.add_argument("--embed_dataroot", default="embed_dataroot")
    p.add_argument("--weights_root",   default="weights_root")
    p.add_argument("--organ",  default=None, help="Process only this organ. Default: all.")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    dataset_root   = os.path.join(root, args.dataset_root)
    embed_dataroot = os.path.join(root, args.embed_dataroot)
    weights_root   = os.path.join(root, args.weights_root)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    encoder, img_transform, _ = load_encoder(ENCODER, device, weights_root, None)
    encoder.eval()

    # img_transform from load_encoder expects PIL image; patch h5 has numpy uint8
    from PIL import Image
    def transform(arr):
        return img_transform(Image.fromarray(arr))

    organs = [args.organ] if args.organ else MOUSE_ORGANS
    for organ in organs:
        patches_dir = Path(dataset_root) / organ / "patches"
        if not patches_dir.exists():
            print(f"SKIP {organ}: no patches dir")
            continue
        h5_files = sorted(patches_dir.glob("*.h5"))
        print(f"\n=== {organ}: {len(h5_files)} slides ===")
        for h5_path in h5_files:
            sid = h5_path.stem
            out = Path(embed_dataroot) / organ / ENCODER / "fp32" / f"{sid}.h5"
            if out.exists():
                print(f"  SKIP {sid}")
                continue
            print(f"  {sid} ...")
            try:
                extract(h5_path, out, encoder, transform, device, BATCH_SIZE)
            except Exception as e:
                print(f"  ERROR {sid}: {e}")

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
