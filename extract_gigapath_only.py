"""Standalone GigaPath feature extraction — bypasses benchmark.py's buggy probe/save step
(cuml import + float32-JSON crash). Only extracts tile features and writes them to
embed_dataroot/<cohort>/gigapath/fp32/<sample>.h5, exactly matching the layout train_cross_organ
expects. Skips already-extracted slides."""
import os
import glob
import argparse

import numpy as np
import torch
from tqdm import tqdm

from stflow.hest_utils.encoder import load_encoder
from stflow.hest_utils.st_dataset import H5TileDataset
from stflow.hest_utils.file_utils import save_hdf5

COHORTS = ["CCRCC", "COAD", "HCC", "IDC", "LUNG", "LYMPH_IDC", "PAAD", "PRAD", "READ", "SKCM"]


def embed_tiles(dataloader, model, embedding_save_path, device, precision=torch.float32):
    def post_collate_fn(batch):
        if batch["imgs"].dim() == 5:
            batch["imgs"] = batch["imgs"].squeeze(0)
        if batch["coords"].dim() == 3:
            batch["coords"] = batch["coords"].squeeze(0)
        return batch

    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        batch = post_collate_fn(batch)
        imgs = batch["imgs"].to(device)
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
            embeddings = model(imgs)
        mode = "w" if batch_idx == 0 else "a"
        asset = {"embeddings": embeddings.cpu().numpy()}
        asset.update({k: np.array(v) for k, v in batch.items() if k != "imgs"})
        save_hdf5(embedding_save_path, asset_dict=asset, mode=mode)
    return embedding_save_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="gigapath")
    p.add_argument("--weights_root", default="weights_root")
    p.add_argument("--source_dataroot", default="dataset")
    p.add_argument("--embed_dataroot", default="embed_dataroot")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, tfm, _ = load_encoder(args.encoder, device, args.weights_root, args.weights_root)
    print(f"[extract] loaded {args.encoder} on {device}")

    total_new = 0
    for c in COHORTS:
        outdir = os.path.join(args.embed_dataroot, c, args.encoder, "fp32")
        os.makedirs(outdir, exist_ok=True)
        tiles = sorted(glob.glob(os.path.join(args.source_dataroot, c, "patches", "*.h5")))
        done = 0
        for h5 in tqdm(tiles, desc=f"{c}", ncols=90):
            sid = os.path.basename(h5)[:-3]
            emb = os.path.join(outdir, f"{sid}.h5")
            if os.path.isfile(emb):
                done += 1
                continue
            ds = H5TileDataset(h5, chunk_size=args.batch_size, img_transform=tfm)
            dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
            embed_tiles(dl, encoder, emb, device)
            total_new += 1
        print(f"[extract] {c}: {len(tiles)} slides ({done} already done)")
    print(f"[extract] DONE, {total_new} new slides extracted")


if __name__ == "__main__":
    main()
