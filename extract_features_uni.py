"""UNI (uni_v1_official) feature extractor — same pipeline as extract_features.py.

UNI is gated: you must (1) accept the license at https://huggingface.co/MahmoodLab/UNI
and (2) be logged in (`huggingface-cli login` or HF_TOKEN) before running this.

Writes embed_dataroot/<dataset>/uni_v1_official/fp32/<sample_id>.h5 with
{embeddings[N,1024], coords, barcodes} — exactly what data/dataset.py reads.
"""
import os
import glob
import timm
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from stflow.hest_utils.st_dataset import H5TileDataset
from stflow.hest_utils.utils import get_constants, get_eval_transforms
from stflow.hest_utils.file_utils import save_hdf5

SRC = "dataset"
EMB = "embed_dataroot"
ENC = "uni_v1_official"
WEIGHTS_DIR = "weights_root/uni"
DATASETS = ["LUNG", "HCC", "CCRCC", "PAAD", "PRAD", "SKCM",
            "IDC", "LYMPH_IDC", "COAD", "READ"]
CHUNK = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed_tiles(dataloader, model, embedding_save_path, device):
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        if batch["imgs"].dim() == 5:
            batch["imgs"] = batch["imgs"].squeeze(0)
        if batch["coords"].dim() == 3:
            batch["coords"] = batch["coords"].squeeze(0)
        imgs = batch["imgs"].to(device)
        with torch.inference_mode():
            embeddings = model(imgs)
        mode = "w" if batch_idx == 0 else "a"
        asset_dict = {"embeddings": embeddings.cpu().numpy()}
        asset_dict.update({k: np.array(v) for k, v in batch.items() if k != "imgs"})
        save_hdf5(embedding_save_path, asset_dict=asset_dict, mode=mode)
    return embedding_save_path


def build_uni():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    ckpt = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin",
                           local_dir=WEIGHTS_DIR)
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0, dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model


encoder = build_uni().eval().to(device)
mean, std = get_constants("imagenet")
img_transforms = get_eval_transforms(mean, std, target_img_size=224)

for ds in DATASETS:
    out_dir = os.path.join(EMB, ds, ENC, "fp32")
    os.makedirs(out_dir, exist_ok=True)
    for tile_h5 in sorted(glob.glob(os.path.join(SRC, ds, "patches", "*.h5"))):
        sample_id = os.path.splitext(os.path.basename(tile_h5))[0]
        out_h5 = os.path.join(out_dir, f"{sample_id}.h5")
        if os.path.isfile(out_h5):
            print(f"skip {ds}/{sample_id} (exists)")
            continue
        tile_ds = H5TileDataset(tile_h5, chunk_size=CHUNK, img_transform=img_transforms)
        loader = torch.utils.data.DataLoader(tile_ds, batch_size=1, shuffle=False, num_workers=1)
        embed_tiles(loader, encoder, out_h5, device)
        print(f"done {ds}/{sample_id} -> {out_h5}")

print("ALL DONE")
