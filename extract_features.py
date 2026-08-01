"""Minimal resnet50_trunc feature extractor for the FiLM ablation.

Reuses STFlow's embed_tiles + H5TileDataset, but skips benchmark.py's __main__
(which re-downloads the full 42GB dataset and gated UNI/GigaPath weights) and the
random-forest linear probe. Writes:
    embed_dataroot/<dataset>/resnet50_trunc/fp32/<sample_id>.h5
with keys {embeddings, coords, barcodes}, exactly what data/dataset.py reads.
"""
import os
import glob
import timm
import numpy as np
import torch
import torch.nn as nn

from stflow.hest_utils.st_dataset import H5TileDataset
from stflow.hest_utils.utils import get_constants, get_eval_transforms
from stflow.hest_utils.file_utils import save_hdf5


def embed_tiles(dataloader, model, embedding_save_path, device, precision=torch.float32):
    """Inlined from benchmark.embed_tiles (avoids importing the linear-probe trainer)."""
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

SRC = "dataset"
EMB = "embed_dataroot"
ENC = "resnet50_trunc"
DATASETS = ["LUNG", "HCC", "CCRCC", "PAAD", "PRAD", "SKCM"]
CHUNK = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Resnet50Trunc(nn.Module):
    """Mirror of TimmCNNEncoder for resnet50_trunc (avoids importing encoder_wrappers,
    which pulls in `transformers`)."""
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "resnet50.tv_in1k", features_only=True, out_indices=(3,),
            pretrained=True, num_classes=0,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.model(x)
        out = out[0] if isinstance(out, list) else out
        return self.pool(out).squeeze(-1).squeeze(-1)


encoder = Resnet50Trunc().eval().to(device)
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
        embed_tiles(loader, encoder, out_h5, device, precision=torch.float32)
        print(f"done {ds}/{sample_id} -> {out_h5}")

print("ALL DONE")
