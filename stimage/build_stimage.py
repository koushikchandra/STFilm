"""Build STImage-1K4M into the HEST-style layout STFlow/train_cross_organ.py consumes.

For each slide in the manifest:
  * crop a patch centered on each spot (box = [x-r, y-r, x+r, y+r] from coord.csv),
  * run UNI (uni_v1_official, local weights) to get 1024-d embeddings,
  * write embed h5  -> <embed_root>/<organ>/uni_v1_official/fp32/<slide>.h5
        keys: embeddings (N,1024), barcodes (N,), coords (N,2) pixel centers,
  * write raw-count h5ad -> <src_root>/<organ>/adata/<slide>.h5ad
        X = raw counts, obs_names = barcodes, var_names = gene symbols
        (load_adata applies log1p at train time, matching HEST).

Resumable: skips a slide whose h5 + h5ad already exist. Feature extraction needs a GPU;
`--limit_spots` gives a fast CPU smoke test of the crop/h5ad/coords path.
"""
import argparse, os, sys, glob
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import scanpy as sc
import anndata as ad
import torch

Image.MAX_IMAGE_PIXELS = None  # full-slide PNGs exceed PIL's default bomb guard

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "STFlow"))
from stflow.hest_utils.encoder import load_encoder
from stflow.hest_utils.file_utils import save_hdf5


def load_uni(weights_root, device):
    enc, tfm, cfg = load_encoder("uni_v1_official", device, weights_root, None)
    return enc, tfm


@torch.no_grad()
def embed_slide(img, coord, encoder, tfm, device, batch=256, limit=None, context_scale=4.0):
    """coord: DataFrame indexed by barcode with columns xaxis,yaxis,r (pixels).

    STImage-1K4M ships downsized full slides (~2000 px), so a spot radius r is only ~8 px.
    We crop a context window of context_scale*r around each spot (upscaled to 224 by the UNI
    transform) so the encoder sees real morphology, not a ~17 px thumbnail."""
    W, H = img.size
    barcodes = list(coord.index)
    if limit:
        barcodes = barcodes[:limit]
    xs = coord["xaxis"].values; ys = coord["yaxis"].values
    rs = coord["r"].values
    embs, coords_out, kept = [], [], []
    buf = []
    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        out = encoder(x)
        embs.append(out.float().cpu().numpy())
        buf.clear()
    for i, bc in enumerate(barcodes):
        x, y, r = float(xs[i]), float(ys[i]), float(rs[i])
        r = max(r, 8.0) * context_scale
        box = (int(round(x - r)), int(round(y - r)), int(round(x + r)), int(round(y + r)))
        # clamp to image bounds
        box = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        # force a square patch: Resize(224) in the UNI transform only touches the shorter
        # side, so border-clamped non-square crops would produce ragged tensors.
        patch = img.crop(box).convert("RGB").resize((224, 224), Image.BILINEAR)
        buf.append(tfm(patch))
        coords_out.append([x, y]); kept.append(bc)
        if len(buf) >= batch:
            flush()
    flush()
    if not embs:
        return None, None, None
    E = np.concatenate(embs, 0)
    return E, np.array(kept, dtype=object), np.array(coords_out, dtype=np.float32)


def build_h5ad(count_path, barcodes, out_path):
    df = pd.read_csv(count_path, index_col=0)
    df = df.reindex(barcodes)              # align to the spots we actually embedded
    df = df[~df.index.duplicated(keep="first")]
    X = df.values.astype(np.float32)
    a = ad.AnnData(X=X,
                   obs=pd.DataFrame(index=df.index.astype(str)),
                   var=pd.DataFrame(index=df.columns.astype(str)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    a.write_h5ad(out_path)
    return a.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--embed_root", required=True)
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--weights_root", default="weights_root")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--context_scale", type=float, default=4.0,
                    help="crop window = context_scale * spot radius (context around each spot)")
    ap.add_argument("--limit_spots", type=int, default=0, help=">0: smoke test (few spots, CPU ok)")
    ap.add_argument("--only_organ", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"start: manifest={args.manifest} device={args.device}", flush=True)
    man = pd.read_csv(args.manifest)
    if args.only_organ:
        man = man[man.organ == args.only_organ]
    print(f"loading UNI from {args.weights_root} ...", flush=True)
    encoder, tfm = load_uni(args.weights_root, args.device)
    print(f"UNI loaded on {args.device}; {len(man)} slides", flush=True)

    done = fail = 0
    for _, row in man.iterrows():
        organ, slide = row["organ"], row["slide"]
        h5_out = os.path.join(args.embed_root, organ, "uni_v1_official", "fp32", f"{slide}.h5")
        ad_out = os.path.join(args.src_root, organ, "adata", f"{slide}.h5ad")
        if not args.limit_spots and os.path.exists(h5_out) and os.path.exists(ad_out):
            done += 1; continue
        try:
            img = Image.open(os.path.join(args.raw_dir, row["image"]))
            coord = pd.read_csv(os.path.join(args.raw_dir, row["coord"]), index_col=0)
            E, bcs, coords = embed_slide(img, coord, encoder, tfm, args.device,
                                         batch=args.batch, limit=args.limit_spots or None,
                                         context_scale=args.context_scale)
            if E is None:
                print(f"  [{organ}/{slide}] no valid patches", flush=True); fail += 1; continue
            shp = build_h5ad(os.path.join(args.raw_dir, row["count"]), list(bcs), ad_out)
            os.makedirs(os.path.dirname(h5_out), exist_ok=True)
            if os.path.exists(h5_out):
                os.remove(h5_out)
            save_hdf5(h5_out, {"embeddings": E, "barcodes": bcs, "coords": coords}, mode="w")
            print(f"  [{organ}/{slide}] emb={E.shape} h5ad={shp}", flush=True)
            done += 1
        except Exception as e:
            print(f"  [{organ}/{slide}] ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
            fail += 1
    print(f"DONE built={done} fail={fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
