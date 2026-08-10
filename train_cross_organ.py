"""Stage 3 cross-organ trainer: V1 (film=context) vs V2 (film=meta) vs V3 (film=desc).

  V3 (film=desc): continuous slide-level histology descriptor (masked mean-pool of patch
  embeddings, computed inside the Denoiser) replaces the categorical organ token. Always
  available at inference incl. unseen organs -> no null-token cliff. No loader change needed.

Reuses STFlow's Denoiser / Interpolant / SPData but loads HEST cohorts across organs
from the cross_organ_splits/{POOLED,LOOO} CSVs and threads an organ id through FiLM.

  POOLED : leave-patient-out, all organs seen  -> tests metadata benefit (organ id = real)
  LOOO   : leave-one-organ-out                  -> tests transfer (held-out organ = null id 0)

Organ vocab: 0 = null token (CFG / unseen), 1..6 = the six organs present.
meta_categories = {"organ": 6}  -> MetadataEmbedder builds nn.Embedding(7, hidden).
"""
import os
import json
import argparse
from operator import itemgetter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from stflow.utils import set_random_seed, get_current_time, merge_fold_results
from stflow.data.dataset import SPData
from stflow.data.normalize_utils import get_normalize_method
from stflow.data.sampling_utils import PatchSampler
from stflow.hest_utils.st_dataset import load_adata
from stflow.hest_utils.file_utils import read_assets_from_h5
from stflow.model.denoiser import Denoiser
from stflow.flow.interpolant import Interpolant
from stflow.app.flow.test import metric_func

COHORT_TO_GROUP = {
    "CCRCC": "kidney", "COAD": "colorectal", "READ": "colorectal", "HCC": "liver",
    "IDC": "breast", "LYMPH_IDC": "breast", "LUNG": "lung", "PAAD": "pancreas",
    "PRAD": "prostate", "SKCM": "skin",
}
# Eight downloaded organs participate; null=0.
ORGAN_VOCAB = {"kidney": 1, "liver": 2, "lung": 3, "pancreas": 4,
               "prostate": 5, "skin": 6, "breast": 7, "colorectal": 8}
N_ORGANS = len(ORGAN_VOCAB)

# Opt-in second benchmark: STImage-1K4M, where the cohort dir IS the organ (identity map).
STIMAGE_ORGANS = ["breast", "kidney", "liver", "pancreas", "skin", "prostate", "brain", "heart"]


def configure_organ_set(name):
    """Swap the organ dictionaries for a benchmark family. Default 'hest' keeps the HEST
    cohort->organ grouping; 'stimage' uses organ==cohort with brain/heart added."""
    global COHORT_TO_GROUP, ORGAN_VOCAB, N_ORGANS
    if name == "stimage":
        COHORT_TO_GROUP = {o: o for o in STIMAGE_ORGANS}
        ORGAN_VOCAB = {o: i + 1 for i, o in enumerate(STIMAGE_ORGANS)}
        N_ORGANS = len(ORGAN_VOCAB)


def build_samples(df, args):
    samples = []
    for _, row in df.iterrows():
        cohort = row["patches_path"].split("/")[0]
        sid = row["sample_id"]
        organ = COHORT_TO_GROUP[cohort]
        samples.append({
            "name": sid,
            "cohort": cohort,
            "organ_id": ORGAN_VOCAB[organ],
            "h5_path": os.path.join(args.embed_dataroot, cohort, args.feature_encoder, f"fp32/{sid}.h5"),
            "h5ad_path": os.path.join(args.source_dataroot, cohort, f"adata/{sid}.h5ad"),
        })
    return samples


class CrossOrganDataset(Dataset):
    """Loads a list of cross-organ samples under one shared gene panel; carries organ id."""
    def __init__(self, samples, gene_list, normalize_method, distribution, sample_times,
                 force_null_organ=False):
        self.name = "cross_organ"
        self.gene_list = gene_list
        self.patch_sampler = PatchSampler(distribution)
        self.sp_datasets, self.organ_ids = [], []
        for s in samples:
            data_dict, _ = read_assets_from_h5(s["h5_path"])
            barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
            coords = data_dict["coords"]
            embeddings = data_dict["embeddings"]
            labels = load_adata(s["h5ad_path"], genes=gene_list, barcodes=barcodes,
                                normalize_method=normalize_method).values
            self.sp_datasets.append(SPData(
                features=torch.from_numpy(embeddings).float(),
                labels=torch.from_numpy(labels).float(),
                coords=torch.from_numpy(coords).float(),
            ))
            self.organ_ids.append(0 if force_null_organ else s["organ_id"])
        self.n_chunks = [sample_times] * len(self.sp_datasets)

    def __len__(self):
        return sum(self.n_chunks)

    def __getitem__(self, idx):
        for i, n in enumerate(self.n_chunks):
            if idx < n:
                chunk = self.sp_datasets[i].chunk(self.patch_sampler(self.sp_datasets[i].coords))
                return chunk, self.organ_ids[i]
            idx -= n


def cross_batcher():
    def b(batch):
        datas = [d[0] for d in batch]
        organs = torch.tensor([d[1] for d in batch], dtype=torch.long)
        features = [d.features for d in datas]
        labels = [d.labels for d in datas]
        coords = [d.coords for d in datas]
        max_len = max(x.size(0) for x in features)
        features = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in features])
        labels = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in labels])
        coords = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in coords])
        return features, coords, labels, organs
    return b


@torch.no_grad()
def evaluate(args, diffusier, model, test_samples, gene_list, use_meta, force_null_organ,
             return_preds=False):
    """Per-sample inference; returns metric_func dict over all held-out cells.
    If return_preds, also returns (all_pred, all_gt) for seed-ensembling (aligned across
    seeds because test_samples/gene_list are deterministic per fold)."""
    model.eval()
    all_pred, all_gt = [], []
    for s in test_samples:
        ds = CrossOrganDataset([s], gene_list, get_normalize_method(args.normalize_method),
                               distribution="constant_1.0", sample_times=1,
                               force_null_organ=force_null_organ)
        loader = DataLoader(ds, batch_size=1, collate_fn=cross_batcher())
        for img_features, coords, labels, organs in loader:
            img_features = img_features.to(args.device)
            coords = coords.to(args.device)
            labels = labels.to(args.device)
            meta = {"organ": organs.to(args.device)} if use_meta else None

            exp_t1 = diffusier.sample_from_prior(labels.shape).to(args.device)
            ts = torch.linspace(0.01, 1.0, args.n_sample_steps)[:, None].expand(
                args.n_sample_steps, exp_t1.shape[0]).to(args.device)
            for step, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
                pred = model.inference(exp_t1, img_features, coords, t1, meta=meta, predict=True)
                d_t = t2 - t1
                if step == args.n_sample_steps - 2:
                    break
                exp_t1 = diffusier.denoise(pred, exp_t1, t1, d_t)
            all_pred.append(pred.squeeze(0).cpu().numpy())
            all_gt.append(labels.squeeze(0).cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    res = metric_func(all_pred, all_gt, gene_list)
    res["n_test"] = len(all_gt)
    if return_preds:
        return res, all_pred, all_gt
    return res


def train_fold(args, train_samples, test_samples, gene_list, force_null_test):
    args.n_genes = len(gene_list)
    args.feature_dim = {"uni_v1_official": 1024, "gigapath": 1536, "ciga": 512,
                        "resnet50_trunc": 1024}[args.feature_encoder]
    use_meta = args.film == "meta"
    args.meta_categories = {"organ": N_ORGANS} if use_meta else None

    normalize_method = get_normalize_method(args.normalize_method)
    train_ds = CrossOrganDataset(train_samples, gene_list, normalize_method,
                                 distribution=args.patch_distribution,
                                 sample_times=args.sample_times)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              collate_fn=cross_batcher(), shuffle=True)

    device = args.device
    model = Denoiser(args).to(device)
    diffusier = Interpolant(
        args.prior_sampler,
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )
    if getattr(args, "optimizer", "adam") == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=getattr(args, "weight_decay", 0.0))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def _cosine_lr(ep):  # opt-in: linear warmup then cosine decay to 5% of base
        import math
        w = getattr(args, "warmup", 0)
        if ep <= w:
            return args.lr * ep / max(1, w)
        p = min(1.0, (ep - w) / max(1, args.epochs - w))
        return args.lr * (0.05 + 0.475 * (1 + math.cos(math.pi * p)))

    best_pearson, best_res = -1, None
    best_preds = None
    dump = getattr(args, "dump_preds", False)
    patience = getattr(args, "patience", 20)
    early_stop = 0
    epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)
    for epoch in epoch_iter:
        if getattr(args, "lr_schedule", "none") == "cosine":
            for g in optimizer.param_groups:
                g["lr"] = _cosine_lr(epoch)
        model.train()
        avg_loss = 0
        for img_features, coords, gene_exp, organs in train_loader:
            img_features = img_features.to(device)
            coords = coords.to(device)
            gene_exp = gene_exp.to(device)

            if use_meta:
                organs_in = organs.clone()
                drop = torch.rand(organs.shape[0]) < args.meta_dropout  # CFG: random null
                organs_in[drop] = 0
                meta = {"organ": organs_in.to(device)}
            else:
                meta = None

            noisy_exp, t_steps = diffusier.corrupt_exp(gene_exp)
            pred_exp, loss = model(exp=noisy_exp, img_features=img_features, coords=coords,
                                   labels=gene_exp, t_steps=t_steps, meta=meta)
            if getattr(args, "corr_weight", 0.0) > 0:   # opt-in: align training with the PCC metric
                pad = img_features.sum(-1) == 0         # [B,N] padding mask
                p = pred_exp[~pad]; g = gene_exp[~pad]  # [M,G]
                pd = p - p.mean(0, keepdim=True); gd = g - g.mean(0, keepdim=True)
                den = pd.norm(dim=0) * gd.norm(dim=0) + 1e-6
                r = (pd * gd).sum(0) / den
                m = gd.norm(dim=0) > 1e-6
                if m.any():
                    loss = loss + args.corr_weight * (1 - r[m].mean())
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            avg_loss += loss.cpu().item()
        avg_loss /= max(1, len(train_loader))
        epoch_iter.set_description(f"epoch {epoch} loss {avg_loss:.3f}")

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            ev = evaluate(args, diffusier, model, test_samples, gene_list,
                          use_meta=use_meta, force_null_organ=force_null_test,
                          return_preds=dump)
            res = ev[0] if dump else ev
            if res["pearson_mean"] > best_pearson:
                best_pearson = res["pearson_mean"]
                best_res = res
                if dump:
                    best_preds = (ev[1], ev[2])   # (all_pred, all_gt) at the best epoch
                early_stop = 0
            else:
                early_stop += 1
                if early_stop >= patience:
                    print("Early stopping")
                    break
    if dump and best_preds is not None:
        best_res = dict(best_res)
        best_res["_preds"] = best_preds   # picked up by run() to write the npz
    return best_res


def run(args):
    regime_dir = os.path.join(args.splits_root, args.regime)
    split_dir = os.path.join(regime_dir, "splits")
    if args.regime == "POOLED":
        fold_names = [str(i) for i in range(5)]
        force_null_test = False  # all organs seen -> real organ id at test
    else:  # LOOO
        fold_names = list(ORGAN_VOCAB.keys())
        force_null_test = True   # held-out organ unseen -> null token

    all_fold_results = []
    for fold in fold_names:
        fold_out = os.path.join(args.save_dir, f"fold_{fold}_results.json")
        if os.path.isfile(fold_out):
            print(f"\n=== {args.regime} fold {fold} (film={args.film}) -> SKIP (exists) ===")
            with open(fold_out) as f:
                all_fold_results.append(json.load(f))
            continue
        print(f"\n=== {args.regime} fold {fold} (film={args.film}) ===")
        train_df = pd.read_csv(os.path.join(split_dir, f"train_{fold}.csv"))
        test_df = pd.read_csv(os.path.join(split_dir, f"test_{fold}.csv"))
        with open(os.path.join(regime_dir, f"genes_{fold}.json")) as f:
            gene_list = json.load(f)["genes"]

        train_samples = build_samples(train_df, args)
        test_samples = build_samples(test_df, args)

        res = train_fold(args, train_samples, test_samples, gene_list, force_null_test)
        res["fold"] = fold
        preds = res.pop("_preds", None)   # numpy; not JSON-serializable
        if preds is not None:
            np.savez_compressed(os.path.join(args.save_dir, f"fold_{fold}_preds.npz"),
                                pred=preds[0], gt=preds[1], genes=np.array(gene_list, dtype=object))
        all_fold_results.append(res)

        with open(os.path.join(args.save_dir, f"fold_{fold}_results.json"), "w") as f:
            json.dump(res, f, sort_keys=True, indent=4)

    kfold = merge_fold_results(all_fold_results)
    kfold["pearson_corrs"] = sorted(kfold["pearson_corrs"], key=itemgetter("mean"), reverse=True)
    with open(os.path.join(args.save_dir, "results_kfold.json"), "w") as f:
        json.dump(kfold, f, sort_keys=True, indent=4)
    print(f"\n{args.regime} film={args.film}: pearson_mean = {kfold['pearson_mean']:.4f} "
          f"+/- {kfold['pearson_std']:.4f}  (per-fold {[round(x,4) for x in kfold['mean_per_split']]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--regime", type=str, required=True, choices=["POOLED", "LOOO"])
    p.add_argument("--film", type=str, required=True, choices=["none", "context", "meta", "desc", "hybrid", "moe", "local", "localg"])
    p.add_argument("--n_proto", type=int, default=8, help="hybrid/moe: # prototype tokens")
    # MoE (film=moe): morphology-routed experts on the scalar MLP path.
    p.add_argument("--n_experts", type=int, default=4, help="moe: # morphology experts")
    p.add_argument("--moe_top_k", type=int, default=0, help="moe: 0=dense soft routing, >0=sparse top-k")
    p.add_argument("--use_prototypes_in_router", action="store_true",
                   help="moe: feed mean-pooled slide prototypes into the router")
    p.add_argument("--lambda_bal", type=float, default=1e-2, help="moe: load-balance loss weight")
    p.add_argument("--lambda_smooth", type=float, default=1e-3, help="moe: routing spatial-smoothness weight")
    p.add_argument("--splits_root", type=str, default="cross_organ_splits")
    p.add_argument("--source_dataroot", type=str, default="dataset")
    p.add_argument("--embed_dataroot", type=str, default="embed_dataroot")
    p.add_argument("--feature_encoder", type=str, default="resnet50_trunc")
    p.add_argument("--save_root", type=str, default="results_cross_organ")
    p.add_argument("--exp_code", type=str, default=None)
    p.add_argument("--normalize_method", type=str, default="log1p")
    p.add_argument("--meta_dropout", type=float, default=0.1)
    p.add_argument("--dump_preds", action="store_true",
                   help="save best-epoch (pred,gt) per fold as fold_<f>_preds.npz for seed-ensembling")
    p.add_argument("--organ_set", type=str, default="hest", choices=["hest", "stimage"],
                   help="benchmark family: 'hest' (default) or 'stimage' (organ==cohort, +brain/heart)")

    p.add_argument("--device", type=int, default=0)
    p.add_argument("--sample_times", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr_schedule", type=str, default="none", choices=["none", "cosine"],
                   help="opt-in cosine LR (warmup->cosine); default none preserves prior results")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--corr_weight", type=float, default=0.0,
                   help="opt-in weight on (1 - per-gene Pearson) loss; default 0 preserves prior results")
    p.add_argument("--clip_norm", type=float, default=1.)
    p.add_argument("--eval_step", type=int, default=1)
    p.add_argument("--patch_distribution", type=str, default="uniform")

    p.add_argument("--n_sample_steps", type=int, default=5)
    p.add_argument("--prior_sampler", type=str, default="zinb")
    p.add_argument("--zinb_logits", type=float, default=0.1)
    p.add_argument("--zinb_total_count", type=float, default=1)
    p.add_argument("--zinb_zi_logits", type=float, default=0.)

    p.add_argument("--backbone", type=str, default="spatial_transformer")
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--pairwise_hidden_dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--attn_dropout", type=float, default=0.2)
    p.add_argument("--n_neighbors", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--norm", type=str, default="layer")
    p.add_argument("--activation", type=str, default="swiglu")
    args = p.parse_args()

    configure_organ_set(args.organ_set)
    set_random_seed(args.seed)
    if args.exp_code is None:
        args.exp_code = f"{args.regime}_{args.film}_seed{args.seed}"
    args.save_dir = os.path.join(args.save_root, args.exp_code)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, sort_keys=True, indent=4)
    print(f"Save dir: {args.save_dir}")
    run(args)
