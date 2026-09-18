"""COAST feature-matched FiLM tests for recent baselines.

Phase 1 supports Gene-DML and FEAST using the same cached UNI features, splits, 50-gene panels,
metrics, seeds, and fold JSON format as film_baselines.py. Upstream trees remain unmodified.
These are explicitly feature-matched adaptations: Gene-DML retains spot/neighborhood/global streams;
FEAST imports the official attention model directly.
"""
import argparse, glob, importlib.util, json, os, sys
from operator import itemgetter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "STFlow"))
import baseline_spatial as B
from stflow.utils import set_random_seed
from MorphoST.evaluation import train_val_split   # nested-validation inner split (+ single-slide fallback)
from stflow.data.normalize_utils import get_normalize_method


class FiLMCond(nn.Module):
    def __init__(self, fdim, dim, mode):
        super().__init__(); self.mode = mode
        if mode != "none":
            self.proj = nn.Linear(fdim, dim)
            self.affine = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.affine.weight); nn.init.zeros_(self.affine.bias)

    def forward(self, x, feat, adj):
        if self.mode == "none": return x
        if self.mode == "desc": d = feat.mean(0, keepdim=True).expand_as(feat)
        else:
            w = adj / adj.sum(1, keepdim=True).clamp(min=1)
            d = w @ feat
        gamma, beta = self.affine(F.gelu(self.proj(d))).chunk(2, -1)
        return x * (1 + gamma) + beta


class GeneDMLAdapt(nn.Module):
    """Feature-level Gene-DML adaptation: target, neighbor, and global pathways + learned fusion."""
    def __init__(self, fdim, dim, genes, heads, depth, dropout, film):
        super().__init__()
        self.spot = nn.Linear(fdim, dim); self.neigh = nn.Linear(fdim, dim); self.glob = nn.Linear(fdim, dim)
        self.film = FiLMCond(fdim, dim, film)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout, batch_first=True, norm_first=True)
        self.fuse = nn.TransformerEncoder(layer, depth)
        self.type_emb = nn.Parameter(torch.zeros(3, dim)); nn.init.normal_(self.type_emb, std=.02)
        self.stream_heads = nn.ModuleList([nn.Linear(dim, genes) for _ in range(3)])
        self.mix = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, genes))

    def forward(self, feat, gxy, adj):
        w = adj / adj.sum(1, keepdim=True).clamp(min=1)
        nfeat = w @ feat; gfeat = feat.mean(0, keepdim=True).expand_as(feat)
        s = self.film(self.spot(feat), feat, adj)
        toks = torch.stack([s, self.neigh(nfeat), self.glob(gfeat)], 1) + self.type_emb[None]
        toks = self.fuse(toks)
        # Preserve Gene-DML's pathway supervision while returning fused prediction.
        self.aux_predictions = [h(toks[:, i]) for i, h in enumerate(self.stream_heads)]
        return self.mix(toks.flatten(1))


def load_official_feast():
    path = os.path.join(ROOT, "baselines", "FEAST", "model", "feast.py")
    spec = importlib.util.spec_from_file_location("coast_official_feast", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.FEAST


class FEASTAdapt(nn.Module):
    """Official FEAST core with a zero-init morphology conditioner on its input stream."""
    def __init__(self, fdim, dim, genes, heads, depth, dropout, film, k):
        super().__init__()
        OfficialFEAST = load_official_feast()
        self.proj = nn.Linear(fdim, dim)
        self.film = FiLMCond(fdim, dim, film)
        self.core = OfficialFEAST(input_dim=dim, num_blocks=depth, num_heads=heads,
                                  dropout=dropout, num_genes=genes, k_neighbors=k,
                                  tau_neg=.6, beta=1.5)

    def forward(self, feat, gxy, adj):
        x = self.film(self.proj(feat), feat, adj)
        # Official FEAST parses coordinates from strings of the form "x_coord x y_coord".
        barcodes = [f"{int(a)}x{int(b)}" for a, b in gxy.detach().cpu().tolist()]
        is_pseudo = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.core(x, barcodes=barcodes, is_pseudo=is_pseudo)


def build(args, genes):
    kw = (args.feature_dim, args.dim, genes, args.heads, args.depth, args.dropout, args.film)
    if args.model == "genedml": return GeneDMLAdapt(*kw)
    if args.model == "feast": return FEASTAdapt(*kw, args.k)
    raise ValueError(args.model)


def evaluate(model, slides, genes, device):
    return B.evaluate(model, slides, genes, device)


def train_fold(args, train_slides, val_slides, test_slides, genes, device):
    model = build(args, len(genes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_state, early = -1e9, None, 0
    order = list(range(len(train_slides)))
    for epoch in range(args.epochs):
        model.train(); np.random.shuffle(order)
        for i in order:
            s=train_slides[i]; feat=s['feat']; gxy=s['gxy']; adj=s['adj']; lab=s['labels']
            if len(feat) > args.max_spots:
                sel=torch.randperm(len(feat))[:args.max_spots]
                feat,gxy,lab,adj=feat[sel],gxy[sel],lab[sel],adj[sel][:,sel]
            feat,gxy,adj,lab=feat.to(device),gxy.to(device),adj.to(device),lab.to(device)
            pred=model(feat,gxy,adj); loss=F.mse_loss(pred,lab)
            if args.model == 'genedml':
                loss = loss + args.aux_weight * sum(F.mse_loss(p,lab) for p in model.aux_predictions) / 3
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        score=evaluate(model,val_slides,genes,device)['pearson_mean']   # select on VAL
        if best_state is None or (score==score and score>best):         # always keep one; NaN-safe
            best,best_state,early=score,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},0
        else:
            early += 1
            if early >= args.patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model,test_slides,genes,device)                     # score TEST once


def _cap(s, cap):
    """Deterministically subsample a slide to <=cap spots (FEAST's full O(N^2) attention OOMs on large
    pooled slides at eval). Seeded by slide_id so none/desc/local score the SAME spots -> fair FiLM
    comparison; independent of the global RNG."""
    n = len(s['feat'])
    if n <= cap:
        return s
    g = torch.Generator().manual_seed(abs(hash(str(s['slide_id']))) % (2**31))
    sel = torch.randperm(n, generator=g)[:cap]
    o = dict(s)
    for key in ('feat', 'gxy', 'labels', 'coords'):
        if key in o:
            o[key] = s[key][sel]
    o['adj'] = s['adj'][sel][:, sel]
    return o


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for this benchmark run, but no usable GPU is available. "
            "Refusing to fall back silently to CPU."
        )
    device=f"cuda:{args.device}"
    set_random_seed(args.seed); args.feature_dim=1024
    regime_dir=os.path.join(args.splits_root,args.regime); split_dir=os.path.join(regime_dir,'splits')
    folds=[os.path.basename(x)[6:-4] for x in glob.glob(os.path.join(split_dir,'train_*.csv'))]
    folds.sort(key=lambda x:(int(x) if x.isdigit() else 1<<30,x))
    tag=f"{args.regime}_{args.model}-{args.film}_seed{args.seed}"
    save=os.path.join(args.save_root,tag); os.makedirs(save,exist_ok=True)
    nm=get_normalize_method(args.normalize_method); all_res=[]
    for fold in folds:
        out=os.path.join(save,f"fold_{fold}_results.json")
        if os.path.isfile(out): all_res.append(json.load(open(out))); continue
        outer_tr=pd.read_csv(os.path.join(split_dir,f'train_{fold}.csv'))
        te=pd.read_csv(os.path.join(split_dir,f'test_{fold}.csv'))
        genes=json.load(open(os.path.join(regime_dir,f'genes_{fold}.json')))['genes']
        cap=args.max_spots if args.model=='feast' else None   # FEAST O(N^2) attn -> cap spots at load
        seed_fold=args.seed+(int(fold) if str(fold).isdigit() else abs(hash(str(fold)))%1000)
        tr,va=train_val_split(outer_tr,seed_fold,args.val_fraction)     # nested inner validation split
        train=B.load_slides(tr,args,genes,nm,args.n_pos,args.k,cap=cap)
        val=B.load_slides(va,args,genes,nm,args.n_pos,args.k,cap=cap)
        test=B.load_slides(te,args,genes,nm,args.n_pos,args.k,cap=cap)
        res=train_fold(args,train,val,test,genes,device); res['fold']=fold; res['n_val_slides']=len(va)
        json.dump(res,open(out,'w'),sort_keys=True,indent=2); all_res.append(res)
        print(tag,fold,res['pearson_mean'],flush=True)
    pm=[r['pearson_mean'] for r in all_res]
    k={'pearson_mean':float(np.mean(pm)),'pearson_std':float(np.std(pm)),'mean_per_split':pm,
       'spearman_mean':float(np.nanmean([r.get('spearman_mean',float('nan')) for r in all_res])),
       'mse':float(np.nanmean([r.get('mse',float('nan')) for r in all_res])),'n_folds':len(all_res)}
    json.dump(k,open(os.path.join(save,'results_kfold.json'),'w'),sort_keys=True,indent=2)
    print(tag,'pearson_mean',k['pearson_mean'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['genedml','feast'],required=True)
    p.add_argument('--film',choices=['none','desc','local'],required=True); p.add_argument('--regime',choices=['LOOO','POOLED'],required=True)
    p.add_argument('--seed',type=int,default=1); p.add_argument('--splits_root',default='../cross_organ_splits8')
    p.add_argument('--source_dataroot',default='../dataset'); p.add_argument('--embed_dataroot',default='../embed_dataroot')
    p.add_argument('--feature_encoder',default='uni_v1_official'); p.add_argument('--save_root',default='results_recent_film')
    p.add_argument('--normalize_method',default='log1p'); p.add_argument('--device',type=int,default=0)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=20)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight_decay',type=float,default=1e-5)
    p.add_argument('--dim',type=int,default=256); p.add_argument('--heads',type=int,default=8); p.add_argument('--depth',type=int,default=2)
    p.add_argument('--dropout',type=float,default=.1); p.add_argument('--n_pos',type=int,default=128); p.add_argument('--k',type=int,default=8)
    p.add_argument('--max_spots',type=int,default=4000); p.add_argument('--aux_weight',type=float,default=.25)
    p.add_argument('--val_fraction',type=float,default=0.15)
    run(p.parse_args())
