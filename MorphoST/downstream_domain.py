"""Downstream task: does predicted expression preserve spatial-domain structure?
For each held-out slide we cluster spots by their expression vector (KMeans, k domains) using
(a) ground-truth expression and (b) predicted expression, then measure agreement of the two spatial
partitions with Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI). High ARI/NMI means
the prediction recovers the same spatial domains as the real data -- a biology-facing check that
complements per-gene PCC. Needs only the saved test_predictions.npz (no external annotations)."""
import sys, glob, argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def slide_ari(pred, target, k, seed=0):
    sc = StandardScaler()
    lt = KMeans(k, n_init=10, random_state=seed).fit_predict(sc.fit_transform(target))
    lp = KMeans(k, n_init=10, random_state=seed).fit_predict(sc.fit_transform(pred))
    return adjusted_rand_score(lt, lp), normalized_mutual_info_score(lt, lp)


def eval_npz(path, k):
    d = np.load(path, allow_pickle=True)
    pred, target, sid = d["pred"], d["target"], d["slide_ids"]
    out = []
    for s in np.unique(sid):
        m = sid == s
        if m.sum() < k * 5:  # need enough spots to form k domains
            continue
        ari, nmi = slide_ari(pred[m], target[m], k)
        out.append((str(s), m.sum(), ari, nmi))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("globs", nargs="+", help="dirs or npz globs")
    p.add_argument("--k", type=int, default=6)
    args = p.parse_args()
    files = []
    for g in args.globs:
        files += glob.glob(g) if g.endswith(".npz") else glob.glob(f"{g}/**/test_predictions.npz", recursive=True)
    aris, nmis = [], []
    for f in sorted(set(files)):
        for s, n, ari, nmi in eval_npz(f, args.k):
            aris.append(ari); nmis.append(nmi)
    if aris:
        print(f"slides={len(aris)}  k={args.k}  ARI={np.mean(aris):.3f}±{np.std(aris):.3f}  "
              f"NMI={np.mean(nmis):.3f}±{np.std(nmis):.3f}")
    else:
        print("no eligible slides")


if __name__ == "__main__":
    main()
