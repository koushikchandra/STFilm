"""Download the raw files (coord/count/image) for the slides in a manifest.
Uses plain HTTPS with `Accept-Encoding: identity` to avoid a brotli decode bug in the
current hf transfer stack. Resumable: skips files already present with nonzero size."""
import argparse, os, sys, time, urllib.request
import pandas as pd

BASE = "https://huggingface.co/datasets/jiawennnn/STimage-1K4M/resolve/main"

def fetch(url, dest, retries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "identity",
                                                       "User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dest + ".part", dest)
            return "ok"
        except Exception as e:
            if a == retries - 1:
                return f"FAIL {type(e).__name__}: {str(e)[:100]}"
            time.sleep(2 * (a + 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--kinds", nargs="+", default=["coord", "count", "image"])
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for i, row in man.iterrows():
        for kind in args.kinds:
            rel = row[kind]                      # e.g. Visium/coord/<slide>_coord.csv
            dest = os.path.join(args.raw_dir, rel)
            status = fetch(f"{BASE}/{rel}", dest)
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                print(f"  [{row.organ}/{row.slide}] {kind}: {status}", flush=True)
        if (i + 1) % 5 == 0 or i == len(man) - 1:
            print(f"[{i+1}/{len(man)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE ok={n_ok} skip={n_skip} fail={n_fail}")
    sys.exit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
