# POOLED run — data bundle (`pooled5-data`)

**For the POOLED experiment on this branch, use the `pooled5-data` Release** (not `looo5-data`).
The POOLED cohorts are `CCRCC  IDC  LUNG  PRAD  SKCM` (breast + skin instead of COAD/PAAD), so
they need their own bundle. Same layout/roots as below.

```bash
gh release download pooled5-data --repo koushikchandra/STFilm --dir ~/pooled5_data
cd ~/pooled5_data
cat pooled5_data.tar.gz.part-* > pooled5_data.tar.gz
tar xzf pooled5_data.tar.gz          # -> ~/pooled5_data/dataset/ and ~/pooled5_data/embed_dataroot/
export DATA_ROOT=~/pooled5_data/dataset
export EMBED_ROOT=~/pooled5_data/embed_dataroot
```
Verify: `ls $DATA_ROOT/IDC/adata/*.h5ad` and `ls $EMBED_ROOT/SKCM/uni_conch/fp32/*.h5` should both
list files. Then run the POOLED commands in [README.md](README.md#pooled-run-5-fold-pooled-cross-validation).

Without `gh`: grab the parts from
`https://github.com/koushikchandra/STFilm/releases/tag/pooled5-data`, then the same
`cat ... > pooled5_data.tar.gz && tar xzf pooled5_data.tar.gz`.

---

# Downloading the data for the 5-organ LOOO test case

The code, splits, and gene panels are in this repo. The **HEST data itself is not** (it is ~3.4 GB
and individual files exceed GitHub's 100 MB tree limit). You need two things per cohort
(`COAD, CCRCC, LUNG, PAAD, PRAD`):

| What | Path it must land at |
|------|----------------------|
| Expression (`.h5ad`) | `<DATA_ROOT>/<COHORT>/adata/<sample>.h5ad` |
| Gene panel (per cohort) | `<DATA_ROOT>/<COHORT>/var_50genes.json` |
| Patch embeddings (UNI+CONCH, 1536-d) | `<EMBED_ROOT>/<COHORT>/uni_conch/fp32/<sample>.h5` |

You point the launchers at these two roots with `DATA_ROOT` and `EMBED_ROOT` (see below).

---

## Option 1 — download the prepared bundle (recommended)

The data is attached to the **`looo5-data` GitHub Release** as one or more split parts (`looo5_data.tar.gz.part-*`).

**With the GitHub CLI (`gh`):**
```bash
gh release download looo5-data --repo koushikchandra/STFilm --dir ~/looo5_data
cd ~/looo5_data
cat looo5_data.tar.gz.part-* > looo5_data.tar.gz
tar xzf looo5_data.tar.gz          # -> ~/looo5_data/dataset/ and ~/looo5_data/embed_dataroot/
```

**Without `gh` (plain download):** grab the parts from
`https://github.com/koushikchandra/STFilm/releases/tag/looo5-data`, then run the same
`cat ... > looo5_data.tar.gz && tar xzf looo5_data.tar.gz`.

Then run the test case:
```bash
cd STFilm/MorphoST/LOOO
export DATA_ROOT=~/looo5_data/dataset
export EMBED_ROOT=~/looo5_data/embed_dataroot
sbatch looo5_mist.sbatch                 # MIST
sbatch baselines/looo5_baselines.sbatch  # ST-Net, Hist2ST, BLEEP, STEM
# no SLURM? run the `python ...` block inside each .sbatch directly.
```

## Option 2 — rebuild from HEST-1k (if you cannot get the bundle)

Heavier: needs gated model weights and a GPU.
1. Download raw HEST-1k for the 5 cohorts from Hugging Face `MahmoodLab/hest` (public) →
   gives `<COHORT>/adata/*.h5ad` and the patch files.
2. Accept the licenses for **UNI** (`MahmoodLab/UNI`) and **CONCH** (`MahmoodLab/CONCH`) on
   Hugging Face and `huggingface-cli login`.
3. Extract UNI+CONCH features (encoder `uni_conch`) into
   `<EMBED_ROOT>/<COHORT>/uni_conch/fp32/*.h5` using the feature-extraction pipeline
   (`extract_features_uni.py` / `benchmark.py`, encoder `uni_conch`, on a GPU).

## Verify the layout before running
```bash
ls $DATA_ROOT/CCRCC/adata/*.h5ad | head
ls $EMBED_ROOT/CCRCC/uni_conch/fp32/*.h5 | head
```
Both should list files. If `EMBED_ROOT` is empty you have the expression but not the
embeddings — get them via Option 1, or extract them via Option 2.
