import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np, json, os

BASE="/lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST/LOOO"
# 6-organ no-COAD LOOO data (Table 5), 4 panels: exclude CCRCC (kidney) and PRAD (prostate)
COHORTS=[("skin","SKCM · Skin"),("lung","LUNG · Lung"),
         ("breast","IDC · Breast"),("pancreas","PAAD · Pancreas")]
METHODS=[("MIST (ours)","C111"),("ST-Net","stnet"),("Hist2ST","hist2st"),("BLEEP","bleep"),("STEM","stem")]
SEEDS=[1,2,3]
MCOLORS={"MIST (ours)":"#E8820C","ST-Net":"#3b6fb0","Hist2ST":"#2ca02c","BLEEP":"#9467bd","STEM":"#17a2b8"}
LGRAY="#c9c9c9"

def path(mk,seed,c):
    if mk=="C111":
        return f"{BASE}/results_looo6nocoad_mist/LOOO_C111_seed{seed}/fold_{c}_results.json"
    return f"{BASE}/baselines/results_looo6nocoad_baselines/LOOO_{mk}_seed{seed}/fold_{c}_results.json"
def pergene(mk,c):
    v=[]
    for s in SEEDS:
        p=path(mk,s,c)
        if os.path.isfile(p): v+=list(json.load(open(p)).get("pearson_per_gene",[]))
    return np.array(v,float)

fig,axes=plt.subplots(1,4,figsize=(12,5.2),sharey=False)  # independent y-scale per cohort
colors=[MCOLORS[m] for m,_ in METHODS]
for ax,(cf,ctitle) in zip(axes,COHORTS):
    data=[pergene(mk,cf) for _,mk in METHODS]
    bp=ax.boxplot(data,widths=0.62,patch_artist=True,showfliers=False,
                  medianprops=dict(color=LGRAY,lw=1.2),
                  whiskerprops=dict(color=LGRAY),capprops=dict(color=LGRAY))
    for patch,c in zip(bp["boxes"],colors):
        patch.set_facecolor(c); patch.set_alpha(0.8); patch.set_edgecolor(LGRAY); patch.set_linewidth(.9)
    for i,d in enumerate(data,1):
        if len(d): ax.scatter(i,np.mean(d),marker="D",s=26,color="white",edgecolor=LGRAY,linewidths=1.1,zorder=4)
    ax.set_xticks([])
    ax.set_xlabel(ctitle,fontsize=11,fontweight="bold",labelpad=6)
    ax.axhline(0,color=LGRAY,lw=.7,ls="--")
    ax.grid(axis="y",ls=":",lw=.5,alpha=.6)
    for sp in ax.spines.values(): sp.set_color(LGRAY); sp.set_linewidth(.9)
    ax.tick_params(color=LGRAY)
    ax.set_ylabel("Per-gene PCC",fontsize=9)
    ax.margins(y=0.08)
handles=[mpatches.Patch(facecolor=MCOLORS[m],edgecolor=LGRAY,label=m) for m,_ in METHODS]
fig.legend(handles=handles,loc="lower center",ncol=len(METHODS),frameon=True,fontsize=10,
           bbox_to_anchor=(0.5,-0.04),edgecolor=LGRAY,columnspacing=1.4,handlelength=1.3)
fig.tight_layout(rect=[0,0.05,1,1])
out="/lustre/hdd/LAS/weile-lab/howlader/STFilm/MorphoST/paper_iclr/figures/box_looo6_4panel.png"
fig.savefig(out,dpi=600,bbox_inches="tight"); print("wrote",out)
fig.savefig(out.replace(".png",".pdf"),bbox_inches="tight"); print("wrote pdf")
