import os
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
from scattermap import scattermap
import matplotlib.pyplot as plt
from matplotlib import lines
import seaborn as sns
import sszpalette
colorsmaps=sszpalette.register()

def plot_bubblemap(dtt,
                    brain_region,
                    celltype,
                    savepath, 
                    mix_brain_region=False):
        
    color_cond= {"HCADD":"#262730","ADD":"#F57764", "ResilientADD":"#77BA99", "RADOther":"#77BA99"}
    color_cond = {"Neur":"#75485E","EXC":"#C23E7E", "INH":"#75485E", "Glials":"#51A3A3","AST-OPCs-OLD":"#51A3A3", "OPCs-Oligo":"#51A3A3","OLD":"#51A3A3","OPCs":"#C0F0F0", "MIC":"#CB904D", "AST":"#C3E991",
                   "bulk":"#cccccc", "CTRH":"#B6F2EC", "SPOR":"#F7D7C8"}
    colors_brain_region= {"CAUD":[18/255.,50/255.,59/255., 1],
            "SMTG":[118/255.,14/255.,63/255.,1], 
            "PARL":[92/255.,64/255.,77/255.,1],
            "HIPP":[255/255., 192/255.,0, 1] }

    cmap_reversed = sns.color_palette('tab20c_r',
                            dtt["DNA_binding_domain"].nunique())
    palette = {pp:cc for pp,cc in zip(dtt["DNA_binding_domain"].unique(),
                                        cmap_reversed)}
    norm=plt.Normalize(1,150)
    colorlist=["darkorange", "gold", "lawngreen", "lightseagreen"]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    #dtt = dtt[dtt.comparison==comp]
    dtt = dtt.sort_values("Motif_Name_", ascending=False)
    dtt = dtt[~dtt["Motif Name"].str.contains("CHR")]
    if mix_brain_region:
        dtt["cond_region_celltype"] =  dtt["ct"]\

    else:
        dtt["cond_region_celltype"] = dtt["CT|BR"]
    key = "# of Target/# Peak DE"
    #key="Percentage of Target Sequences with Motif"
    dtts = dtt.pivot("Motif_Name_", "cond_region_celltype", key)
    dtts.sort_index(ascending=False, inplace=True)
    #dtts = dtt.pivot("Motif_Name_", "cond_region_celltype", "-log(q-value)")

    
    dtt[key] = np.ceil(dtt[key]).astype(int)
    kk = "-log(q-value)"
    dtt[kk] = np.round(dtt[kk].values, 2)
    dtts_size = dtt.pivot("Motif_Name_", "cond_region_celltype",
                    kk).fillna(0)
                    #"Percentage of Target Sequences with Motif").fillna(0)
    dtts_size.sort_index(ascending=False, inplace=True)

    all_comb = ["%s|%s"%(ct,ba) for ct in celltype for ba in brain_region]
    for it in all_comb:
        if not it in dtts.columns.tolist():
            dtts[it] = np.nan
            dtts_size[it] = 0
    dtts = dtts[all_comb]
    dtts_size = dtts_size[all_comb]
    max_or = dtts_size.max().max()
    print(max_or)
    min_or = dtts_size[dtts_size!=0].min().min()
    print(min_or)
    off = 1
    dtts_size = dtts_size.values*off
    seq = ['#1a0207', '#450612', '#720b1f', '#9d142e', '#c22340', '#df3b57', '#f26178', '#fc99a8', '#ffe0e4aa', "#fffafa"]
    seq.reverse()
    cmap = matplotlib.colors.ListedColormap(colors=seq, name="seq9rot_r")
    sns.set(font_scale=2)
    with sns.axes_style("whitegrid"):
        g = sns.clustermap(dtts_size.T, cmap=cmap,
                    vmin=0,
                    linewidth=2,
                        yticklabels=dtts.columns.tolist(),
                        xticklabels=dtts.index.tolist(),
                        #ax=ax,
                       cbar_pos=(0, 0, .03, .4), 
                        figsize=(dtt["Motif Name"].nunique()/2.2, 5), 
                        dendrogram_ratio=0.02,
                        cbar_kws={"shrink":0.5})
    ax = g.ax_heatmap
    fontscale=24
    for label in (ax.get_yticklabels()):
        print(label.get_text())
        reg = label.get_text().split("|")[1]
        ct_t = label.get_text().split("|")[0]
        label.set(color=color_cond[ct_t],
                label=ct_t, fontsize=fontscale, rotation=90 )
    for label in (ax.get_xticklabels()):

        cc = label.get_text().split(":")[0]
        txt = label.get_text().split(":")[-1]
        label.set(color="black", label=label, fontsize=fontscale)
    labels = [item.get_text().split("|")[0] for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels, rotation=90)
    labels = [item.get_text().split("(")[0].split(":")[1] for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    #ax.set_xlabel("")
    #ax.set_ylabel("")
    ll = []

    blue_line = lines.Line2D([], [], color="grey", 
                marker="o",
                linestyle='None',
                markersize=np.sqrt(min_or*10),
                label=min_or)
    ll.append(blue_line)
    blue_line = lines.Line2D([], [], color="grey", marker="o",
            linestyle='None', markersize=np.sqrt(max_or*off), label=max_or)
    ll.append(blue_line)   
    ll = []
    for it in brain_region:#dtt["brain_region"].unique():
        blue_line = lines.Line2D([], [], color=colors_brain_region[it], 
        marker="o",
        linestyle='None',
        markersize=8, label=it)
        ll.append(blue_line)
    print(savepath)
    plt.savefig(savepath + "TF_transcriptor_factor" +  ".svg", bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def main():
    path_h = "../cellformer/bed_diff/peak_homer/"
    all_ts = []
    BR = ["HIPP", "SMTG"]
    CT = ["AST", "EXC", "INH",  "MIC", "OLD", "OPCs"]
    for br in BR:
        for ct in CT:
            npp = path_h + "peakid%s%sgenes_de_with_logFC0.50.01.bed"%(ct,br)
            if os.path.exists(npp +"/knownResults.txt"):
                ts = pd.read_csv(npp + "/knownResults.txt", sep="\t", index_col=None)
                ts["ct"] = ct
                ts["BR"] = br
                ts["CT|BR"] = ct + "|" + br
                nb_pe_bg = eval(ts.iloc[:,7].name.split("of ")[-1][:-1])
                nb_pe_de = eval(ts.iloc[:,5].name.split("of ")[-1][:-1])
                ts["# of Target/# Peak DE"] = ts.iloc[:,5].values/nb_pe_de
                ts["# of Background/# Peak BG"] = ts.iloc[:,7].values/nb_pe_de
                ts["Ratio Target/Background"] = np.round(
                                    ts["# of Target/# Peak DE"].values/
                                        ts["# of Background/# Peak BG"],2)
                nb_pe_de = eval(ts.iloc[:,7].name.split("of ")[-1][:-1])
                all_ts.append(ts)
    all_ts = pd.concat(all_ts)
    all_ts = all_ts.loc[:, all_ts.isna().sum() ==0]
    all_ts["TF_name_"] = all_ts["Motif Name"].str.split("(", expand=True)[0]
    all_ts["TF_name"] = all_ts["Motif Name"].str.split("(", expand=True)[0]
    all_ts["DNA_binding_domain"] = all_ts[
                                    "Motif Name"].str.split(
                                            "(", expand=True)[1].str.split(")",
                                                            expand=True)[0]
    all_ts["TF_name"] = all_ts["DNA_binding_domain"].values + ":" + all_ts["TF_name"].values
    all_ts["Motif_Name_"] = all_ts["DNA_binding_domain"].values + ":" + all_ts["Motif Name"].values
    all_ts["q-value (Benjamini)"] += 1e-7
    all_ts["-log(q-value)"] = -np.log(all_ts["q-value (Benjamini)"].values)
    all_ts_sig = all_ts[all_ts["-log(q-value)"]>=-np.log(0.05)]
    all_ts_sig["Percentage of Target Sequences with Motif"] = np.round(all_ts_sig["% of Target Sequences with Motif"].str.replace("%", "").astype(float),1)
    all_ts_sig["Percentage of Background Sequences with Motif"] = np.round(all_ts_sig["% of Background Sequences with Motif"].str.replace("%", "").astype(float),1)
    all_ts_sig.to_csv(path_h + "TF_enr.csv")
    for ct in CT:
        all_tmp = all_ts_sig[all_ts_sig.ct == ct]
        all_tmp.to_csv(path_h + "TF_enr_%s.csv"%ct)
    all_ts_sig["Percentage of Target Sequences with Motif"]
    alpha = 0.8
    colors_ct = {"EXC":"#C23E7E", "INH":"#75485E",  
                                                            "OLD":"#51A3A3","OPCs":"#C0F0F0", "MIC":"#CB904D", "AST":"#C3E991",
                                                                           }
    colors_brain_region= {"CAUD":[18/255.,50/255.,59/255., 1],
                        "SMTG":[118/255.,14/255.,63/255.,1], 
                        "PARL":[92/255.,64/255.,77/255.,1],
                        "HIPP":[255/255., 192/255.,0, 1] }

    print(all_ts_sig["Motif_Name_"].nunique())
    list_tf =  all_ts_sig["Motif_Name_"].str.upper().unique()
    all_ts_sig = all_ts_sig[all_ts_sig["-log(q-value)"]>=-np.log(0.01)]
    all_ts_sig.sort_values("DNA_binding_domain", inplace=True)
    plot_bubblemap(all_ts_sig,
                    all_ts_sig["BR"].unique(),
                    CT,
                    #all_ts_sig["ct"].unique(),
                        path_h + "p_001", 
                        mix_brain_region=False)

if __name__ == "__main__":
    main()
