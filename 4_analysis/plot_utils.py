import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
from matplotlib import lines
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
import statsmodels.api as sm
import itertools
import itertools
import scanpy as sc
import anndata as ad
import episcanpy.api as epi
import sys
sys.path.append("../")
from utils.scattermap import scattermap
from scipy.stats import ttest_ind, mannwhitneyu
import gseapy
import sszpalette

# register the ssz color palette
colorsmaps = sszpalette.register()
#from utils.supervenn.supervenn import *
sys.path.append("../utils/supervenn/")
from utils.supervenn.supervenn import supervenn



colors_ct = {"EXC":"#C23E7E", "INH":"#75485E",
             "OLD":"#51A3A3","OPCs":"#C0F0F0", 
             "MIC":"#CB904D", "AST":"#C3E991",
               "bulk":"#cccccc"}
color_cond = colors_ct
colors_brain_region= {"CAUD":[18/255.,50/255.,59/255., 1], "SMTG":[118/255.,14/255.,63/255.,1],
                          "PARL":[92/255.,64/255.,77/255.,1],"HIPP":[255/255., 192/255.,0, 1],
                         "SMTG+HIPP":[0.5,0.5,0.5, 0.5]}

def get_top_gene(df_tot, pval=0.01, n_top=25):
    dico_top_gene = {}
    top_genes = []
    df_de = df_tot[df_tot["pvals_adj"]<=pval]
    df_de["abs_scores"] = np.abs(df_de["scores"]).values
    for ct in df_de.celltype.unique():
        for ba in df_de.brain_region.unique():
            df_tmp =df_de[(df_de.celltype==ct) &(df_de.brain_region==ba)]
            df_tmp = df_tmp.sort_values("abs_scores", ascending=False)
            dico_top_gene["%s|%bs"] = df_tmp.loc[:n_top, "names"].values.tolist()
            top_genes += df_tmp.loc[:n_top, "names"].values.tolist()
    return dico_top_gene, top_genes

def plot_dotplot(dtt, savepath="", savename="", key_color="logfoldchanges", key_size="-log(q-value)", cmap="PiYG"):
    
    fig, ax = plt.subplots(figsize=(dtt["names"].nunique()/2, dtt["celltype"].nunique()))
    ax.grid()
    dtts = dtt.pivot("celltype", "names", key_color)      
    fontsize=14
    dtts_size = dtt.pivot("celltype", "names", key_size).fillna(0).astype(int)    
    max_or = dtts_size.max().max()
    min_or = dtts_size[dtts_size!=0].min().min()
    dtts_size = dtts_size.values*10
    plt.rcParams.update({'font.size': 18})
    with sns.axes_style("whitegrid"):
        ax = scattermap(dtts, marker_size=dtts_size, cmap=cmap,#vmin=-max(min_or,-min_or),vmax=max(min_or,-min_or),
                                ax=ax, cbar_kws={"shrink":0.5})

    for label in (ax.get_yticklabels()):
                cc = label.get_text()            
                label.set(color=color_cond[cc], label=cc, fontsize=18)
    labels = [item.get_text().split("-")[0] for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontsize=fontsize) 
    ll = []
    blue_line = lines.Line2D([], [], color="grey", 
                                                    marker="o",
                                                    linestyle='None',
                                                    markersize=np.sqrt(min_or*5),
                                     label=min_or)
    ll.append(blue_line)
    blue_line = lines.Line2D([], [], color="grey", 
                                                    marker="o",
                                                    linestyle='None',
                                                    markersize=np.sqrt(max_or*5), 
                                                     label=max_or)
    ll.append(blue_line)   
    leg1 = plt.legend(handles=ll, bbox_to_anchor=(1.15, 0.25), labelspacing=2, borderpad=2, title=key_size)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title( savename )
    plt.savefig(savepath + "dot_plot_gene" + savename+ ".svg", bbox_inches="tight")
    plt.show()

def plot_heatmap(dtt, savepath="",
                 savename="", 
                 key_color="logfoldchanges", #key_size="-log(q-value)", 
                 cmap="PiYG"):


    fig, ax = plt.subplots(figsize=(dtt["names"].nunique()/6, dtt["celltype"].nunique()))
    dtts = dtt.pivot("celltype", "names", key_color)
    fontsize=14
    plt.rcParams.update({'font.size': 18})
    with sns.axes_style("white"):
        sns.heatmap(dtts, cmap=cmap,#vmin=-max(min_or,-min_or),vmax=max(min_or,-min_or),
                    linewidth=0.2,xticklabels=True,
                                ax=ax, cbar_kws={"shrink":0.5})

    for label in (ax.get_yticklabels()):
                cc = label.get_text()
                label.set(color=color_cond[cc], label=txt, fontsize=18)
    labels = [item.get_text().split("-")[0] for item in ax.get_xticklabels()]
    ax.set_xticklabels(labels, fontsize=fontsize) 

            #ax.add_artist(leg1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title( savename )

    plt.savefig(savepath + "heatmap_gene" + savename+ ".svg", bbox_inches="tight")
    plt.show()


def plotVennBrainRegion(df_tot,
                            disease,
                              savepath,
                              FC=0.5,
                                pval=0.05,
                            x="logfoldchanges",
                              y = "-log_q_value_mn"):
    
    savepath = os.path.join(savepath, "venn_diagram/")
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    colors =[ [48/255.,65/255.,117/255.,0.5],[175/255.,193/255.,247/255.,0.5],[100/255.,136/255.,245/255.,0.5],[79/255.,108/255.,194/255.,0.5] ]
    celltypes = df_tot.celltype.unique()
    brain_regions = df_tot.brain_region.unique()

    colors_ba = [#[18/255.,50/255.,59/255., 0.5],
                     [118/255.,14/255.,63/255.,0.5],
                     #[92/255.,64/255.,77/255.,0.5],
                     [255/255., 192/255.,0, 0.5]]

    for ct in celltypes:
            list_sets = []
            groups = []
            len_tot = 0
            for ba in brain_regions:
                set1 = df_tot[(df_tot.celltype==ct) 
                               & (df_tot.brain_region == ba)
                               & (np.abs(df_tot[x])>=FC)
                               &  (df_tot[y]>=-np.log(pval))
                               ]["peakid"].values.tolist()
                list_sets.append(set1)
                groups.append(ba)
                len_tot += len(set1)
            if len_tot>0:
                labels = venn.get_labels(list_sets, fill=['number', "percent"])

                fig, ax = venn.venn2(labels, tuple(groups),colors=colors_ba)
                ax.set_title(ct)
                plt.savefig(savepath + "Venn_diagram_number_per_cond_" + ct + "_fc_%s.png"%(str(FC)))
                plt.show()
                plt.close()
                
                
def plotVennCelltype(df_tot,
                            disease,
                              savepath,
                              FC=0.5,
                                pval=0.01,
                            x="logfoldchanges",
                              y = "-log(q-value)"):
    
    savepath = os.path.join(savepath, "venn_diagram/")
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    colors =[ [48/255.,65/255.,117/255.,0.5],[175/255.,193/255.,247/255.,0.5],[100/255.,136/255.,245/255.,0.5],[79/255.,108/255.,194/255.,0.5] ]
    celltypes = df_tot.celltype.unique()
    brain_regions = df_tot.brain_region.unique()

    colors_ba = [#[18/255.,50/255.,59/255., 0.5],
                     [118/255.,14/255.,63/255.,0.5],
                     #[92/255.,64/255.,77/255.,0.5],
                     [255/255., 192/255.,0, 0.5]]

    celltypes = ["EXC", "INH", "AST", "MIC", "OLD", "OPCs"]
   
    for ba in brain_regions:
            list_sets = []
            groups = []
            len_tot = 0
            for ct in celltypes:
                set1 = df_tot[(df_tot.celltype==ct) 
                               & (df_tot.brain_region == ba)
                               & (np.abs(df_tot[x])>=FC)
                               &  (df_tot[y]>=-np.log(pval))
                               ]["peakid"].values.tolist()
                list_sets.append(set(set1))
                groups.append(ct)
                len_tot += len(set1)
            if len_tot>0:
                fig, ax =plt.subplots(figsize=(20, 8))
                #labels = venn.get_labels(list_sets, fill=['number', "percent"])
                supervenn(list_sets, groups,side_plots=False, ax=ax,
                          widths_minmax_ratio=0.005,
                          chunks_ordering='size',
                          #sets_ordering=celltypes,
                          color_cycle=[colors_ct[it] for it in celltypes],
                          min_width_for_annotation=100,
                          alternating_background=False,
                          side_plot_color="white",
                          bar_alpha=1,
                          col_annotations_area_height=1.3,
                          fontsize=16,
                         rotate_col_annotations=True)
                ax.set_title(ba)
                plt.savefig(savepath + "SuperVenn_diagram_celltype_" + ba + "_fc_%s.svg"%(str(FC)))
                plt.show()
                plt.close()
                
def stackedbarRegion(df_tot,
                            disease,
                        pair_brain_regions,
                              savepath,
                              FC=0.5,
                                pval=0.05,
                            x="logfoldchanges",
                              y = "-log(q-value)"):
    
    savepath = os.path.join(savepath, "venn_diagram/")
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    colors =[ [48/255.,65/255.,117/255.,0.5],[175/255.,193/255.,247/255.,0.5],[100/255.,136/255.,245/255.,0.5],[79/255.,108/255.,194/255.,0.5] ]
    celltypes = df_tot.celltype.unique()

    colors_ba=  [ [255/255., 192/255.,0, 0.5],
                     #[92/255.,64/255.,77/255.,0.5],
            [118/255.,14/255.,63/255.,0.5]
                    ]  
    df_ = pd.DataFrame(columns=["celltype", 
                                "region",
                                "Fraction OCR",
                               "error"])
                                #pair_brain_regions[0], 
                                #pair_brain_regions[0]+"+"+pair_brain_regions[1],
  
    for ba in pair_brain_regions:#pair_brain_regions[1]])
    
        list_sets = []
        groups = []
        tot = 0
        for ct in celltypes:
            set1 = df_tot[(df_tot.celltype==ct) 
                               & (df_tot.brain_region == ba)
                               & (np.abs(df_tot[x])>=FC)
                               &  (df_tot[y]>=-np.log(pval))
                               ]["peakid"].values.tolist()
            list_sets.append(set1)
            groups.append(ct)
            tot = len(set1) 
        inter = [it for it in list_sets[0] if it in list_sets[1]]
        l1 = (len(list_sets[0])/2)/tot
        l2 = len(inter)/tot
        l3 = (len(list_sets[0])+(len(list_sets[1])/2))/tot
        df_.loc[len(df_),:]=[ct,pair_brain_regions[0],l1,len(list_sets[0])/(2*tot) ]#,l3]
        df_.loc[len(df_),:]=[ct,pair_brain_regions[1],l3, len(list_sets[1])/(2*tot)]#,l3]

    colors = [colors_ba[0],
              colors_ba[1]]
    fig, ax =plt.subplots(figsize=(5, 6))
    sns.pointplot('region', 'Fraction OCR', hue='celltype',scale=0.1,
                  linewidth=2,
        data=df_, dodge=True, join=False, ci=None, palette=colors,size=1,
                     ax=ax)

    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
    errors = df_[df_.region==pair_brain_regions[0]]["error"].values.tolist()+df_[df_.region==pair_brain_regions[1]]["error"].values.tolist()
    colors = [colors[0]]*6 + [colors[1]]*6
    ax.errorbar(x_coords, y_coords, yerr=errors,linewidth=2,
        ecolor=colors, fmt=' ', zorder=-1)
    ax.legend(bbox_to_anchor=(1.55,0.5))

def heatmapRegion(df_tot,
                    disease,
                    pair_brain_regions,
                    savepath,
                    FC=0.5,
                    pval=0.05,
                    x="logfoldchanges",
                    y = "-log(q-value)"):


    savepath = os.path.join(savepath, "venn_diagram/")
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    colors =[ [48/255.,65/255.,117/255.,0.5],[175/255.,193/255.,247/255.,0.5],[100/255.,136/255.,245/255.,0.5],[79/255.,108/255.,194/255.,0.5] ]
    celltypes = df_tot.celltype.unique()

    colors_ba=  [ [255/255., 192/255.,0, 0.5],
                     #[92/255.,64/255.,77/255.,0.5],
            [118/255.,14/255.,63/255.,0.5]
                    ]  
    df_ = pd.DataFrame(columns=["celltype", 
                          
                                pair_brain_regions[0], 
                                pair_brain_regions[0]+"+"+pair_brain_regions[1],
                                pair_brain_regions[1]])
    for ct in celltypes:
        list_sets = []
        groups = []
        for ba in pair_brain_regions:
            set1 = df_tot[(df_tot.celltype==ct) 
                               & (df_tot.brain_region == ba)
                               & (np.abs(df_tot[x])>=FC)
                               &  (df_tot[y]>=-np.log(pval))
                               ]["peakid"].values.tolist()
            list_sets.append(set1)
            groups.append(ba)
        tot = len(list_sets[0]) + len(list_sets[1])
        inter = [it for it in list_sets[0] if it in list_sets[1]]
        l1 = (len(list_sets[0])-len(inter))*100/tot
        l2 = len(inter)*100/tot
        l3 = (len(list_sets[1])-len(inter))*100/tot
        df_.loc[len(df_),:]=[ct,np.round(l1,2)+0.001,np.round(l2,2)+0.001,np.round(l3,2) ]#,l3]

    colors = [colors_ba[0],
              #(np.asarray(colors_ba[0]) + np.asarray(colors_ba[1]))/2, 
              colors_ba[1]]
    df_.set_index("celltype", inplace=True)
    df_ = df_.astype(float)
    fig, ax = plt.subplots(figsize=(6,10))
    cmap_r = ['#001414', '#00312f', '#004d4b', '#006a68', '#008685', '#18a2a1', '#48bfbe', '#86dbdb', '#d0f7f7']

    newcmp = LinearSegmentedColormap.from_list("cmap_petrol", cmap_r[::-1], N=255)
    sns.heatmap(df_, annot=True,fmt='.2f',  ax=ax, 
                linewidth=0.2,
                cmap=newcmp, norm=matplotlib.colors.LogNorm())

    #ax.legend(bbox_to_anchor=(1.55,0.5))
    for label in (ax.get_yticklabels()):

        cc = label.get_text()
                
        label.set(color=colors_ct[cc], label=label, fontsize=18)
    for label in (ax.get_xticklabels()):

        cc = label.get_text()
                
        label.set(color=colors_brain_region[cc], label=label, fontsize=18, rotation=45)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig(savepath + "Heatmap_overlap_de_ocrd_" + ct + "_fc_%s.svg"%(str(FC)))
    plt.show()
    plt.close()

def get_go(gene_list, background,
        dtbs = ['GO_Biological_Process_2021','GO_Molecular_Function_2021', 'GO_Cellular_Component_2021',"BioPlanet_2019"]
          ):
    terms = []
    for it in dtbs :
        enr_res = gseapy.enrichr(gene_list=gene_list,
                     organism='Human',
                    background=background,
                     gene_sets=it,
                     cutoff = 0.5)
        gg = enr_res.res2d
        gg["database"] = it
        terms.append(gg)
    return pd.concat(terms)

def get_go_prerank(gene_list_df, background,
                dtbs = ['GO_Biological_Process_2021',
                        'GO_Molecular_Function_2021', 
                        'GO_Cellular_Component_2021',
                        "BioPlanet_2019"]
                  ):
    terms = []
    
    for it in dtbs :
        
        enr_res = gseapy.prerank(rnk=gene_list_df,
                     organism='Human',
                    min_size=1,
                    background=background,
                     gene_sets=it,
                     cutoff = 0.5)
        gg = enr_res.res2d
        gg["database"] = it
        terms.append(gg)
    return pd.concat(terms)



def plot_GO(concat_all, brain_region, celltype,savepath, FC, score_key="overlap_ratio", mix_brain_region=False):
    
    color_cond= {"HCADD":"#262730","ADD":"#F57764", "ResilientADD":"#77BA99", "RADOther":"#77BA99"}
    colors_brain_region= {"CAUD":[18/255.,50/255.,59/255., 1], "SMTG":[118/255.,14/255.,63/255.,1], "PARL":[92/255.,64/255.,77/255.,1],"HIPP":[255/255., 192/255.,0, 1] }
    #concat_all = concat_all[concat_all["Number of genes"].astype(int)>1]
    palette = { 'GO_Biological_Process_2021':"#885A89", ##66c2a5",
               #'GO_Cellular_Component_2021':"#fc8d62",
               "GO_Cellular_Component_2021":"#8AA8A1",
               'GO_Molecular_Function_2021':"#3B3B3B",
               'DisGeNET':"#e78ac3","BioPlanet_2019":"#2E294E", 'KEGG_2021_Human':"#a6d854", "MSigDB_Hallmark_2020":"#e78ac3",
              'WikiPathways_2019_Human':"#ff7f00",'GWAS_Catalog_2019':"#ff7f00",
              'Reactome_2016':"#cab2d6",'BioCarta_2016':"#DB7C26"}
    list_to_plot = [['GO_Biological_Process_2021','GO_Cellular_Component_2021','GO_Molecular_Function_2021'],
                    ['KEGG_2021_Human',"MSigDB_Hallmark_2020",
                  'WikiPathways_2019_Human',
              'Reactome_2016','BioCarta_2016']]

    norm=plt.Normalize(1,150)
    colorlist=["darkorange", "gold", "lawngreen", "lightseagreen"]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    list_to_plot = [concat_all.Gene_set.unique().tolist()]
    concat_all = concat_all[concat_all.brain_region.isin(brain_region)]
    for dtb in list_to_plot:
        dtt = concat_all[concat_all.Gene_set.isin( dtb)]
        if len(dtt) >0:
            fig, ax = plt.subplots(figsize=(5, dtt["Term"].nunique()/2*1))
            ax.grid()
            #dtt = dtt[dtt.comparison==comp]
            if mix_brain_region:
                dtt["cond_region_celltype"] =  dtt["comparison"] + "|" + dtt["celltype"]\

            else:
                dtt["cond_region_celltype"] =  dtt["comparison"] + "|" + dtt["celltype"]+ "_" + dtt["brain_region"]
            dtt["Term"] = dtt["Gene_set"] + "|" + dtt["Term"]
            dtts = dtt.pivot("Term", "cond_region_celltype", "-log(p-value)")
            
            dtt["overlap_ratio"] *= 100
            dtt["overlap_ratio"] = np.ceil(dtt["overlap_ratio"]).values
            dtt[score_key] = np.ceil(dtt[score_key]).values

        
            dtts_size = dtt.pivot("Term", "cond_region_celltype", score_key).fillna(0).astype(int)
          
            all_comb = ["RADvsAll|%s_%s"%(ct,ba) for ct in celltype for ba in brain_region]
            for it in all_comb:
                if not it in dtts.columns.tolist():
                    dtts[it] = np.nan
                    dtts_size[it] = 0
            dtts = dtts[all_comb]
            dtts_size = dtts_size[all_comb]
            max_or = dtts_size.max().max()
            min_or = dtts_size[dtts_size!=0].min().min()
            dtts_size = dtts_size.values*5
            plt.rcParams.update({'font.size': 18})
            with sns.axes_style("whitegrid"):
                ax = scattermap(dtts, marker_size=dtts_size, cmap=newcmp,
                                ax=ax, cbar_kws={"shrink":0.5})
            for label in (ax.get_xticklabels()):
                txt = label.get_text().split("|")[1]
                reg = txt.split("_")[1]
                ct_t = txt.split("_")[0]
                cond = label.get_text().split("|")[0]
                label.set(color=colors_brain_region[reg], label=ct_t, fontsize=18, rotation=90 )
            for label in (ax.get_yticklabels()):

                cc = label.get_text().split("|")[0]
                txt = label.get_text().split("|")[-1] + "(" + cc[1] + ")"
                label.set(color=palette[cc], label=txt, fontsize=18)
            labels = [item.get_text().split("|")[1].split("_")[0] for item in ax.get_xticklabels()]
            ax.set_xticklabels(labels)
            labels = [item.get_text().split("|")[-1].split("(")[0] for item in ax.get_yticklabels()]
            ax.set_yticklabels(labels) 
            ll = []

            blue_line = lines.Line2D([], [], color="grey", 
                                                    marker="o",
                                                    linestyle='None',
                                                    markersize=np.sqrt(min_or*5),
                                     label=min_or)
            ll.append(blue_line)
            blue_line = lines.Line2D([], [], color="grey", 
                                                    marker="o",
                                                    linestyle='None',
                                                    markersize=np.sqrt(max_or*5), 
                                                     label=max_or)
            ll.append(blue_line)   
            leg1 = plt.legend(handles=ll, bbox_to_anchor=(1.5, 0.25), labelspacing=1, borderpad=0.5, title=score_key)
            ll = []
            for it in brain_region:#dtt["brain_region"].unique():
                blue_line = lines.Line2D([], [], color=colors_brain_region[it], 
                                                    marker="o",
                                                    linestyle='None',
                                                    markersize=8, label=it)
                ll.append(blue_line)
            leg2 = plt.legend(handles=ll, bbox_to_anchor=(1.8, 0.55))
            ax.add_artist(leg1)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title( "".join(dtb) + " FC " + str(FC) )
            if mix_brain_region:
                plt.savefig(savepath + "ALLregionEnrichr" + str(FC) + "".join(dtb) + ".png", bbox_inches="tight")
            else:
                plt.savefig(savepath + "Enrichr" + str(FC) + "".join(dtb) + ".svg", bbox_inches="tight")
            plt.show()
            plt.close("all")
            
def draw_volcanos_ct_ba_pairs(df_tot, celltypes, 
                              List_brain_regions,
                              model_path,
                              hue="peakType2",
                              x = "log2_mean_ratio",
                              y = "-log_q_value_mn",
                              annotation=False,
                              annotation_AD=None,
                              ylim = -np.log(0.05),
                              xlim = 1,
                             figsize=(22, 26),
                              specific_peak=None,
                              ymax=40,
                              xmax=5,
                              colorlist=('#724600', '#004d4b'),
                              extension=".png"
                             ):
    model_path = os.path.join(model_path, "volcano_plots/")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    annot_peak = df_tot[hue].unique().tolist()
    colors = list(sns.color_palette('husl',len(annot_peak)))
    #colors = ["#A7B7E8", "#7B9C51","#E88F79", "#9C6659", "#6DCDE8", "#9BE8A4"]

    dico_col = {an:cc for an,cc in zip(annot_peak, colors)}
    hue_order = annot_peak

    if not specific_peak ==None:
        df_tot = df_tot[df_tot[hue].isin(specific_peak)]
    df_tot = df_tot.loc[~df_tot.isin([np.nan, np.inf, -np.inf]).any(1), :]
    for ct in celltypes:
        df_ = df_tot[df_tot.celltype==ct].reset_index()
        fig, axes = plt.subplots(1, len(List_brain_regions), figsize=figsize, sharey=True)
        axes =axes.flatten()
        for i,ba in enumerate(List_brain_regions):
                    ax = axes[i]
                    jj = df_[df_["brain_region"] == ba]
                    #jj.reset_index(inplace=True)
                    sns.set(font_scale=1)                    
                    #ymax = 40 #max(jj["-log_q_value"].max(),xlim+1)
                   
                    sns.set_style("whitegrid", {'axes.grid' : False})
                    jj_s1 = jj[((jj[y]>=ylim) & (jj[x] >=xlim))]
                    g =sns.scatterplot(ax=ax, data=jj_s1, y=y, x=x, color=colorlist[1],alpha=0.7)#,hue=hue,hue_order=hue_order)#, color="#92374D"
                    jj_s2 = jj[((jj[y]>=ylim) & (jj[x] <=-xlim))]
                    g =sns.scatterplot(ax=ax, data=jj_s2, y=y, x=x, color=colorlist[0],alpha=0.7)#,hue=hue,hue_order=hue_order)#, color="#92374D"
                    xmax = max(jj[jj[y]>=ylim][x].max(), -jj[jj[y]>=ylim][x].min())
                    if np.isnan(xmax):
                        xmax=3
                    xmax =max(xmax, 3)                    
                    jj_ns = jj[((jj[y]<ylim) & (jj[x] <=xlim)) | ((jj[y]<ylim) & (jj[x] >=-xlim)) |((jj[y]>ylim) & (jj[x] >=-xlim) & (jj[x] <=xlim))]
                    g =sns.scatterplot(ax=ax, data=jj_ns, y=y, x=x, color="grey", s=8, alpha=0.5)
                    ax.set_xlim(-xmax, xmax)
                    ax.hlines(y=ylim,xmin=-xmax,  xmax =xmax, color= 'k', linestyle='--', linewidth=0.8)
                    ax.vlines(x=-xlim,ymin=0.0,  ymax =ymax+1, color= 'k', linestyle='--', linewidth=0.8)
                    ax.vlines(x=xlim,ymin=0.0,  ymax =ymax+1, color= 'k', linestyle='--', linewidth=0.8)

                    ax.title.set_fontsize(22)
                    ax.text(-xmax, ymax+1, "#peaks: %d"%(jj_s1.shape[0]+ jj_s2.shape[0]), alpha=1, fontsize=25)
                    ax.tick_params(labelsize=22)
                    ax.xaxis.label.set_size(22)
                    ax.set_xlabel(" ")
                    ax.yaxis.label.set_size(22)
                    ax.set_title(ba, fontsize=22)
                    if i<len(List_brain_regions)*len(pairs) and ax.get_legend() is not None:
                        ax.get_legend().remove()

                    if annotation:
                        jj_s1["abs_%s"%x] = np.abs(jj_s1[x])
                        gg = jj_s1.sort_values("abs_%s"%x, ascending=False)[:10]
                        gg = gg[~gg["nearestGeneChip"].str.contains("LOC10")]
                        # if False:
                        #     for line in range(0,jj_s.shape[0]):
                        #         if jj_s["nearestGeneChip"].iloc[line] in list(set(do_map.Symbol.values.tolist())): #["APOE","GFAP" ] :
                        #             #if "lOC10"
                        #             ax.text(jj_s[x].iloc[line], jj_s[y].iloc[line]+0.005, jj_s["nearestGene"].iloc[line], horizontalalignment='left', color='black')
                        for lk, line in enumerate(range(0,gg.shape[0])):
                         #if  (jj["nearestGene"].iloc[line] in list_gene_ad) :#or (jj_s["nearestGene"].iloc[line] in list_neaw):
                             ax.text(gg[x].iloc[line], gg[y].iloc[line], gg["nearestGeneChip"].iloc[line], horizontalalignment='left', color='black')
                        jj_s2["abs_%s"%x] = np.abs(jj_s2[x])
                        gg = jj_s2.sort_values("abs_%s"%x, ascending=False)[:10]
                        gg = gg[~gg["nearestGeneChip"].str.contains("LOC10")]

                        for lk, line in enumerate(range(0,gg.shape[0])):
                         #if  (jj["nearestGene"].iloc[line] in list_gene_ad) :#or (jj_s["nearestGene"].iloc[line] in list_neaw):
                             ax.text(gg[x].iloc[line], gg[y].iloc[line], gg["nearestGeneChip"].iloc[line], horizontalalignment='left', color='black')
                    if annotation_AD is not None:
                            targets = annotation_AD
                            for line in range(0,jj_s1.shape[0]):
                                if jj_s1["nearestGeneChip"].iloc[line] in targets: #LAP3", "MACROD1", "SEMA7A"]: #UNC5C"]:#MAPT", "GFAP", "BIN1"]:
                                    ax.text(jj_s1[x].iloc[line], jj_s1[y].iloc[line]+0.005, jj_s1["nearestGeneChip"].iloc[line], 
                                            horizontalalignment='left', color='#9C2400', fontsize=14)
                            for line in range(0,jj_s2.shape[0]):
                                if jj_s2["nearestGeneChip"].iloc[line] in targets: #LAP3", "MACROD1", "SEMA7A"]: #UNC5C"]:#MAPT", "GFAP", "BIN1"]:
                                    ax.text(jj_s2[x].iloc[line], jj_s2[y].iloc[line]-0.005, jj_s2["nearestGeneChip"].iloc[line], 
                                            horizontalalignment='left', color='#9C2400', fontsize=14)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(model_path + ct +ba + "Volcano_"+ y + "_FC_" + str(xlim) +extension, bbox_inches='tight', dpi=1200)
        plt.show()
        plt.close("all")
        plt.clf()


def plot_GO_2(concat_all, brain_region, celltype,savepath, FC, score_key="overlap_ratio", mix_brain_region=False):
    
    color_cond= {"HCADD":"#262730","ADD":"#F57764", "ResilientADD":"#77BA99", "RADOther":"#77BA99"}
    colors_brain_region= {"CAUD":[18/255.,50/255.,59/255., 1], "SMTG":[118/255.,14/255.,63/255.,1], "PARL":[92/255.,64/255.,77/255.,1],"HIPP":[255/255., 192/255.,0, 1] }
    colors ={"genic":"#BBBE64", "intergenic":"#8E5572"}
    palette = { 'GO_Biological_Process_2021':"#885A89", ##66c2a5",
               #'GO_Cellular_Component_2021':"#fc8d62",
               "GO_Cellular_Component_2021":"#8AA8A1",
               'GO_Molecular_Function_2021':"#3B3B3B",
               'DisGeNET':"#e78ac3","BioPlanet_2019":"#2E294E", 'KEGG_2021_Human':"#a6d854", "MSigDB_Hallmark_2020":"#e78ac3",
              'WikiPathways_2019_Human':"#ff7f00",'GWAS_Catalog_2019':"#ff7f00",
              'Reactome_2016':"#cab2d6",'BioCarta_2016':"#DB7C26"}
    list_to_plot = [['GO_Biological_Process_2021','GO_Cellular_Component_2021','GO_Molecular_Function_2021'],
                    ['KEGG_2021_Human',"MSigDB_Hallmark_2020",
                  'WikiPathways_2019_Human',
              'Reactome_2016','BioCarta_2016']]

    norm=plt.Normalize(1,150)
    colorlist=["darkorange", "gold", "lawngreen", "lightseagreen"]
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    list_to_plot = [concat_all.Gene_set.unique().tolist()]
    concat_all = concat_all[concat_all.brain_region.isin(brain_region)]
    for dtb in list_to_plot:
        dtt = concat_all[concat_all.Gene_set.isin( dtb)]
        if len(dtt) >0:
            fig, ax = plt.subplots(figsize=(10, dtt["Term"].nunique()/2*1))
            ax.grid()
            #dtt = dtt[dtt.comparison==comp]
            if mix_brain_region:
                dtt["cond_region_celltype"] =  dtt["comparison"] + "|" + dtt["celltype"]\

            else:
                dtt["cond_region_celltype"] =  dtt["comparison"] + "|" + dtt["celltype"]+ "_" + dtt["Type_enhancer"]
            dtt["Term"] = dtt["Gene_set"] + "|" + dtt["Term"]
            dtts = dtt.pivot("Term", "cond_region_celltype", "-log(p-value)")
            
            dtt["overlap_ratio"] *= 100
            dtt["overlap_ratio"] = np.ceil(dtt["overlap_ratio"]).values
            dtt[score_key] = np.ceil(dtt[score_key]).values

        
            dtts_size = dtt.pivot("Term", "cond_region_celltype", score_key).fillna(0).astype(int)
          
            all_comb = ["RADvsAll|%s_%s"%(ct,ba) for ct in celltype for ba in dtt["Type_enhancer"].unique()]
            for it in all_comb:
                if not it in dtts.columns.tolist():
                    dtts[it] = np.nan
                    dtts_size[it] = 0
            dtts = dtts[all_comb]
            dtts_size = dtts_size[all_comb]
            max_or = dtts_size.max().max()
            min_or = dtts_size[dtts_size!=0].min().min()
            dtts_size = dtts_size.values*20
            plt.rcParams.update({'font.size': 18})
            with sns.axes_style("whitegrid"):
                ax = scattermap(dtts, marker_size=dtts_size, cmap=newcmp,
                                ax=ax, cbar_kws={"shrink":0.5})


            for label in (ax.get_xticklabels()):
                txt = label.get_text().split("|")[1]
                reg = txt.split("_")[1]
                ct_t = txt.split("_")[0]
                cond = label.get_text().split("|")[0]
                label.set(color=colors[reg], label=ct_t, fontsize=18, rotation=90 )
            for label in (ax.get_yticklabels()):

                cc = label.get_text().split("|")[0]
                txt = label.get_text().split("|")[-1] + "(" + cc[1] + ")"
                label.set(color=palette[cc], label=txt, fontsize=18)
            labels = [item.get_text().split("|")[1].split("_")[0] for item in ax.get_xticklabels()]
            ax.set_xticklabels(labels)
            labels = [item.get_text().split("|")[-1].split("(")[0] for item in ax.get_yticklabels()]
            ax.set_yticklabels(labels) 
            ll = []

            blue_line = lines.Line2D([], [], color="grey", 
                                                    marker="o",
                                                    linestyle='None',
                                                    markersize=np.sqrt(min_or*5),
                                     label=min_or)
            ll.append(blue_line)
            blue_line = lines.Line2D([], [], color="grey", 
                                                    marker="o",
                                                    linestyle='None',
                                                    markersize=np.sqrt(max_or*5), 
                                                     label=max_or)
            ll.append(blue_line)   
            leg1 = plt.legend(handles=ll, bbox_to_anchor=(1.5, 0.25), labelspacing=1, borderpad=0.5, title=score_key)
            ll = []
            for it in dtt["Type_enhancer"].unique():
                blue_line = lines.Line2D([], [], color=colors[it], 
                                                    marker="o",
                                                    linestyle='None',
                                                    markersize=8, label=it)
                ll.append(blue_line)
            leg2 = plt.legend(handles=ll, bbox_to_anchor=(1.8, 0.55))
            ax.add_artist(leg1)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title( "".join(dtb) + " FC " + str(FC) )
            if mix_brain_region:
                plt.savefig(savepath + "ALLregionEnrichr" + str(FC) + "".join(dtb) + ".png", bbox_inches="tight")
            else:
                plt.savefig(savepath + "Enrichr" + str(FC) + "".join(dtb) + ".svg", bbox_inches="tight")
            plt.show()
            plt.close("all")