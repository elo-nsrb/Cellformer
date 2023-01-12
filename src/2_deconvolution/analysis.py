import os 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.transforms as transforms
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
#from statannot import add_stat_annotation
import scanpy as sc
import anndata as ad
import episcanpy.api as epi

def bland_altman_plot(data1, data2, ax, *args, **kwargs):

    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2                   # Difference between data1 and data2
    md = np.mean(diff)                   # Mean of the difference
    sd = np.std(diff, axis=0)            # Standard deviation of the difference

    ax.scatter(mean, diff, *args, **kwargs)
    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    ax.set_ylabel("mean")
    ax.set_xlabel("Difference")

def plot_bland_altman(pred, true, model_dir, celltypes,prefix):
    if len(celltypes)<4:
        fig, axes = plt.subplots(1,4, figsize=(16,8))
    else:
        fig, axes = plt.subplots(2,4, figsize=(16,8))
    axes = axes.flatten()
    for i, it in enumerate(celltypes):
        x = true[:,i,:].ravel().astype("float32")
        y = pred[:,i,:].ravel().astype("float32")
        bland_altman_plot(x, y, axes[i])
        axes[i].set_title(it,fontsize=16)
                                
    plt.savefig(model_dir + "/_" + prefix +"bland_altman_plot_"+ ".png",
                bbox_inches="tight")
                            #plt.show()
    plt.close("all")

def prepare_mixtures(mixtures, true, annot, type="sc"):
    if type =="pseudobulk":
        mixtures["Brain_region"] = mixtures["Sample_num"].str.split("_", expand=True)[2]
        mixtures["Sample_num"] = mixtures["Sample_num"].str.split("_", expand=True)[0] + "_" + mixtures["Sample_num"].str.split("_", expand=True)[1]
    else:
        mixtures["Brain_region"] = mixtures["Sample_num"].str.split("_", expand=True)[2]
        mixtures["Sample_num"] = mixtures["Sample_num"].str.split("_", expand=True)[0] + "_" + mixtures["Sample_num"].str.split("_", expand=True)[1]
    array_color = np.zeros_like(true)
    map_colors = {"CAUD":1, "PARL":2, "SMTG":3, "MDFG":4, "HIPP":5, "SUNI":7}
    for col in mixtures.Brain_region.unique():
        select_indices = list(np.where(mixtures.Brain_region == col)[0])
        array_color[select_indices,:] = map_colors[col]
    array_color_perso = np.zeros_like(true)
    nb_samples = mixtures["Sample_num"].nunique()
    sub_id = mixtures["Sample_num"].unique().tolist()
    map_colors_perso = {id:k for id,k in zip(sub_id, np.arange(nb_samples))}
    #map_colors_perso = {"11_0393":1, "09_1589":2, "14_0586":3, "09_35":4, "06_0615":5, "14_1018":6, }
    for col in mixtures.Sample_num.unique():
            select_indices = list(np.where(mixtures.Sample_num == col)[0])
            array_color_perso[select_indices,:] = map_colors_perso[col]
    return mixtures, array_color, map_colors, array_color_perso, map_colors_perso

def plot_scatter_plot_per_metadata_color(true, pred, 
                                        celltypes, model_dir,
                                        prefix,
                                        array_color, map_colors, metadata):

    thres=2
    jj = 1000
    cmap = {it:plt.cm.tab20(it) for it in np.arange(20)}
    #cmap = { 0:'#A40AFC',1:'#BEFC9A',2:'#F7075F',3:'#1F8C64',4:'#1283C4',5:"#D1C732" , 6:"#F67E7D", }
    if len(celltypes)<=4:
        fig, axes = plt.subplots(1,4, figsize=(16,8))
    else:
        fig, axes = plt.subplots(2,4, figsize=(16,8))

    axes=axes.flatten()
    for it, ct in enumerate(celltypes):
        #for ind, ba in enumerate(mixtures.Brain_region.unique()):
        x = true[:,it,:].ravel().astype("float32")
        y = pred[:,it,:].ravel().astype("float32")
        cc = array_color[:,it,:].ravel().astype("float32")
        #x = x[cc==map_colors[ba]]
        #y = y[cc==map_colors[ba]]
        #cc = cc[cc==map_colors[ba]]
        colors = [cmap[j] for j in cc]
        #axes[it].plot(x, x, color="r", linewidth=2,label='x=x')

        axes[it].scatter(x, y, c=colors, alpha=0.3)

        axes[it].set_ylabel("prediction")
        axes[it].set_xlabel("True")
        #axes[it].legend(loc="upper left")
        axes[it].set_title(ct,fontsize=16)
        # The following two lines generate custom fake lines that will be used as legend entries:
    markers = [plt.Line2D([0,0],[0,0],color=cmap[color], marker='o', linestyle='') for color in map_colors.values()]
    plt.legend(markers, map_colors.keys(), numpoints=1)
    plt.title(ct)
    fig.tight_layout()
    plt.savefig(model_dir + "/_" + prefix +"_New_scatterplot_colors_"
                + metadata + "_" + ct + ".png", bbox_inches="tight")
    plt.close("all")


def main():
    dataset = "/home/eloiseb/data/scatac-seq/"
    model_dir = "/home/eloiseb/experiments/deconv_peak/n_sep_2_mse_normMax_random_data/"
    pure = False

    prefix = "test_False" 
    if pure:
        pred = np.load(model_dir + "predictions_" + prefix + "random_nb_cells__pure_pure_.npz")["mat"]
        true = np.load(model_dir + "true_" + prefix +"random_nb_cells__pure_pure_.npz")["mat"]
        mixtures = pd.read_csv(model_dir + "mixtures_" + prefix +"random_nb_cells__pure_pure_.csv", index_col=1)
        prefix += "_pure"
    else:
        pred = np.load(model_dir + "predictions_" + prefix + ".npz")["mat"]
        true = np.load(model_dir + "true_" + prefix +".npz")["mat"]
        mixtures = pd.read_csv(model_dir + "mixtures_" + prefix +".csv", index_col=1)
    (mixtures, 
     array_color_ba, map_colors_ba,
     array_color_perso, 
     map_colors_perso) = prepare_mixtures(mixtures, true)
    annot = pd.read_csv(dataset + "no_filt_ctrl_annotations.csv")
    celltypes = ["Glials", "Neurons"]
    model_dir = os.path.join(model_dir, "analysis/")
    print(model_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    plot_scatter_plot_per_metadata_color(true, pred, celltypes, model_dir,
                                        prefix,
                                        array_color_ba, map_colors_ba, 
                                        "brain_area")
    plot_scatter_plot_per_metadata_color(true, pred, celltypes, model_dir,
                                        prefix,
                                        array_color_perso, map_colors_perso, 
                                        "Subject")
    plot_bland_altman(pred, true, model_dir, celltypes,prefix)

if __name__ == '__main__':
    main()
 
