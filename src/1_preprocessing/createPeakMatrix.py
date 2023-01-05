#!/usr/bin/env python
# coding: utf-8
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
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import episcanpy.api as epi
from tqdm import tqdm
import statsmodels.stats.multitest as multi
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', 
                    default="./data/",
                    help='Location to save pseudobulks data')

parser.add_argument('--savepath', 
                    default="./data/",
                    help='Location to save pseudobulks data')
parser.add_argument('--FC', default=1)
parser.add_argument('--pvalue', default=0.001)




def read_adata(path):
    peaks = pd.read_csv(path + "peakset.csv", index_col=None)
    mtx = path + "my_mat.mtx"
    adata = ad.read_mtx(mtx).T
    col = pd.read_csv(path + "coldata.csv", index_col=None)
    gg = col.groupby(["DonorID", "Region", "celltype"]).size()
    ax = gg.unstack(fill_value=0).plot.bar(stacked=True,figsize=(20, 10),colormap='PiYG')
    for rect in ax.patches:
        # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        label_text = '%s'%str(int(round(height,0)))#f'{height:.0f}'
        label_x = x + width / 2
        label_y = y + height / 2

        if height > 0:
            ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=8)
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.savefig(path + "/proportion_Cells_per_subject.png")

    adata.var["chrm"] = peaks.seqnames.values
    adata.var["start"] = peaks.start.values
    adata.var["end"] = peaks.end.values
    adata.var["score"] = peaks.score.values
    adata.var["groupScoreQuantile"] = peaks.groupScoreQuantile.values
    adata.var["distToGeneStart"] = peaks.distToGeneStart.values
    adata.var["nearestGene"] = peaks.nearestGene.values
    adata.var["peakType"] = peaks.peakType.values
    adata.var["distToTSS"] = peaks.distToTSS.values

    adata.obs["Sample_num"] = col.DonorID.values
    adata.obs["Clusters"] = col.Clusters.values
    adata.obs["celltype"] = col.celltype.values
    adata.obs["Region"] = col.Region.values
    return adata



def filter_matrix(adata, path, 
                  fdr_th = 0.05,
                    log2fc = 1):
    adata.X = csr_matrix(adata.X)
    adata.var["peakID"] = adata.var["chrm"].astype("str") + "_" + adata.var["start"].astype("str") + "_" + adata.var["end"].astype("str")
    adata.var
    marker_peak_mean = pd.read_csv(path + "markerspeak_mean.csv", index_col=0)
    marker_peak_mean_diff = pd.read_csv(path + "markerspeak_MeanDiff.csv", index_col=0)
    marker_peak_mean
    marker_peak_fdr = pd.read_csv(path + "markerspeak_FDR.csv", index_col=0)
    marker_peak_log2fc = pd.read_csv(path + "markerspeak_log2fc.csv", index_col=0)
    marker_row = pd.read_csv(path + "markerspeak_rowdata.csv", index_col=0)
    marker_row["peakID"] = marker_row["seqnames"] + "_" + marker_row["start"].astype("str") + "_" + marker_row["end"].astype("str")
    marker_row
    adata[:, adata.var["peakID"].isin(marker_row.peakID.values.tolist())]
    marker_peak_mean.index = adata.var.index
    marker_peak_fdr.index = adata.var.index
    marker_peak_log2fc.index = adata.var.index
    mat = np.where((marker_peak_fdr.values < fdr_th) & (np.abs(marker_peak_log2fc) > log2fc))[0]
    adata.var.iloc[mat] ##148243
    tmp = np.zeros_like(marker_peak_fdr.values)
    tmp[mat] = 1
    peaks_to_keep = np.where(tmp.sum(1)>0)
    print("# of peaks after filtering :" + str(len(peaks_to_keep[0])))##142412
    hh = [it for it in range(marker_row.shape[0]) if it not in peaks_to_keep[0]]
    marker_peak_log2fc.iloc[peaks_to_keep[0],:]
    marker_peak_fdr.iloc[peaks_to_keep[0],:]
    adata_filt = adata[:,adata.var.iloc[peaks_to_keep[0]].index]
    print(adata_filt)###142412
    return adata_filt


def main():
    args = parser.parse_args()
    path = args.path 
    adata = read_adata(path)
    adata = filter_matrix(adata, path, 
                  fdr_th = args.pvalue,
                    log2fc = args.FC)
    savepath = args.savepath
    adata.write_h5ad(savepath + "adata_peak_matrix.h5")
    peaks_tsv = adata.var[["chrm", "start", "end"]]
    peaks_tsv["strand"] = "."
    peaks_tsv.to_csv(savepath + "peaks" + ".tsv", 
                                   sep="\t",
                                   index=None,
                                  header=None)



if __name__ == "__main__":
    main()
