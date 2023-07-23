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
parser.add_argument('--path', default="./data/",
                    help='Location to save pseudobulks data')


def main():
    args = parser.parse_args()
    path = args.path 
    
    ##load data
    adata = ad.read_h5ad(path + "adata_peak_matrix.h5")
    adata = adata[adata.obs.celltype != "UnknownNeurons"]
    celltypes = adata.obs["celltype"].sort_values().unique().tolist()
    print(celltypes)
    df_ = adata.to_df()
    df_["Sample_num"] = (adata.obs["Sample_num"].astype(str).values 
            + "_" + adata.obs["Region"].astype(str).values)
    df_["celltype"] = adata.obs["celltype"].astype(str).values 

    # Total number of cells per sample
    total_cell_per_sample = df_.groupby(["Sample_num"]).size().reset_index()
    print(total_cell_per_sample)

    ## Aggragate cell per sample &cell type
    df_sum = df_.groupby(["Sample_num", "celltype"]).sum()
    df_sum = df_sum.reset_index()
    df_sum.to_csv(path + "aggregated_sc_per_celltype.csv", index=None)

    ## Aggragate cell per sample
    df_mixture = df_.groupby(["Sample_num"]).sum()
    df_mixture = df_mixture.reset_index()
    #Save synthetic mixtures
    df_mixture.to_csv(path + "aggregated_sc_mixture.csv", index=None)

    #derive cell type specific mixtures for each sample
    array_agg_sc = np.zeros((df_mixture.Sample_num.nunique(),
                            len(celltypes),
                            adata.shape[1]))
    for su_idx, su in enumerate(df_mixture["Sample_num"].values.tolist()):
        tmp = df_sum[df_sum.Sample_num == su]
        print(su)
        nb_cell = total_cell_per_sample[
                total_cell_per_sample.Sample_num == su]
        print(nb_cell[0])
        tmp.iloc[:,2:] = tmp.iloc[:,2:].divide(nb_cell[0].values[0])
        for idx, ct in enumerate(celltypes):
            tmp2 = tmp[tmp.celltype == ct]
            tmp2.drop(["celltype","Sample_num"], inplace=True, axis=1)
            print(ct)
            if len(tmp2) == 0:
                array_agg_sc[su_idx,idx,:] = np.zeros(array_agg_sc[su_idx,idx,:].shape)
            else:
                array_agg_sc[su_idx,idx,:] = tmp2.values.reshape(-1)
    np.savez_compressed(path + "agg_sc_separate.npz", mat=array_agg_sc)
if __name__ == "__main__":
    main()
