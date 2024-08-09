import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, lines
import matplotlib.transforms as transforms
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
import statsmodels.api as sm
import itertools
from sklearn.feature_selection import SelectKBest, f_classif
import itertools
import scanpy as sc
import anndata as ad
from scipy.stats import ttest_ind, mannwhitneyu


import warnings

def prepareData(path, annot_pa, model_path_cv, disease, metadata):

    list_conds = ["CTRL", "LOPD", "GBA1", "LRRK2"]
    List_brain_regions = ["CAUD", "PTMN", "SMTG","MDFG", "HIPP"]
    order = ["CAUD", "PTMN", "SMTG","MDFG", "HIPP"]
    #model_path = model_path + "PD/"
    annotations = pd.read_csv(annot_pa)
    if annotations.isna().sum().sum() !=0:
                    annotations.fillna("Unknown",inplace=True)
    annot=annotations
    annot["Geneid"] = (annot["chrm"] + "." 
                      + annot["start"].astype("str")
                      + "."+annot["end"].astype("str"))

    list_peaks = [str(it) for it in range(annot.shape[0])]
    list_peaks_2 = [it for it in range(annot.shape[0])]
    df_pred_bulk = pd.read_csv(os.path.join(path,
                            "PD_All_samples_inference_with_ANNOT.csv"))

    list_predict = [it for it in df_pred_bulk.columns.tolist() if it not in list_peaks]
    df_pred = df_pred_bulk#[list_peaks +["PatientID","brain_region","celltype"]

    if model_path_cv is not None:
        celltype = ["AST","MIC", "NEU", "OPCs", "OLD"]
        df_metrics = pd.read_csv(os.path.join(model_path_cv, "metrics_all_per_genes.csv"))
        df_metrics["genes"] = df_metrics["genes"].astype(str)
        mat = df_metrics[df_metrics.metrics == "spearman"]
        mat = mat.pivot(index=["celltype", "fold"], 
                    columns=["genes"], values="res").fillna(0).reset_index().groupby("celltype")[
                                df_metrics.genes.unique().tolist()].mean()
        mat = mat.loc[celltype,:]
        mat[mat<=0.2] = 0
        mask = mat.values
        mask[mask>0] = 1
        np.save(os.path.join(model_path_cv, "Mask_cv_0.2.npy"), mask)
        for ii,ct in enumerate(celltype):
            df_pred.loc[(df_pred.celltype==ct), list_peaks] = df_pred.loc[(df_pred.celltype==ct), list_peaks].astype(float)*mask[ii,:, np.newaxis].T

    adata = ad.AnnData(df_pred[list_peaks])
    adata.obs["celltype"] = df_pred.celltype.values
    adata.obs["replicate"] = df_pred.replicate.values
    adata.obs["brain_region"] = df_pred.brain_region.values
    adata.obs["condition"] = df_pred.Type.values
    adata.obs["Sex"] = df_pred.Gender.values
    adata.obs["batch"] = df_pred.Batch.values
    #adata.obs["Rep"] = df_pred_bulk['xxx.TechRep.1'].values
    adata.obs["PatientID"] = df_pred['PatientID'].values

    for it in list_predict:
        adata.obs[it] = df_pred[it].values
    adata.var["nearestGeneChip"] = annot["nearestGeneChip"].values
    adata.var["nearestGene"] = annot["nearestGene"].values
    adata.var["peakType"] = annot["peakType"].values

    adata.var["peakid"] = annot["chrm"].astype(str).values + "_" + annot["start"].astype(str).values + "_" + annot["end"].astype(str).values
    adata.var["shortAnnotChip"] = annot["shortAnnotChip"].values
    print(annot.columns)
    adata.var["start"] = annot["start"].values
    adata.obs = adata.obs.loc[:, (adata.obs.isna().sum()<2000)]
    adata.obs.rename({"xxx.expired_age":"age at death"},axis=1, inplace=True)
    adata = adata[adata.obs["age at death"]>60]
    adata.obs.rename({"xxx.expired_age":"age at death"},axis=1, inplace=True)
    adata.obs["condition"] = adata.obs["condition"].replace({"LRRK": "LRRK2"}).values
    adata.obs["group"] = adata.obs["condition"].values
    adata.obs["condition"] = adata.obs["condition"].astype(str)
    adata.obs.loc[(adata.obs["xxx.PD"]=="yes")&(adata.obs["condition"]=="GBA1"), "condition"] = "GBA1_PD+"
    adata.obs.loc[(adata.obs["xxx.PD"]=="no")&(adata.obs["condition"]=="GBA1"), "condition"] = "GBA1_PD-"
    adata.obs.loc[(adata.obs["xxx.PD"]=="yes")&(adata.obs["condition"]=="LRRK2"), "condition"] = "LRRK2_PD+"
    adata.obs.loc[(adata.obs["xxx.PD"]=="no")&(adata.obs["condition"]=="LRRK2"), "condition"] = "LRRK2_PD-"
    adata.obs["GBA1_PD+"] = adata.obs["condition"].map({"SPOR":0, "GBA1_PD+":1,"GBA1_PD-":0,  "LRRK2_PD+":0, "LRRK2_PD-": 0, "CTRL":0})
    adata.obs["GBA1_PD-"] = adata.obs["condition"].map({"SPOR":0, "GBA1_PD+":0,"GBA1_PD-":1,  "LRRK2_PD+":0, "LRRK2_PD-": 0, "CTRL":0})
    adata.obs["LRRK2_PD+"] = adata.obs["condition"].map({"SPOR":0, "GBA1_PD+":0,"GBA1_PD-":0,  "LRRK2_PD+":1, "LRRK2_PD-": 0, "CTRL":0})
    adata.obs["LRRK2_PD-"] = adata.obs["condition"].map({"SPOR":0, "GBA1_PD+":0,"GBA1_PD-":0,  "LRRK2_PD+":0, "LRRK2_PD-": 1, "CTRL":0})


    adata.obs["Neuritic plaque density"] = adata.obs['xxx.Plaque density'].map({"zero":0,"sparse":1,"moderate":2,"frequent":3}).values
    adata.obs["Braak score NFT"] = adata.obs['xxx.Braak score'].map({"0":0,"I":1,"II":2,"III":3,"IV":4, "V":5,"VI":6}).values
    adata.obs["Unified LB Stage"] = adata.obs['xxx.Unified LB Stage'].map({"0. No Lewy bodies":0,"lV. Neocortical":1,"lll. Brainstem/Limbic":2,
                                                                               "lla. Brainstem Predominant":2,"llb. Limbic Predominant":2}).values
    adata.obs["Neocortical LB"] = adata.obs['xxx.Unified LB Stage'].map({"0. No Lewy bodies":0,"lV. Neocortical":1,
                                                                               "lll. Brainstem/Limbic":0,"lla. Brainstem Predominant":0,"llb. Limbic Predominant":0}).values
    adata.obs["Brainstem/Limbic LB"] = adata.obs['xxx.Unified LB Stage'].map({"0. No Lewy bodies":0,"lV. Neocortical":0,
                                                                                    "lll. Brainstem/Limbic":1,"lla. Brainstem Predominant":1,"llb. Limbic Predominant":1}).values

    adata.obs["ApoE_4"] = adata.obs["ApoE"].map({"2_2":0,"2_3":0,"3_3":0,"3_4":1,"4_4":1}).values
    adata.obs["ApoE_2"] = adata.obs["ApoE"].map({"2_2":1,"2_3":1,"3_3":0,"3_4":0,"4_4":0}).values
    adata.obs["SPOR"] = adata.obs["condition"].map({"SPOR":1, "GBA1_PD+":0, "GBA1_PD-":0, "LRRK2_PD+":0,  "LRRK2_PD-":0,"CTRL":0})
    adata.obs["CTRL"] = adata.obs["condition"].map({"SPOR":0,  "GBA1_PD+":0, "GBA1_PD-":0, "LRRK2_PD+":0,  "LRRK2_PD-":0,"CTRL":1})
    adata.obs["GBA1"] = adata.obs["condition"].map({"SPOR":0, "GBA1_PD+":1, "GBA1_PD-":1, "LRRK2_PD+":0,  "LRRK2_PD-":0, "CTRL":0})
    adata.obs["LRRK2"] = adata.obs["condition"].map({"SPOR":0,"GBA1_PD+":0, "GBA1_PD-":0, "LRRK2_PD+":1,  "LRRK2_PD-":1,"CTRL":0})
    adata.obs["GBA1_PD+"] = adata.obs["condition"].map({"SPOR":0, "GBA1_PD+":1, "GBA1_PD-":0, "LRRK2_PD+":0,  "LRRK2_PD-":0, "CTRL":0})


    adata.obs["Neuritic plaque density binary"] = adata.obs['xxx.Plaque density'].map({"zero":0,"sparse":0,
                                                                           "moderate":1,"frequent":1}).values

    adata.obs["Braak score NFT binary"] = adata.obs['xxx.Braak score'].map({"0":0,"I":0,"II":0,"III":1, 
                                                                            "IV":1, "V":1,"VI":1}).values

    adata.obs["AD path"] = adata.obs["Braak score NFT"].astype(int).apply(lambda x: "low/no" if x<4 else "moderate/high").values
    adata.obs["PD clinical"] = adata.obs["xxx.PD"]
    adata.obs["Cognitive Imp"] = adata.obs["xxx.AD"].values
    adata.obs.loc[adata.obs["xxx.MCI"] == "yes", "Cognitive Imp"] = "yes"
    adata.obs.loc[adata.obs["xxx.dementia_nos"] == "yes","Cognitive Imp"] = "yes"
    adata.obs["PD path"] = adata.obs["Unified LB Stage"].values#.astype(int).apply(lambda x: "low/no" if x<1 else "moderate/high").values


    return adata

def main():
    path = "/home/eloiseb/experiments/deconv_peak/ct_5_train_all/exp_0/bulk/bulk_sample_decon/maskedPD/"
    annot_pa = "/home/eloiseb/data/ATAC-seq_2024/mergeAnnot.csv"
    disease = "PD"
    model_path_cv ="/home/eloiseb/experiments/deconv_peak/ct_5_2024/"

    metadata = pd.read_excel("/home/eloiseb/stanford_drive/data/ATAC-seq/ATAC-seq_raw_data/190215_Brain-All_Metadata_Merged.xlsx")
    adata = prepareData(path, annot_pa, model_path_cv, disease,metadata)
    __import__('ipdb').set_trace()
    adata.write("/home/eloiseb/data/ATAC-seq_2024/adata_deconvolution_PD_2024_trained_all_filtered_feb24.h5ad", compression="gzip")
    sc.pp.log1p(adata)
    adata.write("/home/eloiseb/data/ATAC-seq_2024/adata_deconvolution_PD_2024_trained_all_filtered_feb24_lognorm.h5ad", compression="gzip")
if __name__ == "__main__":
    main()
