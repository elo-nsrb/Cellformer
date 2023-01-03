
#!/usr/bin/env python
# coding: utf-8

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scanpy as sc
import anndata as ad
import warnings
def prepareData(path, annot_pa, model_path, disease):

    if disease == "PD":
        list_conds = ["CTRL", "LOPD"]
        List_brain_regions = ["CAUD", "MDFG", "PTMN", "HIPP"]
        order = ["CAUD","MDFG", "PTMN","HIPP"]
        model_path = model_path + "PD/"
    else:
        order = ["CAUD","SMTG","PARL","HIPP"]
        List_brain_regions = ["CAUD", "SMTG", "PARL", "HIPP"]
        list_conds = ["CTRL", "LOAD", "ADAD"]
        pairs = [("CTRL", "CTRH"), ("CTRL", "LOAD"), ("CTRH","LOAD")]#, ("CTRL", "ADAD")]

        model_path = model_path + "AD/"
    annotations = pd.read_csv(annot_pa)
    if annotations.isna().sum().sum() !=0:
                    annotations.fillna("Unknown",inplace=True)
    annot=annotations
    annot["Geneid"] = (annot["chrm"] + "." 
                      + annot["start"].astype("str")
                      + "."+annot["end"].astype("str"))

    list_peaks = [str(it) for it in range(annot.shape[0])]
    list_peaks_2 = [it for it in range(annot.shape[0])]

    df_pred_bulk = pd.read_csv(model_path 
                            + "All_samples_inference_with_ANNOT_not_clip.csv")

    list_predict = ["xxx.ch_lastCasiScore", 
                    "xxx.expired_age", "PMI",
                    "xxx.last_mmse_test_score",
                    "xxx.Cerad NP","xxx.Braak score" ,
                    "xxx.calc_NIA_AA","xxx.calc_B","xxx.C","xxx.calc_A",
                    "xxx.AP_freshBrainWeight","xxx.calc_thalPhase",
                    "xxx.GE_atherosclerosis_ID", "xxx.calc_B",
                    "xxx.micro_AmyloidAngiopathyOccipitalLobe_ID"]

    list_predict.append("binarize_braak")
    list_predict.append("label")
    list_predict.append("label_cerad")
    adata = ad.AnnData(df_pred_bulk[list_peaks])
    adata.obs["celltype"] = df_pred_bulk.celltype.values
    adata.obs["replicate"] = df_pred_bulk.replicate.values
    adata.obs["brain_region"] = df_pred_bulk.brain_region.values
    adata.obs["condition"] = df_pred_bulk.Type.values
    adata.obs["gender"] = df_pred_bulk.Gender.values
    adata.obs["batch"] = df_pred_bulk.Batch.values
    adata.obs["expired_age"] = df_pred_bulk['xxx.expired_age'].values
    adata.obs["xxx.CognitiveStatus"] = df_pred_bulk['xxx.CognitiveStatus'].values
    adata.obs["Rep"] = df_pred_bulk['xxx.TechRep.1'].values
    adata.obs["PatientID"] = df_pred_bulk['PatientID'].values

    for it in list_predict:
        adata.obs[it] = df_pred_bulk[it].values
    adata = adata[adata.obs["label"] !="CTRH"]
    adata = adata[adata.obs.condition !="ADAD"]
    adata.var["nearestGeneChip"] = annot["nearestGeneChip"].values
    adata.var["nearestGene"] = annot["nearestGene"].values
    adata.var["peakType"] = annot["peakType"].values
    #adata.var["GeneType"] = annot["GeneType"].values

    adata.var["peakid"] = annot["chrm"].astype(str).values + "_" + annot["start"].astype(str).values + "_" + annot["end"].astype(str).values
    adata.var["shortAnnotChip"] = annot["shortAnnotChip"].values
    print(annot.columns)
    adata.var["start"] = annot["start"].values
    adata = adata[(adata.obs.PatientID !="11_1686")]
    return adata


