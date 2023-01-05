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
import episcanpy.api as epi
import random
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import itertools
from statannotations.Annotator import Annotator
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', 
                default="./data/",
                    help='Location to save pseudobulks data')
parser.add_argument('--model_results', default="./cellformer/bulk/bulk_sample_decon/masked/",
                    help='Location to save pseudobulks data')
parser.add_argument('--annotation_path', default="./data/mergeAnnotwithEnhancer.csv",
                    help='Path to annotation file')
parser.add_argument('--metadata_path', default="./data/ATAC-seq/190215_Brain-All_Metadata_Merged.xlsx",
                    help='Path to metadata file')
parser.add_argument('--disease', default="AD", help="AD or PD")


np.random.seed(10)


def gatherResults(bulk_path,
                  model_path,
                  savepath,
                  disease,
                  annot,
                  metadata,
                  list_peaks,
                  List_brain_regions,
                  list_conds_ini):
    all_samples = []
    all_mixtures = []
    for brain_region in List_brain_regions:
        for condition in list_conds_ini:
            idi = condition + "_" + brain_region + "_" + disease
            mixture_path = bulk_path + disease + "/" + disease + "/" + condition + "/" + idi + ".peak_countMatrix.txt" 

            if os.path.exists(model_path + idi + ".csv"):
                pred_bulk = pd.read_csv(model_path + idi + ".csv")
                pred_bulk.columns = ["Unnamed: 0"] + list_peaks + ["celltype"]
                mixture = pd.read_csv(mixture_path, sep="\t" ,header=1)
        
                mixture = mixture[mixture.Geneid.isin(annot.Geneid)]

                mix = mixture.iloc[:, 6:].T.copy()
                mix.columns = list_peaks

                pred_bulk["replicate"] = pred_bulk["Unnamed: 0"].str.split("_B1_T",expand=True)[1].str.split("_", expand=True)[0]
                pred_bulk["Sample_id"] = pred_bulk["Unnamed: 0"].str.split("L0|L1",expand=True)[0].str.split("/", expand=True)[7]
                pred_bulk["brain_region"] = brain_region
                pred_bulk["condition"] = condition
                mix["brain_region"] = brain_region
                mix["condition"] = condition
                mix["Unnamed: 0"] = mix.reset_index()["index"].values
                mix["replicate"] = mix["Unnamed: 0"].str.split("_B1_T",expand=True)[1].str.split("_", expand=True)[0]
                mix["Sample_id"] = mix["Unnamed: 0"].str.split("L0|L1",expand=True)[0].str.split("/", expand=True)[7]
                mix["celltype"] ="bulk"
                
                all_samples.append(pred_bulk)
                all_samples.append(mix)
    df_all_samples = pd.concat(all_samples, axis=0)
    df_all_samples["Name"] = df_all_samples.reset_index()["Unnamed: 0"].str.split("/", expand=True)[7].str.split(".", expand=True)[0].tolist()
    metadata = metadata[metadata.HarmonizedName.isin(df_all_samples.Name.tolist())]
    metadata["Name"] = metadata["HarmonizedName"].tolist()
    df_all_samples = df_all_samples.join(metadata.set_index("Name"), on="Name")
    df_all_samples["condition"] = df_all_samples["Type"].values
    celltypes = df_all_samples.celltype.unique()
    for ct in celltypes:
        if ct != "bulk":
            df_all_samples.loc[
                df_all_samples.celltype ==ct, 
                list_peaks] = (df_all_samples[
                                df_all_samples.celltype ==ct][
                                    list_peaks].clip(lower=0)).astype(int)
    df_all_samples = df_all_samples[df_all_samples.condition !="ADAD"]
    df_all_samples = df_all_samples.loc[:,~(df_all_samples.isna().sum()>1500)]


    list_predict = ["xxx.ch_lastCasiScore", 
                "xxx.expired_age", "PMI",
                "xxx.last_mmse_test_score", "ApoE",
                "xxx.Cerad NP","xxx.Braak score" ,
                "xxx.calc_NIA_AA","xxx.calc_B","xxx.C","xxx.calc_A",
                "xxx.AP_freshBrainWeight","xxx.calc_thalPhase",
                "xxx.GE_atherosclerosis_ID", "xxx.calc_B",
                "xxx.micro_AmyloidAngiopathyOccipitalLobe_ID"]

    df_all_samples.loc[df_all_samples.loc[:,'ApoE'].isna(), "ApoE"] = "Unk"
    braak_map = {'0':0, 'II':2, 'IV':4, 'III':3, 'V':5, 'VI':6, 'I':1}
    df_all_samples["xxx.Braak score"] = df_all_samples["xxx.Braak score"].map(braak_map) 
    df_all_samples["binarize_braak"] = (df_all_samples["xxx.Braak score"]<3).astype(int)
    list_predict.append("binarize_braak")

    cerad_map = {"No neuritic plaques (C0)":0, "Moderate (C2)":2,
    "Frequent (C3)":3}
    df_all_samples["xxx.Cerad NP"] = df_all_samples["xxx.Cerad NP"].map(cerad_map)
    df_all_samples["binarize_cerad_np"] = (df_all_samples["xxx.Cerad NP"]<1).astype(int)
    list_predict.append("binarize_cerad_np")
    df_all_samples["Pathology_B"] = df_all_samples["binarize_braak"].values
    mapp_path = {1:"low", 0:"high"}
    df_all_samples["Pathology_B"] = df_all_samples["Pathology_B"].apply(lambda x : mapp_path[x])

    df_all_samples["label"] = df_all_samples["condition"].values
    df_all_samples.loc[(df_all_samples['xxx.CognitiveStatus'].str.contains("No") )& (df_all_samples["Pathology_B"]=="high"), "label"] ="RAD"
    df_all_samples.loc[(df_all_samples['xxx.CognitiveStatus']=="Dementia") & (df_all_samples["Pathology_B"]=="high"), "label"] ="ADD"
    df_all_samples.loc[(df_all_samples['xxx.CognitiveStatus']=="No dementia") & (df_all_samples["Pathology_B"]=="low"), "label"] ="NC"
    df_all_samples.loc[((df_all_samples.label =="RAD") &(df_all_samples["xxx.Cerad NP"] == 0)),"label"] = "Na"

    list_predict.append("label")

    df_all_samples["label_cerad"] = df_all_samples["condition"].values
    df_all_samples.loc[(df_all_samples['xxx.CognitiveStatus'].str.contains("No") )& (df_all_samples["binarize_cerad_np"]==0), "label_cerad"] ="RAD"
    df_all_samples.loc[(df_all_samples['xxx.CognitiveStatus']=="Dementia") & (df_all_samples["binarize_cerad_np"]==0), "label_cerad"] ="ADD"
    df_all_samples.loc[(df_all_samples['xxx.CognitiveStatus']=="No dementia") & (df_all_samples["binarize_cerad_np"]==1), "label_cerad"] ="NC"
    list_predict.append("label_cerad")

    df_all_samples.to_csv(savepath + "/All_samples_inference_with_ANNOT_not_clip.csv", index=0)
    return df_all_samples

def concat_metadata(df, metadata):
    metadata["sample_id"] = metadata["Name"].str.split("L0|L1",expand=True)[0]
    metadata = metadata[metadata["sample_id"].isin(df["sample_id"].tolist())]
    metadata = metadata[metadata.columns[metadata.isna().sum(0)==0]]
    df = df.join(metadata.set_index("sample_id"), on="sample_id")
    return df

def randomBioreplicatesTesting(df_all_samples,
                               bulk_path,
                               metadata,
                               disease,
                               annot,
                              list_peaks, 
                              List_brain_regions,
                              list_conds_ini):
    
    tmp = df_all_samples.groupby(["Sample_id", "celltype"]).replicate.size()
    list_non_replicate = tmp[tmp==1].reset_index()["Sample_id"].tolist()
    print(df_all_samples[df_all_samples.Sample_id.isin(list_non_replicate)].groupby("celltype").size())
    df_all_sample_filt = df_all_samples[~df_all_samples.Sample_id.isin(list_non_replicate)]

    df_corr_pseudo_pred_bulk_bulk = pd.DataFrame(columns=["sample_id","Brain_region", "condition",
                                                      "spearman_corr", "pearson_corr", "mse"])
    df_corr_pseudo_pred_bulk_bulk_random = pd.DataFrame(
                        columns=["sample_id","Brain_region",
                            "condition", 
                             "spearman_corr",
                             "pearson_corr",
                             "mse"])


    df_corr_pseudo_pred_replicates_bulk = pd.DataFrame(columns=["sample_id", 
                                                    "Brain_region", 
                                                    "condition",
                                                    "celltype", 
                                                    "spearman_correlation"])
    df_corr_pseudo_pred_replicates_bulk_random = pd.DataFrame(
                                                        columns=["sample_id", 
                                                       "Brain_region", 
                                                       "condition",
                                                       "celltype", 
                                                       "spearman_correlation"])
    df_corr_pseudo_pred_replicates_bulk_random_all_br = pd.DataFrame(
                                                    columns=["sample_id", 
                                                          "Brain_region", 
                                                          "condition",
                                                           "celltype", 
                                                          "spearman_correlation"])
    df_corr_pseudo_pred_replicates_bulk_random_all_condition = pd.DataFrame(columns=["sample_id", 
                                                                                     "Brain_region", 
                                                                                     "condition",
                                                                                     "celltype",
                                                                                     "spearman_correlation"])
    for brain_region in List_brain_regions:
        for condition in list_conds_ini:
            idi = condition + "_" + brain_region + "_" + disease
            mixture_path = bulk_path + disease + "/" + disease + "/" +  condition + "/" + idi + ".peak_countMatrix.txt" 

            pred_bulk = df_all_sample_filt[
                    (df_all_sample_filt.brain_region ==brain_region)
                              & (df_all_sample_filt.condition ==condition )]
            mixture = pd.read_csv(mixture_path, sep="\t" ,header=1)


            pseudo_1 = pred_bulk[pred_bulk.replicate =="1"].drop(["replicate","Unnamed: 0"], 
                                                                 axis=1).set_index(["Sample_id", 
                                                                                    "celltype"])

    
            pseudo_2 = pred_bulk[pred_bulk.replicate =="2"].drop(["replicate",
                                                                  "Unnamed: 0"], axis=1).set_index(["Sample_id",
                                                                                                    "celltype"])

            cor_repl = pseudo_1.corrwith(pseudo_2, axis=1, method="spearman")

            pseudo_2_random = pred_bulk[pred_bulk.replicate =="2"].drop(["replicate",
                                                                         "Unnamed: 0" ], axis=1).copy(deep=True)

            sufffle_id, shuffle_celltype = shuffle(pseudo_2_random["Sample_id"].values.tolist(),
                             pseudo_2_random["celltype"].values.tolist(),
                                                  random_state=0)
            ll = sufffle_id
            pseudo_2_random["Sample_id"] = sufffle_id
            pseudo_2_random["celltype"] = shuffle_celltype
            pseudo_2_random = pseudo_2_random.set_index(["Sample_id", "celltype"])
            cor_repl_random = pseudo_1.corrwith(pseudo_2_random, 
                                                axis=1, 
                                                method="spearman")

            pseudo_2_random_2 = df_all_sample_filt[
                                (df_all_sample_filt.condition ==condition ) 
                                & (df_all_sample_filt.replicate =="2")
                                ].drop(["replicate","Unnamed: 0" ],
                                        axis=1).copy(deep=True)
            sufffle_id, shuffle_celltype,shuffle_br = shuffle(
                                    pseudo_2_random_2["Sample_id"].values.tolist(),
                                   pseudo_2_random_2["celltype"].values.tolist(),
                                   pseudo_2_random_2["brain_region"].values.tolist(),
                                  random_state=0)
            pseudo_2_random_2["Sample_id"] = sufffle_id
            pseudo_2_random_2["celltype"] = shuffle_celltype
            pseudo_2_random_2["brain_region"] = shuffle_br
            pseudo_2_random_2 =pseudo_2_random_2[pseudo_2_random_2.brain_region == brain_region]
            pseudo_2_random_2 = pseudo_2_random_2.set_index(["Sample_id", "celltype"])
            cor_repl_random_2 = pseudo_1.corrwith(pseudo_2_random_2, axis=1, method="spearman")

            pseudo_2_random_3 = df_all_sample_filt[
                        (df_all_sample_filt.brain_region ==brain_region ) 
                    & (df_all_sample_filt.replicate =="2")].drop(
                                    ["replicate","Unnamed: 0" ],
                                    axis=1).copy(deep=True)
            sufffle_id, shuffle_celltype,shuffle_condition= shuffle(
                                pseudo_2_random_3["Sample_id"].values.tolist(),
                                                   pseudo_2_random_3["celltype"].values.tolist(),
                                                   pseudo_2_random_3["condition"].values.tolist(),
                                                  random_state=0)
            pseudo_2_random_3["Sample_id"] = sufffle_id
            pseudo_2_random_3["celltype"] = shuffle_celltype
            pseudo_2_random_3["condition"] = shuffle_condition
            pseudo_2_random_3 =pseudo_2_random_3[pseudo_2_random_3.condition == condition]
            pseudo_2_random_3 = pseudo_2_random_3.set_index(["Sample_id", "celltype"])
            cor_repl_random_3 = pseudo_1.corrwith(pseudo_2_random_3, axis=1, method="spearman")

            hh = cor_repl.dropna().reset_index()#.groupby(["celltype"]).mean()
            hh_random = cor_repl_random.dropna().reset_index()
            hh_random_2 = cor_repl_random_2.dropna().reset_index()
            hh_random_3 = cor_repl_random_3.dropna().reset_index()

            hh.columns = ["Sample_id", "celltype", "sp"]
            hh_random.columns = ["Sample_id", "celltype", "sp"]
            hh_random_2.columns = ["Sample_id", "celltype", "sp"]
            hh_random_3.columns = ["Sample_id", "celltype", "sp"]


            for it in hh.Sample_id.unique():
                for cl in hh.celltype.unique():
                    tmp = hh[(hh.Sample_id == it) & (hh.celltype ==cl)]
                    tmp_random = hh_random[(hh_random.Sample_id == it) & (hh_random.celltype ==cl)]
                    tmp_random_2 = hh_random_2[(hh_random_2.Sample_id == it) & (hh_random_2.celltype ==cl)]
                    tmp_random_3 = hh_random_3[(hh_random_3.Sample_id == it) & (hh_random_3.celltype ==cl)]

                    #print(tmp.loc[:,"sp"].values[0])
                    #print(tmp_random)
                    df_corr_pseudo_pred_replicates_bulk.loc[
                        len(df_corr_pseudo_pred_replicates_bulk),:] = [it,
                                                brain_region,
                                                condition,
                                                cl, 
                                                tmp.loc[:,"sp"].values[0]]
                    df_corr_pseudo_pred_replicates_bulk_random.loc[
                        len(df_corr_pseudo_pred_replicates_bulk_random),
                                                                    :] = [it,
                                                            brain_region, 
                                                            condition, cl, 
                                                            tmp_random.loc[:,
                                                                "sp"].values[0]]

                    df_corr_pseudo_pred_replicates_bulk_random_all_br.loc[
                                            len(df_corr_pseudo_pred_replicates_bulk_random_all_br),
                                            :] = [it, 
                                                  brain_region,
                                                  condition, cl, 
                                                  tmp_random_2.loc[:,"sp"].values[0]]
                    df_corr_pseudo_pred_replicates_bulk_random_all_condition.loc[
                                            len(df_corr_pseudo_pred_replicates_bulk_random_all_condition),
                                            :] = [it, 
                                                brain_region,
                                                condition, cl, 
                                                tmp_random_3.loc[:,"sp"].values[0]]


            list_file = mixture.columns[6:]
            list_new_file = [it.split("/")[-1].split(".")[0] for it in list_file]
            gg = pred_bulk.loc[:,["Unnamed: 0"]+list_peaks ].groupby(["Unnamed: 0"]).sum()
            spear_l = []
            pear_l = []
            spear_l_random = []
            pear_l_random = []
            for it, ct in enumerate(gg.index.tolist()):
                #print(ct)
                name = ct.split("/")[-1].split(".")[0]
                x = gg.loc[ct,:]
                x = x/x.max()
                y = mixture[ct].astype("float32")/50000.
                y = y/y.max()



                mse = mean_squared_error(x, y)
                #print("RMSE : %f"%(mse))

            sp, pval = stats.spearmanr(x, y)
            spear_l.append(sp)
            #print("spearman, pval: %f"%(sp))
            r, pval = stats.pearsonr(x, y)
            pear_l.append(r)

            df_corr_pseudo_pred_bulk_bulk.loc[len(df_corr_pseudo_pred_bulk_bulk),
                                              :] = [ct, 
                                                    brain_region, 
                                                    condition, 
                                                    sp, r, 
                                                    mse]
            np.random.shuffle(np.array(y, copy=True))
            y_random = shuffle(y)
            mse_random = mean_squared_error(x, y_random)
            #print("RMSE : %f"%(mse))
            sp_random, pval_random = stats.spearmanr(x, y_random)
            spear_l_random.append(sp_random)
            #print("spearman, pval: %f"%(sp))
            r_random, pval_random = stats.pearsonr(x, y_random)
            pear_l_random.append(r_random)

            df_corr_pseudo_pred_bulk_bulk_random.loc[
                                        len(df_corr_pseudo_pred_bulk_bulk_random),
                                                    :] = [ct, 
                                                          brain_region, 
                                                          condition,
                                                          sp_random, 
                                                          r_random, 
                                                          mse_random]
    df_corr_pseudo_pred_bulk_bulk["Name"] = df_corr_pseudo_pred_bulk_bulk.reset_index()["sample_id"].str.split("/",
                                                                                                               expand=True)[7].str.split(".", 
                                                                                                                                         expand=True)[0].tolist()
    df_corr_pseudo_pred_bulk_bulk_random["Name"] = df_corr_pseudo_pred_bulk_bulk_random.reset_index()["sample_id"].str.split("/",
                                                                                                                             expand=True)[7].str.split(".", 
                                                                                                                                         expand=True)[0].tolist()

    metadata = metadata[metadata.HarmonizedName.isin(df_corr_pseudo_pred_bulk_bulk.Name.tolist())]
    metadata["Name"] = metadata["HarmonizedName"].tolist()
    metadata = metadata[metadata.columns[metadata.isna().sum(0)==0]]
    df_corr_pseudo_pred_bulk_bulk = df_corr_pseudo_pred_bulk_bulk.join(metadata.set_index("Name"), 
                                                                       on="Name", 
                                                                       lsuffix="meta")
    metadata = metadata[metadata.HarmonizedName.isin(df_corr_pseudo_pred_bulk_bulk_random.Name.tolist())]
    metadata["Name"] = metadata["HarmonizedName"].tolist()
    metadata = metadata[metadata.columns[metadata.isna().sum(0)==0]]
    df_corr_pseudo_pred_bulk_bulk_random = df_corr_pseudo_pred_bulk_bulk_random.join(metadata.set_index("Name"), 
                                                                                     on="Name",
                                                                                     lsuffix="meta")
    #df_corr_pseudo_pred_bulk_bulk

    df_corr_pseudo_pred_bulk_bulk_random["condition"] = df_corr_pseudo_pred_bulk_bulk_random.Type.values
    df_corr_pseudo_pred_bulk_bulk["condition"] = df_corr_pseudo_pred_bulk_bulk.Type.values

    df_corr_pseudo_pred_bulk_bulk["type"] = "normal"
    df_corr_pseudo_pred_bulk_bulk_random["type"]= "random"
    df_bulk_bulk = pd.concat([df_corr_pseudo_pred_bulk_bulk.loc[:,~df_corr_pseudo_pred_bulk_bulk.columns.duplicated()], 
                                                  df_corr_pseudo_pred_bulk_bulk_random], 
                                                     axis=0,ignore_index=True, 
                                                     keys=["normal","random"])

    df_corr_pseudo_pred_replicates_bulk["typeTest"] = "normal"
    df_corr_pseudo_pred_replicates_bulk_random["typeTest"]= "SameConditionSameBR"
    df_corr_pseudo_pred_replicates_bulk_random_all_br["typeTest"]= "randomAcrossBRSameCondition"
    df_corr_pseudo_pred_replicates_bulk_random_all_condition["typeTest"]= "randomAcrossConditionSameBR"

    df_replicates_bulk = pd.concat([df_corr_pseudo_pred_replicates_bulk,
                            df_corr_pseudo_pred_replicates_bulk_random,
                            df_corr_pseudo_pred_replicates_bulk_random_all_br,
                           df_corr_pseudo_pred_replicates_bulk_random_all_condition], axis=0)

    return df_replicates_bulk, df_bulk_bulk


def plotRandomnTestComparison(df_replicates_bulk,
                              disease,
                              model_path,
                              order,
                              format_to_save=".svg"):

    conditions = df_replicates_bulk.condition.unique().tolist()
    ct = df_replicates_bulk.celltype.unique().tolist()
    fig, ax = plt.subplots(len(ct),len(conditions), figsize=(26,25))
    ax= ax.flatten()
    k=0
    hue_order=['normal', "SameConditionSameBR",
                    "randomAcrossBRSameCondition",
                    "randomAcrossConditionSameBR"]
    colors_h= { 'normal':"#7310A8", "SameConditionSameBR":"#EEF7C8", 
                "randomAcrossBRSameCondition":"#D7F564",
                "randomAcrossConditionSameBR":"#87A805"}
    sns.set(font_scale=2, style="white")
    for j, cc in enumerate(ct):
        for i, cond in enumerate(conditions):
            idi = str(cond) + "_" + str(cc) + "_" + disease
            print(idi)
            df_ = df_replicates_bulk[(df_replicates_bulk["condition"]==cond)
                                   &(df_replicates_bulk["celltype"]==cc)]
            pair_com = list(itertools.combinations(hue_order,2))
            pair_com = [("normal", tt1) for tt1 in ["SameConditionSameBR",
                        "randomAcrossBRSameCondition",
                        "randomAcrossConditionSameBR"]]
            if cond=="ADAD":
                box_pair = [((bb, tt1), (bb, tt2)) for bb in 
                            order  for tt1,tt2 in pair_com if bb!="HIPP"]
            else:
                box_pair = [((bb, tt1), (bb, tt2)) for bb in
                            order  for tt1,tt2 in pair_com]
            print(box_pair)
            sns.boxplot(data=df_, x="Brain_region",
                        y="spearman_correlation", 
                        hue="typeTest", dodge=True,
                        palette=colors_h, ax=ax[k], 
                        order=order, 
                       hue_order=hue_order, linewidth=0.5)
            annotator = Annotator(ax[k], box_pair, data=df_,
                                x="Brain_region",
                                y="spearman_correlation",
                                hue="typeTest",order=order,
                                     hue_order=hue_order)
            annotator.configure(test='t-test_ind',  text_format="star", 
                                loc='inside', fontsize="8", 
                                comparisons_correction="BH")
            annotator.apply_and_annotate()
            ax[k].set_title(cond + "_"+cc)
            ax[k].set_xlabel("")

            if k<len(ct)*len(conditions)-1:
                ax[k].get_legend().remove()
            k+=1
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(model_path + idi + "_correlation_bbetween_bulks_replicates_predictions_per_ba_norm" + format_to_save, bbox_inches="tight")

    plt.show()
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8,8))
    hue_order=['normal', "SameConditionSameBR",
                    "randomAcrossBRSameCondition",
                    "randomAcrossConditionSameBR"]
    colors_h= { 'normal':"#7310A8", "SameConditionSameBR":"#EEF7C8", 
                "randomAcrossBRSameCondition":"#D7F564",
                "randomAcrossConditionSameBR":"#87A805"}
    df_ = df_replicates_bulk
    pair_com = list(itertools.combinations(hue_order,2))
    pair_com = [("normal", tt1) for tt1 in ["SameConditionSameBR",
                "randomAcrossBRSameCondition",
                "randomAcrossConditionSameBR"]]
    if cond=="ADAD":
        box_pair = [((bb, tt1), (bb, tt2)) for bb in 
                    order  for tt1,tt2 in pair_com if bb!="HIPP"]
    else:
        box_pair = [((bb, tt1), (bb, tt2)) for bb in
                    order  for tt1,tt2 in pair_com]
    print(box_pair)
    sns.boxplot(data=df_, x="Brain_region",
                y="spearman_correlation", 
                hue="typeTest", dodge=True,
                palette=colors_h, ax=ax, 
                order=order, 
               hue_order=hue_order, linewidth=0.5)
    annotator = Annotator(ax, box_pair, data=df_,
                        x="Brain_region",
                        y="spearman_correlation",
                        hue="typeTest",order=order,
                             hue_order=hue_order)
    annotator.configure(test='t-test_ind',  text_format="star", 
                        loc='inside', fontsize="8", 
                        comparisons_correction="BH")
    annotator.apply_and_annotate()
    ax.set_xlabel("")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(model_path + idi + "_correlation_between_bulks_replicates_predictions_ALL_norm" + format_to_save, bbox_inches="tight")

    plt.show()

def plotComparisonCorrelation(df_replicates_bulk,
                                model_path,
                                disease,
                                order):
    colors_ct={"Neur":"#75485E", 
                "INH":"#75485E",
            "EXC":"#C23E7E",
                "Glials":"#51A3A3",
                "AST-OPCs-OLD":"#51A3A3", 
                "OPCs-Oligo":"#51A3A3", 
                "OLD":"#51A3A3", 
                "OPCs":"#C0F0F0", 
                "MIC":"#CB904D", 
                "AST":"#C3E991",
           "bulk":"#cccccc", "CTRH":"#B6F2EC", "SPOR":"#F7D7C8"}##DFCC74

    df_corr_pseudo_pred_replicates_bulk = df_replicates_bulk[df_replicates_bulk["typeTest"] == "normal"]
    pair_com = list(itertools.combinations(df_corr_pseudo_pred_replicates_bulk.celltype.unique().tolist(),2))
    box_pair = [((bb, tt1), (bb, tt2)) for bb in order  for tt1,tt2 in pair_com ]
    sns.set(font_scale=2, style="white")
    fig, ax = plt.subplots(figsize=(18,12))
    print(pair_com)
    box_pair = [((bb, tt1), (bb, tt2)) for bb in order  for tt1,tt2 in pair_com ]

    sns.boxplot(ax=ax, data=df_corr_pseudo_pred_replicates_bulk,
                    x="Brain_region",
                    y="spearman_correlation", 
                        hue="celltype", dodge=True,
                        palette=colors_ct,
                        linewidth=0.5
                        )
    annotator = Annotator(ax,box_pair,
                    data=df_corr_pseudo_pred_replicates_bulk,
                    x="Brain_region", y="spearman_correlation", 
                        hue="celltype")
    annotator.configure(test='t-test_ind',  text_format="star", loc='inside', fontsize="8", comparisons_correction="BH")
    annotator.apply_and_annotate()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig(model_path + disease + "_correlation_bbetween_replicates_predictions_COMP" + ".svg", bbox_inches="tight")
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12,8))
    sns.boxplot(ax=ax, data=df_corr_pseudo_pred_replicates_bulk,
                    x="celltype",
                    y="spearman_correlation", 
                        hue="celltype", dodge=True,
                        palette=colors_ct,
                        linewidth=0.5
                        )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(model_path + disease + "_correlation_between_replicates_ALL_regions" + ".svg", bbox_inches="tight")



def plotCorrelationBetweenRandomPeakOutput(df_bulk_bulk, 
                                        model_path, 
                                        disease):
    conditions = df_bulk_bulk.condition.unique().tolist()
    fig, axes = plt.subplots(1,len(conditions), figsize=(15,6))
    if len(conditions)<=1:
        ax = axes
    else:
        ax = axes.flatten()
    k=0
    if True:
        for i, cond in enumerate(conditions):
            df_ = df_bulk_bulk[(df_bulk_bulk["condition"]==cond)]
            sns.stripplot(data=df_, x="Brain_region", 
                            y="spearman_corr",
                            hue="type", dodge=True, 
                            palette="Set2", ax=ax[i])
            ax[i].set_title(cond)
            ax[i].set_xlabel("")
            k+=1
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(model_path + disease + "_correlation_bbetween_bulks_bulk_predictions_per_ba" + ".svg", bbox_inches="tight")

    plt.show()
    
def plotOutputPseudoBulkInputComparison(df_bulk_bulk, model_path, disease):
    df_corr_pseudo_pred_bulk_bulk = df_bulk_bulk[df_bulk_bulk["type"] == "normal"]
    colors={"CTRL":"#28A89C", "LOAD":"#F57764", "LOPD":"#F57764", "CTRH":"#B6F2EC", "SPOR":"#F7D7C8", "ADAD":"#A84434"}
    sns.boxplot(data=df_corr_pseudo_pred_bulk_bulk, x="Brain_region", y="mse", hue="condition", palette=[".8", "0.8"], linewidth=0.3)

    sns.swarmplot(data=df_corr_pseudo_pred_bulk_bulk, x="Brain_region", y="mse", hue="condition", dodge=True, palette=colors, size=4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(model_path + disease + "_mse_bbetween_predictions_and_bulk_per_ba.png", bbox_inches="tight")
    plt.close("all")
    colors={"CTRL":"#28A89C", "LOAD":"#F57764", "LOPD":"#F57764", "CTRH":"#B6F2EC", "SPOR":"#F7D7C8","ADAD":"#A84434"}

    sns.boxplot(data=df_corr_pseudo_pred_bulk_bulk, x="Brain_region", y="spearman_corr",hue="condition", palette=[".8", "0.8"], linewidth=0.3)

    sns.swarmplot(data=df_corr_pseudo_pred_bulk_bulk, x="Brain_region", y="spearman_corr", hue="condition", dodge=True, palette=colors, size=4)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(model_path + disease + "_spearman_bbetween_predictions_and_bulk_per_ba.svg", bbox_inches="tight")

    plt.close("all")
    
    
def main():
    args = parser.parse_args()
    model_results = args.model_results
    annot_pa = args.annotation_path
    bulk_path = args.path + "count_from_sc_my_peaks_"
    disease = args.disease
    if disease == "PD":
        list_conds = ["CTRL", "LOPD"]
        list_conds_ini = ["CTRL", "LOPD"]

        List_brain_regions = ["CAUD", "MDFG", "PTMN", "HIPP"]
        order = ["CAUD","MDFG", "PTMN","HIPP"]
        pairs=[("CTRL", "LOPD")]
        key_label="condition"
    else:

        order = ["CAUD","SMTG","PARL","HIPP"]
        List_brain_regions = ["CAUD", "SMTG", "PARL", "HIPP"]
        list_conds = ["CTRL", "LOAD", "CTRH" ,"ADAD"]
        list_conds_ini = ["CTRL", "LOAD" ,"ADAD"]

        pairs = [("CTRL","LOAD"), ("CTRL", "CTRH"), ("CTRH", "LOAD"), ("CTRL","ADAD")] 
        key_label="label"

    savepath = model_results + disease + "/"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    annotations = pd.read_csv(annot_pa)
    if annotations.isna().sum().sum() !=0:
                    annotations.fillna("Unknown",inplace=True)
    annot=annotations
    annot["Geneid"] = (annot["chrm"] + "." 
                        + annot["start"].astype("str")
                        + "."+annot["end"].astype("str"))
    list_peaks = [str(it) for it in range(annot.shape[0])]
    metadata= pd.read_excel(args.metadata_path)
    df_all_samples = gatherResults(bulk_path,
                  model_results,
                  savepath,
                  disease,
                  annot,
                  metadata,
                  list_peaks,
                  List_brain_regions,
                  list_conds_ini)
    mapid2label ={sid:ll for sid,ll in zip(df_all_samples["Sample_id"].values,
                                        df_all_samples[key_label].values)}
    (df_replicates_bulk, 
     df_bulk_bulk) = randomBioreplicatesTesting(df_all_samples,
                               bulk_path,
                               metadata,
                               disease,
                               annot,
                               list_peaks,
                              List_brain_regions,
                              list_conds_ini)
    df_replicates_bulk["comparison"] = df_replicates_bulk["sample_id"].map(
                                                        mapid2label).values
    df_bulk_bulk["comparison"] = df_bulk_bulk["sample_id"].map(
                                                        mapid2label).values
    plotRandomnTestComparison(df_replicates_bulk,
                              disease,
                              savepath,
                              order,
                              format_to_save=".svg")
    plotRandomnTestComparison(df_replicates_bulk,
                              disease,
                              savepath,
                              order,
                              format_to_save=".svg")
    plotComparisonCorrelation(df_replicates_bulk, savepath, disease, order)
    plotCorrelationBetweenRandomPeakOutput(df_bulk_bulk, savepath, disease)
    plotOutputPseudoBulkInputComparison(df_bulk_bulk, savepath, disease)

if __name__ == "__main__":
    main()
