import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from statannotations.Annotator import Annotator
from sklearn import metrics
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
import sszpalette
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
colorsmaps = sszpalette.register()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_data',
                    default="./pbmc_data/",
                    help='Location to save pseudobulks data')

parser.add_argument('--model_path',
                    default="./cellformer_pbmc/",
                    help='Location to save pseudobulks data')
parser.add_argument('--dataset',
                    default="PBMC",
                    help='Location to save pseudobulks data')
def nmf_decomposition(X, X_gt):

    start = time.time()
    model = NMF(n_components=X_gt.shape[1],
                init='random',
                random_state=0)
    W = model.fit_transform(X)#n_samples, n_component
    H = model.components_#n_component, n_features
    pred = W[:,:,np.newaxis]*H[np.newaxis,:,:]
    end = time.time() - start
    return pred, end

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(32, input_dim=n_inputs,
            kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer='adam')
    return model

def cv_iteration(x_, y_, train_index, test_index, clf_name):
        start = time.time()
        X_train, X_test = x_[train_index], x_[test_index]
        y_train, y_test = y_[train_index], y_[test_index]
        clf = get_clf(clf_name,
                        n_inputs=X_train.shape[1],
                        n_outputs=y_train.shape[1])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        end = time.time() - start
        return (y_pred, test_index, end)


def multiOut_regression(X, X_gt, groups, clf_name):

    X_pred = np.zeros_like(X_gt)
    list_times=[]
    for i in range(X_gt.shape[1]):
        x_ = X
        y_ = X_gt[:,i,:]
        logo = LeaveOneGroupOut()
        for train_index,test_index in logo.split(x_, y_, groups):
            start = time.time()
            X_train, X_test = x_[train_index], x_[test_index]
            y_train, y_test = y_[train_index], y_[test_index]
            clf = get_clf(clf_name,
                            n_inputs=X_train.shape[1],
                            n_outputs=y_train.shape[1])
            clf.fit(X_train, y_train)
            X_pred[test_index,i,:] = clf.predict(X_test)
            end = time.time() - start
            list_times.append(end)
    mean_time = np.mean(list_times)
    return X_pred, mean_time
def multiOut_regression_Comp(X_train, X_gt_train,
                                groups,
                             X_test, X_gt_test,
                             groups_test,
                                clf_name):

    X_pred = np.zeros_like(X_gt_test)
    list_times=[]
    for i in range(X_gt_train.shape[1]):
        x_ = X_train
        y_ = X_gt_train[:,i,:]
        logo = LeaveOneGroupOut()
        for train_index,test_index in logo.split(x_, y_, groups):
            try:
                start = time.time()
                x_train, x_test = x_[train_index], x_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]
                test_group = np.asarray(groups)[test_index]
                test_index_i = [i for i, e in enumerate(groups_test) if e in test_group]
                clf = get_clf(clf_name,
                                n_inputs=x_train.shape[1],
                                n_outputs=y_train.shape[1])
                clf.fit(x_train, y_train)
                X_pred[test_index_i,i,:] = clf.predict(X_test[test_index_i])
                end = time.time() - start
                print(end)
                list_times.append(end)
            except:
                __import__('ipdb').set_trace()
    mean_time = np.mean(list_times)
    return X_pred, mean_time
def multiOut_regressionHOldOut(X, X_gt, train_index,
                            test_index, clf_name):

    X_pred = np.zeros_like(X_gt)
    for i in range(X_gt.shape[1]):
        x_ = X
        y_ = X_gt[:,i,:]
        X_train, X_test = x_[train_index], x_[test_index]
        y_train, y_test = y_[train_index], y_[test_index]
        clf = get_clf(clf_name,
                            n_inputs=X_train.shape[1],
                            n_outputs=y_train.shape[1])
        clf.fit(X_train, y_train)
        X_pred[test_index,i,:] = clf.predict(X_test)
        X_pred[train_index,i,:] = clf.predict(X_train)
    return X_pred

def get_clf(clf_name, n_inputs=None, n_outputs=None):
    if clf_name=="LinearRegression":
        return LinearRegression()
    elif clf_name=="knn":
        return KNeighborsRegressor()
    elif clf_name=="RandomForestRegressor":
        return RandomForestRegressor(max_depth=4, random_state=2)
    elif clf_name=="MLP":
        return get_model(n_inputs,n_outputs)



def compute_metrics(X_pred, X_gt, celltypes,
                metrics=["spearman", "rmse", "R2", "auc", "auprc"]):
    df_metrics = pd.DataFrame(columns=["celltype", "metrics", "res"])
    for met in metrics:
        for i,ct in enumerate(celltypes):
            res  = get_metrics(X_gt[:,i,:], X_pred[:,i,:], met)
            df_metrics.loc[len(df_metrics),:] = [ct, met, res]
    return df_metrics

def compute_metrics_per_subject(X_pred, X_gt, celltypes,list_sub,
            metrics=["spearman", "rmse",  "auc", "auprc"]):
    df_metrics = pd.DataFrame(columns=["celltype", "metrics",
                                    "res", "individualID"])
    for met in metrics:
        for i,ct in enumerate(celltypes):
            for j,sb in enumerate(list_sub):
                res  = get_metrics(X_gt[j,i,:], X_pred[j,i,:], met)
                df_metrics.loc[len(df_metrics),:] = [ct, met, res, sb]
    return df_metrics


def get_metrics(X_gt, X_pred, metric):
    if metric=="spearman":
        if len(X_gt.shape)>1:
            ress = []
            for it in range(X_gt.shape[0]):
                res, _ = spearmanr(X_gt[it, :], X_pred[it, :], axis=None)
                if not np.isnan(res):
                    ress.append(res)
            res= np.mean(ress)
        else:
            res, _ = spearmanr(X_gt, X_pred)


    elif metric=="rmse":
        res = mean_squared_error(X_gt, X_pred)
    elif metric=="pearson":
        if len(X_gt.shape)>1:
            ress = []
            for it in range(X_gt.shape[0]):
                res, _ = pearsonr(X_gt[it,:], X_pred[it,:])
                if not np.isnan(res):
                    #res = 0
                    ress.append(res)
            res= np.mean(ress)
        else:
            res, _ = pearsonr(X_gt, X_pred)
    elif metric=="R2":
        res = r2_score(X_gt, X_pred)
    elif metric=="auc":
        b_gt = np.zeros_like(X_gt)
        b_gt[X_gt>0] = 1
        try:
            res = metrics.roc_auc_score(b_gt.flatten(),
                                    X_pred.flatten())
        except:
            res = np.nan
    elif metric=="auprc":
        b_gt = np.zeros_like(X_gt)
        b_gt[X_gt>0] = 1
        try:
            res = metrics.average_precision_score(b_gt.flatten(),
                                    X_pred.flatten())
        except:
            res = np.nan
    return res

def get_metrics_per_subject(X_gt, X_pred, metric):
    if metric=="spearman":
        ress = []
        for it in range(X_gt.shape[0]):
            res, _ = spearmanr(X_gt[it,:], X_pred[it,:], axis=None)
            #if np.isnan(res):
            #    res = 0
            ress.append(res)

    elif metric=="rmse":
        ress = []
        for it in range(X_gt.shape[0]):
            res = mean_squared_error(X_gt[it,:], X_pred[it,:])
            ress.append(res)
    elif metric=="pearson":
        ress = []
        for it in range(X_gt.shape[0]):
            res, _ = pearsonr(X_gt[it,:], X_pred[it,:])
            ress.append(res)
    elif metric=="R2":
        ress = []
        for it in range(X_gt.shape[0]):
            res = r2_score(X_gt[it,:], X_pred[it,:])
            ress.append(res)
    elif metric=="auc":
        ress = []
        for it in range(X_gt.shape[0]):
            b_gt = np.zeros_like(X_gt[it,:])
            b_gt[X_gt[it,:]>0] = 1
            res = metrics.roc_auc_score(b_gt.flatten(),
                                        X_pred[it,:].flatten())
            ress.append(res)
    elif metric=="auprc":
        ress = []
        for it in range(X_gt.shape[0]):
            b_gt = np.zeros_like(X_gt[it,:])
            b_gt[X_gt[it,:]>0] = 1
            res = metrics.average_precision_score(b_gt.flatten(),
                                        X_pred[it,:].flatten())
            ress.append(res)
    return res

def main():
    args = parser.parse_args()
    path = args.path_data
    model_path = args.model_path
    if args.dataset == "Brain":
        celltypes = ["AST", "EXC", "INH", "MIC", "OPCs", "OLD"]
        palette_cell = {"EXC":"#C23E7E", "INH":"#75485E",
                    "OLD":"#51A3A3","OPCs":"#C0F0F0",
                    "MIC":"#CB904D", "AST":"#C3E991"}
    elif args.dataset == "PBMC":
        celltypes = ["B", "CD4", "CD8", "Myeloid", "NK"]
        palette_cell = {"B":"#071E22",
                "CD4":"#1D7874","CD8":"#679289",
                "Myeloid":"#F4C095",
                "NK":"#EE2E31"}

    list_df= []
    list_df_per_it= []
    df_metrics_sep = pd.read_csv(model_path + "metrics_all_per_sub.csv")
    df_metrics_sep["method"] = "sep"
    list_df.append(df_metrics_sep)
    df_metrics_sep = pd.read_csv(model_path + "metrics_all_per_sub.csv")
    df_metrics_sep["method"] = "sep"
    list_df_per_it.append(df_metrics_sep)
    #if args.dataset == "Brain":
    #    df_metrics = pd.read_csv("/home/eloiseb/stanford_drive/data/ATAC-seq_sc_detailed/bayesprism_metrics_per_it.csv")
    #    df_metrics["method"] = "BayesPrism"
    #    list_df_per_it.append(df_metrics)
    savename = model_path + "comp_ML_full_train_100_with_it"
    print(savename)

    if not os.path.exists(savename + ".csv"):
            gt= np.load(path + "cell_count_norm_pseudobulks_concatenate_separate_signals.npz")["mat"]
            inp= pd.read_csv(path + "cell_count_norm_pseudobulks_synthesize_pseudobulk_data_with_sparse.csv")
            inp = inp.groupby("Sample_num").sample(50, random_state=1)
            sample_list = inp["Sample_num"].values
            gt = gt[inp.index.tolist(), :, :]
            pred_, m_time = nmf_decomposition(
                                    inp.drop("Sample_num", axis=1),
                                    gt)
            df_metrics_pr = compute_metrics_per_subject(pred_,#*mask_test,
                                            gt, #*mask_test,
                                            celltypes,
                                            sample_list)
            df_metrics_pr["method"] = "NMF_pseudo_bulk_mask"
            df_metrics_pr["time"] = m_time
            list_df.append(df_metrics_pr)
            group_train = [it.split("_")[0]+it.split("_")[1] for it in inp.Sample_num.values]
            list_preds = [ "knn", "LinearRegression",
                            ]
            for pr in list_preds:
                if True:
                    pred_, mean_time = multiOut_regression(
                                        inp.drop("Sample_num", axis=1).values,
                                                gt,
                                                group_train,
                                                pr)
                    df_metrics_pr = compute_metrics_per_subject(pred_,
                            gt, celltypes, sample_list)
                    df_metrics_pr["method"] = pr + "_same_train"
                    df_metrics_pr["time"] = mean_time
                    list_df.append(df_metrics_pr)
            df_metrics_tot = pd.concat(list_df, axis=0)
            df_metrics_tot.to_csv(savename + ".csv")

            df_metrics_tot_per_it = pd.concat(list_df_per_it, axis=0)
            df_metrics_tot_per_it.to_csv(savename + "_per_it.csv")
        else:
            df_metrics_tot = pd.read_csv(savename+".csv")
            df_metrics_tot_per_it = pd.read_csv(savename+"_per_it.csv")
            df_metrics_tot = df_metrics_tot[~df_metrics_tot.method.str.contains("pseudobulk")]
            df_metrics_tot_per_it = df_metrics_tot_per_it[~df_metrics_tot_per_it.method.str.contains("pseudobulk")]
            list_df.append(df_metrics_tot)
            list_df_per_it.append(df_metrics_tot_per_it)
    savename_fig = "filt"

    df_metrics_tot = pd.concat(list_df, axis=0)
    df_metrics_tot_per_it = pd.concat(list_df_per_it, axis=0)
    mapping = {"sep":"Cellformer",
            #"BayesPrism":"BayesPrism",
                "NMF_pseudo_bulk_mask":"NMF",
                "LinearRegression_same_train":"Linear regression",
                "knn_same_train":"KNN"}
    df_metrics_tot["method"] = df_metrics_tot["method"].map(mapping).values
    df_metrics_tot_per_it["method"] = df_metrics_tot_per_it["method"].map(mapping).values
    #df_metrics_tot = df_metrics_tot[~df_metrics_tot.duplicated(["Unnamed: 0", "method", "metrics", "celltype"])]
    com = df_metrics_tot.method.unique()
    print(com)
    pairs = [(("Cellformer", "Cellformer"),("NMF","NMF")),
            (("Cellformer", "Cellformer"), ("Linear regression", "Linear regression")),
            (("Cellformer", "Cellformer"), ("KNN","KNN"))]
    palette = {"Cellformer":"#004d4b",
                "KNN":"#9d6100",
                "Linear regression":"#df9114",
                "NMF":"#fccb7b"}

    hue_order=["Cellformer", "NMF", "Linear regression", "KNN"]

    sns.set(font_scale=2, style="white")
    fontsize=18
    ####PER CELL TYPE
    fig, axes = plt.subplots(1,3, figsize=(20,6))
    axes = axes.flatten()
    sns.set(font_scale=2, style="white")
    fontsize=18
    tmp_method = "Cellformer"
    df_tmp = df_metrics_tot#[df_metrics_tot.method ==tmp_method]
    for indx, it in enumerate(["spearman", "auc", "auprc"]):
        tmp = df_tmp[df_tmp.metrics==it].groupby(["method", "celltype"]).res.mean().reset_index()#.res.mean()
        ax = axes[indx]
        sns.scatterplot(data=tmp,x="celltype",
                hue="method",
                y="res", palette=palette,
                ax=ax,
                #showmeans=True,
                #dodge=False,
                #meanprops={"marker":"o",
                #    "markerfacecolor":"black",
                #    "markeredgecolor":"black",
                #    "markersize":"5"},
                linewidth=0.3)#, notch=True)
        means = tmp.groupby(['celltype'])['res'].mean().round(2)
        vertical_offset = tmp['res'].mean() * 0.02
        if False:
            for xtick in ax.get_xticks():
                ax.text(xtick,
                    means[xtick] + vertical_offset,
                    means[xtick],
                    horizontalalignment='center',
                    size='x-small',color='black',weight='semibold')
        ax.set_xlabel("")
        ax.set_ylim(0,1)
        ax.set_title(tmp_method + " | " + it)
        ax.set_ylabel("")
    plt.savefig(os.path.join(model_path,savename_fig+ "CV_CELLTYPE_comparison.svg"),
                bbox_inches="tight")
    plt.close("all")

    print("PLOT done")
    df_metrics_tot.loc[df_metrics_tot.fold.isna(),"fold"] = df_metrics_tot.loc[df_metrics_tot.fold.isna(),"individualID"]

    fig, axes = plt.subplots(1,3, figsize=(20,6))
    axes = axes.flatten()
    for indx, it in enumerate(["spearman", "auc", "auprc"]):
        tmp = df_metrics_tot[df_metrics_tot.metrics==it].groupby(["method","fold"]).res.mean().reset_index()
        ax = axes[indx]
        sns.boxplot(data=tmp,y="res",
                hue="method",
                x="method", palette=palette,
                hue_order=hue_order,
                order=hue_order,
                ax=ax,
                showmeans=True,
                dodge=False,
                meanprops={"marker":"o",
                    "markerfacecolor":"black",
                    "markeredgecolor":"black",
                    "markersize":"5"},
                linewidth=0.8)#, notch=True)
        #sns.swarmplot(data=tmp, y="res",
        #        hue="method",
        #        ax=ax,
        #        size=3,
        #        hue_order=hue_order,
        #        order=hue_order,
        #        zorder=0,
        #        x="method", palette=["black"], #"Accent_r",
        #        linewidth=0.1)
        annotator = Annotator(ax, pairs, data=tmp,
                            y="res",
                            x="method",
                            hue="method",
                            hue_order=hue_order,
                            order=hue_order,
                                 )
        annotator.configure(test='Mann-Whitney',  text_format="star",
                            loc='inside', fontsize="8",
                            comparisons_correction="BH")
        annotator.apply_and_annotate()
        ax.legend("")
        means = tmp.groupby(['method'])['res'].mean().round(2)
        vertical_offset = tmp['res'].mean() * 0.02 # offset from median for display

        for xtick in ax.get_xticklabels():
            lab = xtick.get_text()
            print(lab)
            pos = xtick.get_position()[0]
            ax.text(pos,
                    means.loc[lab] + vertical_offset,
                    means.loc[lab],
                    horizontalalignment='center',
                    size='x-small',color='black',weight='semibold')
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=70, fontsize=fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel("")
        ax.set_title(it)
        ax.set_ylabel("")
    plt.savefig(os.path.join(model_path,savename_fig+ "CV_performance.svg"),
                bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()







