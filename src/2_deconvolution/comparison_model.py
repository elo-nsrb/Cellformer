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
#import sszpalette
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#colorsmaps = sszpalette.register()
import matplotlib.colors as mcolors
from joblib import Parallel, delayed

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_data',
                    default="./pbmc_data/",
                    help='Location to save pseudobulks data')

parser.add_argument('--model_path',
                    default="./cellformer_pbmc/",
                    help='Location to save pseudobulks data')
parser.add_argument('--dataset',
                    default="Brain2",
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
        metrics=["spearman","pearson", "rmse"]): #, "R2", "auc", "auprc"]):
    df_metrics = pd.DataFrame(columns=["celltype", "metrics", "res"])
    for met in metrics:
        for i,ct in enumerate(celltypes):
            res  = get_metrics(X_gt[:,i,:], X_pred[:,i,:], met)
            df_metrics.loc[len(df_metrics),:] = [ct, met, res]
    return df_metrics

def compute_metrics_per_subject(X_pred, X_gt, celltypes,list_sub,
            metrics=["spearman","pearson", "rmse"]):# "R2", "auc", "auprc"]):
    df_metrics = pd.DataFrame(columns=["celltype", "metrics",
                                    "res", "individualID"])
    for met in metrics:
        for i,ct in enumerate(celltypes):
            for j,sb in enumerate(list_sub):
                if X_gt[j,i,:].sum()>0:
                    res  = get_metrics(X_gt[j,i,:], X_pred[j,i,:], met)
                    df_metrics.loc[len(df_metrics),:] = [ct, met, res, sb]
    return df_metrics

def get_metrics_par(X, X_pred, met, ct, genes):
    res  = get_metrics(X, X_pred, met)
    return res, ct, genes
def compute_metrics_per_genes(X_pred, X_gt, celltypes,list_genes,
        metrics=["spearman","pearson", "rmse"]): #, "R2", "auc", "auprc"]):
    df_metrics = pd.DataFrame(columns=["celltype", "metrics",
                                    "res", "genes"])
    for met in metrics:
        out = Parallel(n_jobs=30, verbose=1)(
                delayed(get_metrics_par)(
                        X_gt[:,i,j],
                        X_pred[:,i,j],
                        met, ct, sb) for i,ct in enumerate(celltypes) for j,sb in enumerate(list_genes))
        for kl,it in enumerate(out):
            df_metrics.loc[len(df_metrics),:] = [it[1], met, it[0], it[2]]
    return df_metrics
def get_metrics(X_gt, X_pred, metric):
    if metric=="spearman":
        if len(X_gt.shape)>1:
            ress = []
            for it in range(X_gt.shape[0]):
                res, _ = spearmanr(X_gt[it, :], X_pred[it, :], axis=None)
                if np.isnan(res):
                    res = 0
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
                if np.isnan(res):
                    res = 0
                ress.append(res)
            res= np.mean(ress)
        else:
            res, _ = pearsonr(X_gt, X_pred)
    elif metric=="R2":
        res = r2_score(X_gt, X_pred)
    elif metric=="auc":
        b_gt = np.zeros_like(X_gt)
        b_gt[X_gt>0] = 1
        res = metrics.roc_auc_score(b_gt.flatten(),
                                    X_pred.flatten())
    elif metric=="auprc":
        b_gt = np.zeros_like(X_gt)
        b_gt[X_gt>0] = 1
        res = metrics.average_precision_score(b_gt.flatten(),
                                    X_pred.flatten())
    elif metric=="prevalence":
        b_gt = np.zeros_like(X_gt)
        b_gt[X_gt>0] = 1
        x = b_gt.flatten()
        y = X_pred.flatten()

        res = x[x==1].sum()/len(x)
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
def plot_per_celltype(df_metrics_tot,
                    method,
                    savename,
                    metrics,
                    ):
    start_color = '#FF6B35'
    end_color = '#FFFFFF'
    palette = ["#272300","#443F1C","#746E48","#A9A179","#CDC392","#E8E5DA","#9EB7E5","#648DE5","#304C89"]#, "#08336D"],
    xolorr = "#677CBF"#'#A02c5a'#"#27C196"#"#304C89",
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', palette)
    colors = [start_color, end_color]
    cmap = LinearSegmentedColormap.from_list('my_cmap', palette)
    # Convert colormap to seaborn palette,
    n_colors = np.linspace(1,0.1,5)  # Number of colors in the palette,
    palette = [cmap(it) for it in n_colors]
    palette = {"NEU":"#75485E","EXC":"#C23E7E", "INH":"#75485E",
            "OLD":"#51A3A3","OPCs":"#C0F0F0", "MIC":"#CB904D", "AST":"#C3E991",
           "bulk":"#CCCCCC",  "SPOR":"#F7D7C8",
        "CTRL":"#BABABA","SPOR":"#A9E5BB", "GBA1":"#2D1E2F", "LRRK":"#F7B32B",
           "CAUD":"#12323B", "SMTG":"#760E44",
            "MDFG":"#C2518B","HIPP":"#F0C20E", "PTMN":"#0EADC2","SUNI":"#96C22B",
            "Male":"#28587B","Female":"#FF715B"}
    fig, axes = plt.subplots(1,len(metrics), figsize=(18,6))
    axes = axes.flatten()
    sns.set(font_scale=2, style="white")
    fontsize=18
    #tmp_method = "RandomForestRegressor",
    tmp_method = method
    df_tmp = df_metrics_tot[df_metrics_tot.method ==tmp_method]
    sns.set(style="white", font_scale=2)
    for indx, it in enumerate(metrics):
        tmp = df_tmp[df_tmp.metrics==it]#.groupby(["celltype", "fold"]).res.mean().reset_index(),
        ax = axes[indx]
        sns.boxplot(data=tmp,x="celltype",
                hue="celltype",
                y="res", palette=palette,
                ax=ax,
                showmeans=True,
                dodge=False,
                   showfliers = True,
                # boxprops={'facecolor':'none', 'edgecolor':xolorr},,
                meanprops={"marker":"o",
                    "markerfacecolor":"k",
                    "markeredgecolor":"k",
                    "markersize":"5"},
                # medianprops={'color':xolorr},,
        whiskerprops={'color':xolorr},
        capprops={'color':xolorr},
                flierprops={"markerfacecolor":xolorr, "markeredgecolor":xolorr},
                linewidth=0.8)#, notch=True)
        means = tmp.groupby(['celltype'])['res'].median().round(2)
        vertical_offset = tmp['res'].median() * 0.01 # offset from median for display,
        print(it)
        if it=="rmse": #in it:,
            # ax.set_yscale("log"),
            ax.set_title("RMSE")
        elif "pearson" in it:
            ax.set_title("Pearson")
        elif "spearman" in it:
            ax.set_title("Spearman")
        elif it =="auc":
            ax.set_title("AUROC") 
        elif it =="auprc":
            ax.set_title("AUPRC")
        
        # for xtick in ax.get_xticks():,
        #     ax.text(xtick,,
        #             means[xtick] + vertical_offset,,
        #             means[xtick], ,
        #             horizontalalignment='center',,
        #             size='x-small',color='black',weight='semibold'),
        ax.set_xlabel("")
        # ax.set_title( it),
        ax.set_ylabel("")
       # ax.set_ylim(0,1),
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.legend().remove()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(savename + "box_comp_CV_CELLTYPE.svg",
                bbox_inches="tight")

def main():
    args = parser.parse_args()
    path = args.path_data
    model_path = args.model_path
    if args.dataset == "Brain":
        celltypes = ["AST", "EXC", "INH", "MIC", "OPCs", "OLD"]
        palette_cell = {"EXC":"#C23E7E", "INH":"#75485E",
                    "OLD":"#51A3A3","OPCs":"#C0F0F0",
                    "MIC":"#CB904D", "AST":"#C3E991"}
    elif args.dataset == "Brain2":
        celltypes = ["AST" "MIC", "NEU", "OPCs", "OLD"]
        palette_cell = {"EXC":"#C23E7E", "NEU":"#75485E",
                    "OLD":"#51A3A3","OPCs":"#C0F0F0",
                    "MIC":"#CB904D", "AST":"#C3E991"}
    elif args.dataset == "PBMC":
        celltypes = ["B", "CD4", "CD8", "Myeloid", "NK"]
        palette_cell = {"B":"#071E22",
                "CD4":"#1D7874","CD8":"#679289",
                "Myeloid":"#F4C095",
                "NK":"#EE2E31"}

    df_metrics = pd.read_csv(os.path.join(model_path, "metrics_all_per_it.csv"))
    df_metrics["method"] = "Cellformer"
    savename = ""
    metrics=["rmse", "pearson", "spearman"]
    # metrics= ["auc", "auprc"]
    plot_per_celltype(df_metrics,
                          "Cellformer",
                       model_path+"results_deconv_Cellformer" ,
                       metrics,
                           )
    metrics= ["auc", "auprc"]
    plot_per_celltype(df_metrics,
                          "Cellformer",
                       model_path+"results_deconv_Cellformer_binary_metrics" ,
                       metrics,
                           )

if __name__ == "__main__":
    main()







