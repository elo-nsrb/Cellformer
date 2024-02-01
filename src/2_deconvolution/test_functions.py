from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc,precision_recall_curve, r2_score
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns
from scipy import stats
import pandas as pd
#import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.colors import Normalize , LogNorm
import math
import utils
import sys
from sklearn import linear_model


def defineMask(input_signals, pred_signals, celltypes,
                savedir, name="test"):
    mask = (np.abs(input_signals 
            - pred_signals).mean(0)
            /(np.abs(input_signals).mean(0) + 1e-5))
    thrs = np.around(np.arange(-0.5,1,0.1, dtype=float),2)
    mat_optimal_sp = np.zeros((len(celltypes), thrs.shape[0])) 
    mat_optimal_r2 = np.zeros((len(celltypes), thrs.shape[0])) 
    for idx,th in enumerate(thrs):
        tmp = np.zeros_like(mask)
        for it in range(mask.shape[0]):
            tmp[it, mask[it]<mask[it].mean() - th*mask[it].std()] = 1
            pp = pred_signals[:,it,:]*tmp[it,:]
            pp = pp.ravel()
            tt = input_signals[:,it,:].ravel()
            print("threshold %f"%th)
            sp, pval = stats.spearmanr(tt, pp)
            print("spearman, pval: %f, %s"%(sp, pval))
            r2 = r2_score(tt, pp)
            print("R2: %f"%(r2))
            if math.isnan(sp):
                sp = 0
            if math.isnan(r2):
                r2 = 0
            mat_optimal_sp[it,idx] = sp
            mat_optimal_r2[it,idx] = r2

    df_sp = pd.DataFrame(mat_optimal_sp.T, columns = celltypes)
    df_sp["Threshold_std"] = thrs
    dfm = df_sp.melt('Threshold_std', var_name='celltypes',
                    value_name='Spearman')
    g = sns.catplot(x="Threshold_std", y="Spearman", hue='celltypes',
                    data=dfm, kind='point', palette="husl")
    plt.savefig(savedir + "/Optimal_std_thrs_spearman_" + name +  ".png", bbox_inches="tight")
    plt.close("all")
    df_sp = pd.DataFrame(mat_optimal_r2.T, columns = celltypes)
    df_sp["Threshold_std"] = thrs
    dfm = df_sp.melt('Threshold_std', var_name='celltypes',
            value_name='R2')
    g = sns.catplot(x="Threshold_std", y="R2", hue='celltypes',
                    data=dfm, kind='point', palette="husl")
    plt.savefig(savedir + "/Optimal_std_thrs_R2_" + name +  ".png", bbox_inches="tight")
    optimal_std = []
    tmp = np.zeros_like(mask)
    for it in range(len(celltypes)): 
        optth = np.argmax(mat_optimal_sp[it])
        tmp[it, mask[it]<mask[it].mean() - thrs[optth]*mask[it].std()] = 1
        optimal_std.append(thrs[optth])
    mask = tmp
    fig, ax = plt.subplots(figsize=(26,11))
    sns.heatmap(mask,
            ax=ax,
             yticklabels=celltypes)
    plt.savefig(savedir + "/Mask_binary_after_filtered"+ name + ".png",
                bbox_inches="tight")

    #plot_pred_gt_reg(input_signals, 
    #            pred_signals*mask,savedir, celltypes, 
    #            "filtered_masked")
    return mask

