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
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.colors import Normalize , LogNorm
from scipy.interpolate import interpn
import math


import utils
import sys
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde
sys.setrecursionlimit(1000000)

def plot_aurc_from_sig(input_signals, pred_signals, celltypes, savedir,
        binarize_type="threshold", threshold=1, binary=False, name="test",
        plot_filtered=False,
        strict=False, normalize=False):
    if not binary :#or normalize:
        input_signals_b = np.zeros_like(input_signals)
        if binarize_type=="threshold":
            if strict:
                input_signals_b[input_signals > threshold] =1
            else:
                input_signals_b[input_signals >= threshold] =1
        elif binarize_type == "mean":
            input_signals_b =binarizeMeanSeparateSignal(input_signals, celltypes)
    else:
        input_signals_b = input_signals
        pred_signals_b = pred_signals
    if normalize:
        pred_signals_b = pred_signals
        input_signals_b = np.zeros_like(input_signals)
        input_signals_b[input_signals > 0] =1

    if True:
        perfs = []
        fig, axes = plt.subplots(2, 4, figsize=(18,14))
        axes=axes.ravel()
        overall_auprc = []
        optimal_thr = []
        for idx, nm in enumerate(celltypes):
            y_true = input_signals_b[:,idx,:].ravel()
            y_pred = pred_signals[:, idx,:].ravel()
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)

            # Calculate the G-mean
            gmean = np.sqrt(tpr * (1 - fpr))

            # Find the optimal threshold
            index = np.argmax(gmean)
            thresholdOpt = round(thresholds[index], ndigits = 4)
            gmeanOpt = round(gmean[index], ndigits = 4)
            fprOpt = round(fpr[index], ndigits = 4)
            tprOpt = round(tpr[index], ndigits = 4)
            print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
            print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

            # Calculate the Youden's J statistic
            youdenJ = tpr - fpr

            # Find the optimal threshold
            index = np.argmax(youdenJ)
            thresholdOpt = round(thresholds[index], ndigits = 4)
            youdenJOpt = round(gmean[index], ndigits = 4)
            fprOpt = round(fpr[index], ndigits = 4)
            tprOpt = round(tpr[index], ndigits = 4)
            print('Best Threshold: {} with Youden J statistic: {}'.format(thresholdOpt, youdenJOpt))
            print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

            ## locate the index of the largest f score
            precision, recall, thresholds = precision_recall_curve(y_true,
                                                                    y_pred)
            auc_precision_recall = auc(recall, precision)
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.argmax(fscore)
            opt_threshold = thresholds[ix]
            optimal_thr.append(opt_threshold)
            print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], 
                                                            fscore[ix]))
            fscoreOpt = round(fscore[ix], ndigits = 4)
            recallOpt = round(recall[ix], ndigits = 4)
            precisionOpt = round(precision[ix], ndigits = 4)
            print('Recall: {}, Precision: {}'.format(recallOpt,
                                                    precisionOpt))
            no_skill = len(y_true[y_true==1]) / len(y_true)
            print("no skills : " +str(no_skill))
            # plot the no skill precision-recall curve
            axes[idx].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
            axes[idx].legend()
            axes[idx].plot(recall, precision)
            axes[idx].set_ylabel("precision")
            axes[idx].set_xlabel("recall")
            axes[idx].set_title(nm + " %s"%str(auc_precision_recall),fontsize=16)
            overall_auprc.append(auc_precision_recall)
        opr = np.mean(overall_auprc)
        fig.suptitle("Mean :%s"%str(opr))
        print("mean opr " + str(opr))
        fig.tight_layout()
        plt.savefig(savedir + "/prauc_" + str(threshold) +"_" + str(opr) + "_" + binarize_type + name + ".png", bbox_inches="tight")
        plt.close("all")
        fig, axes = plt.subplots(2, 4, figsize=(18,14))
        axes=axes.ravel()
        overall_auc = []
        for idx, nm in enumerate(celltypes):
            y_true = input_signals_b[:,idx,:].ravel()
            y_pred = pred_signals[:, idx,:].ravel()
            fpr, tpr, _ = roc_curve(y_true,y_pred)
            auc_precision_recall = roc_auc_score(y_true, y_pred)
            axes[idx].plot(fpr, tpr)
            axes[idx].set_ylabel("True Positive Rate")
            axes[idx].set_xlabel("False Positive Rate")
            axes[idx].set_title(nm + " %s"%str(auc_precision_recall),fontsize=16)
            overall_auc.append(auc_precision_recall)
        ocr = np.mean(overall_auc)
        fig.suptitle("Mean :%s"%str(ocr))
        fig.tight_layout()
        plt.savefig(savedir + "/auc_" + str(threshold) + "_" + str(ocr) +"_" + binarize_type + name + ".png", bbox_inches="tight")
        plt.close("all")
        if binary or normalize:
            #threshold = opt_threshold
            input_signals_b = np.zeros_like(input_signals)
            if binarize_type=="threshold":
                input_signals_b[input_signals > threshold] =1
            elif binarize_type == "mean":
                input_signals_b =binarizeMeanSeparateSignal(input_signals, celltypes)
                pred_signals_b = binarizeMeanSeparateSignal(pred_signals_b,
                                                                celltypes)
        pred_thres = np.zeros_like(pred_signals)
        for idx, nm in enumerate(celltypes):
            tmp = pred_signals[:,idx,:].copy()
            tmp[tmp<optimal_thr[idx]] = 0
            pred_thres[:, idx,:] = tmp.copy()

        np.save(savedir + "/predictions_filter_fmax.npz", pred_thres)
        np.save(savedir + "/true_filter_%d.npz"%threshold, input_signals_b)
        fig, axes = plt.subplots(2, 4, figsize=(18,14))
        axes=axes.ravel()
        for idx, nm in enumerate(celltypes):
            y_pred = pred_signals[:,idx, :].ravel()
            y_pred[y_pred <=optimal_thr[idx]] = 0
            y_pred[y_pred >optimal_thr[idx]] = 1
            y_true = input_signals_b[:,idx,:].ravel()
            labels = [1,0]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                            display_labels=labels)
            disp.plot(ax=axes[idx])
            # plot the no skill precision-recall curve
            axes[idx].legend()
            axes[idx].set_title(nm + " ",fontsize=16)
        fig.tight_layout()
        plt.savefig(savedir + "/filtered_confusion_matrix_" + str(threshold) +"_" + "_" + binarize_type + name + ".png", bbox_inches="tight")
        plt.close("all")
        if False:
            fig, axes = plt.subplots(2,4, figsize=(16,8))#, projection='scatter_density')
            axes=axes.flatten()
            thres=2
            jj = 1000
            tol=1
            for idx, nm in enumerate(celltypes):
                x = input_signals_b[:,
                                idx,:].ravel().astype("float32")
                y = pred_signals[:,idx, :].ravel().astype("float32")

                y[y < optimal_thr[idx]] = 0
                density_scatter(x,y, ax = axes[idx], 
                        sort = True, bins = 50, fig=fig)
                axes[idx].set_ylabel("prediction")
                axes[idx].set_xlabel("True")
                axes[idx].legend(loc="upper left")
                axes[idx].set_title(nm,fontsize=16)
            fig.tight_layout()
            plt.savefig(savedir + "/Binary_scatterplot_correlation_pure_heat_filtered"+ name + ".png",
                        bbox_inches="tight")
            plt.close("all")
        return input_signals, pred_thres, optimal_thr

def plot_pred_gt(input_signals, pred_signals, savedir, celltypes, name,
        annot=None, keys=None, binarize=False):
    
    #celltypes = input_signals.shape[1]
    if keys is None:
        fig, axes = plt.subplots(2, 4, figsize=(18,14))
        axes=axes.ravel()
        for it, ct in enumerate(celltypes):
            tt = input_signals[:200,it,:].ravel()
            pp = pred_signals[:200,it,:].ravel()
            axes[it].scatter(tt, pp)
            x = np.linspace(0, max(tt.max(), pp.max()), 1000)
            axes[it].set_ylabel("prediction")
            axes[it].set_xlabel("True")
            axes[it].set_title(ct,fontsize=16)
        fig.tight_layout()
        plt.savefig(savedir + "/scatterplot" + name + ".png", bbox_inches="tight")
        plt.close("all")
    else:
        for key in keys:
            if key not in ["distToTSS", "distToGeneStart"]:
                if annot[key].isna().sum() !=0:
                    annot[key].fillna("Unknown",inplace=True)
                uni = annot[key].unique()
                cmap = plt.cm.Set3## define the colormap
                cmap2 = plt.cm.Set2## define the colormap
                cmap3 = plt.cm.Set1## define the colormap
                # extract all colors from the .jet map
                from matplotlib.colors import rgb2hex
                cmaplist = [rgb2hex(cmap(i)) for i in range(0,10)]
                cmaplist += [rgb2hex(cmap2(i)) for i in range(0,10)]
                cmaplist += [rgb2hex(cmap3(i)) for i in range(0,10)]

                colors = {u:cc for (u,cc) in zip(uni,cmaplist)} 
                #an = np.tile(annot[key], input_signals.shape[0])
            else:
                an = np.tile(annot[key], input_signals.shape[0])
                colors = (an-an.min())/(an.max()-an.min())
                
            fig, axes = plt.subplots(2, 4, figsize=(18,14))
            axes=axes.ravel()
            for it, ct in enumerate(celltypes):
                nb_samples = 1000
                tt = input_signals[:nb_samples,it,:].ravel()
                pp = pred_signals[:nb_samples,it,:].ravel()
                an = np.tile(annot[key], 
                            min(input_signals.shape[0], nb_samples))
                if key not in ["distToTSS", "distToGeneStart"]:
                    plt_list= []
                    print(key)
                    print(set(an))
                    coll = [colors[ant] for ant in an]

                    #plot_idx = np.random.permutation(tt.shape[0])
                    axes[it].scatter(tt, pp,
                            c=np.asarray(coll), #label=coll,
                            alpha=0.6)
                    h = lambda c: plt.Line2D([],[],color=c, ls="",marker="o")
                    axes[it].legend(handles=[h(colors[i]) for i in uni],
                    labels=list(uni), bbox_to_anchor=(1, 1))
                    #col = np.vectorize(colors.get)(an)
                    #scat = axes[it].scatter(tt, pp, c=col)
                else:
                    an = np.tile(annot[key], 
                                min(input_signals.shape[0], nb_samples))
                    colors = (an-an.min())/(an.max()-an.min())
                    scat = axes[it].scatter(tt, pp, 
                                    c=colors, cmap="RdYlGn")
                    plt.colorbar(scat)
                axes[it].set_ylabel("prediction")
                axes[it].set_xlabel("True")
                axes[it].set_title(ct,fontsize=16)
                #axes[-1].legend(handles=scat.legend_elements()[0], labels=annot[key])
            fig.tight_layout()
            plt.savefig(savedir + "/scatterplot" + "_" + key +".png", bbox_inches="tight")
            plt.close("all")

def plot_pred_gt_reg(input_signals, pred_signals, 
                    savedir, celltypes, name,
                    annot=None,
                    pure=False,
                    binary=False,
                    normalizeMax=False):
    
    fig, axes = plt.subplots(2, 4, figsize=(18,14))
    axes=axes.ravel()
    estimators = [('OLS', LinearRegression()),
                         ]
    colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen'}
    lw = 2
    assert input_signals.shape[0] == pred_signals.shape[0]
    nb_s = min(1000, pred_signals.shape[0])
    predictable = np.zeros_like(pred_signals[ :,:])
    thres=2
    if binary:
        threshold = 0.5
    elif normalizeMax:
        threshold = 0
    else:
        threshold = 1
    for it, ct in enumerate(celltypes):
        tt = input_signals[:nb_s,it,:].ravel()
        pp = pred_signals[:nb_s,it,:].ravel()

        for nm, estimator in estimators:
            estimator.fit(tt.reshape(-1,1), pp)
            pred_lig = estimator.predict(tt.reshape(-1,1))
            #print('Coefficient: \n', estimator.coef_)
            print('Mean squared error: %.2f'
                          % mean_squared_error(pred_lig, pp))
            rrr = r2_score(pp, pred_lig)
            print('Coefficient of determination: %.2f'
                          %rrr)
            x_mean = np.mean(tt)
            sse = np.sum(np.square(pp - pred_lig))
            mse = sse / (len(pp) - 2)
            #t_val = qt(0.95, len(pp)-2)

            se = np.sqrt(mse) #/ np.sqrt(np.sum(np.square(tt - x_mean)))
            #if not pure and not binary:
            if False:
                axes[it].plot(tt, pred_lig + thres*se, "--",color="black", 
                        linewidth=2, label="+/- 2*RMSE")
        #        axes[it].plot(tt, pred_lig - thres*se, "--",color="black", linewidth=2)
                axes[it].plot(tt, pred_lig, color=colors[nm], linewidth=lw,
                         label='%s (Coeff determination: %.2f)' % (nm, rrr))
        sp, pval = stats.spearmanr(tt, pp)
        print("spearman, pval: %f, %s"%(sp, pval))
        r2 = r2_score(tt, pp)
        print("R2: %f"%(r2))
        #axes[it].scatter(tt, pp)
        color="pink"
        axes[it].plot(tt, tt, "--",color="red", label = "x=x Rho: %f, R2: %f"%(sp, r2), linewidth=2)
        if threshold >0:
            cond = (((np.abs(pp)>=threshold) &(tt==0)) | ((np.abs(pp)<threshold) &(tt>0)))
        else:
            cond = (((np.abs(pp)>threshold) &(tt==0)) | ((np.abs(pp)<=threshold) &(tt>0)))
        #axes[it].scatter(tt[cond], pp[cond], c=color)
        #axes[it].scatter(tt[~cond], pp[~cond], c="grey")
        axes[it].scatter(tt, pp, c="grey",alpha=0.5, s=20, edgecolors='none')
        pp = pp.reshape((nb_s,-1))
        tt = tt.reshape((nb_s,-1))
        if threshold >0:
            cond = (((np.abs(pp)>=threshold) &(tt==0)) | ((np.abs(pp)<threshold) &(tt>0)))
        else:
            cond = (((np.abs(pp)>threshold) &(tt==0)) | ((np.abs(pp)<=threshold) &(tt>0)))
        predictable[:,it,:][~cond] = 1
        axes[it].set_ylabel("prediction")
        axes[it].set_xlabel("True")
        axes[it].legend(loc="upper left")
        axes[it].set_title(ct,fontsize=16)
    fig.tight_layout()
    plt.savefig(savedir + "/scatterplot_correlation" + name + "_" + str(threshold) + ".png", bbox_inches="tight")
    plt.close("all")
    if True:
        fig, axes = plt.subplots(2,4, figsize=(16,8))
        axes=axes.flatten()
        thres=2
        jj = 1000
        tol=1
        for idx, nm in enumerate(celltypes):
            x = input_signals[:jj,
                            idx,:].ravel().astype("float32")
            y = pred_signals[:jj, idx,:].ravel().astype("float32")
            density_scatter(x,y, ax = axes[idx], 
                    sort = True, bins = 50, fig=fig)
            axes[idx].set_ylabel("prediction")
            axes[idx].set_xlabel("True")
            axes[idx].legend(loc="upper left")
            axes[idx].set_title(nm,fontsize=16)
        fig.tight_layout()
        plt.savefig(savedir + "/Reg_scatterplot_correlation_pure_heat"+ name + ".png",
                    bbox_inches="tight")
        plt.show()
    if annot is not None:
        try:
            ax= sns.clustermap(predictable.mean(0)[:len(celltypes)],
                     yticklabels=celltypes, xticklabels=annot["nearestGene"].tolist(), figsize=(26,8))
        except:
            fig, ax = plt.subplots(figsize=(26,8))
            sns.heatmap(predictable.mean(0)[:len(celltypes)],
                    ax=ax,
                     yticklabels=celltypes, xticklabels=annot["nearestGene"].tolist())

    else:
        ax =sns.clustermap(predictable.mean(0)[:len(celltypes)],
                yticklabels=celltypes, figsize=(26,8))
    #ax.set_ylabel("predictability", fontsize=22)
    plt.savefig(savedir + "/scatterplot_heatmap_predictable" + name + ".png", bbox_inches="tight")
    plt.close("all")

def plotMSEPerPeaks(input_signals,
                         pred_signals,
                    savedir, celltypes, name,
                    annot=None,
                    pure=False,
                    binary=False,
                    normalizeMax=False):
    pred_sc = np.zeros_like(pred_signals[0, :,:])
    #pred_sc_pval = np.zeros_like(pred_signals[0, :,:])
    for it, ct in enumerate(celltypes):
        for peaks in range(input_signals.shape[-2]):
            y_true = input_signals[:,it,peaks].ravel()
            y_pred = pred_signals[:, it,peaks].ravel()
            #precision, recall, thresholds = precision_recall_curve(y_true,
            #                                                        y_pred)
            #pred_auc[it, peaks] = auc(recall, precision)
            #fpr, tpr, _ = roc_curve(y_true,y_pred)
            #pred_roc[it, peaks] = roc_auc_score(y_true, y_pred)
            pred_sc[it, peaks] = mean_squared_error(
                                    y_true,
                                    y_pred)
            
    list_mat_metric = [ pred_sc]#, pred_sc_pval]#pred_auc, pred_roc,
    list_mat_metric_name = ["mat_rms"]#, "mat_spearman_pval"]#"mat_auc", "mat_roc", 
    for mat, name in list(zip(list_mat_metric, list_mat_metric_name)):
        #ax= sns.clustermap(mat,
        #             yticklabels=celltypes, 
        #             figsize=(26,8))
        #plt.savefig(savedir + "/clusterheatmap_%s"%name + name + ".png",
        #        bbox_inches="tight")
        #plt.close("all")
        fig, ax = plt.subplots(figsize=(26,8))
        sns.heatmap(mat,
                ax=ax,
                 yticklabels=celltypes) 
        plt.savefig(savedir + "/heatmap_%s"%name + name + ".png",
                bbox_inches="tight")
        plt.close("all")
        np.save(savedir + "mat_%s"%name + ".npy", mat)


def get_metrics2(input_signals, pred_signals, sample_size=100, binary=False):
    acc = []
    correlation = []
    l1 = []
    proportion_acc=[]
    proportion_correlation=[]
    acc_balanced=[]
    roc_auc=[]
    
    for i in range(input_signals.shape[0]):
        acc.append(compute_accuracy(input_signals[i,:], pred_signals[i,:]))
        correlation.append(compute_correlation(pred_signals[i,:], input_signals[i,:]))
        l1.append(compute_l1(pred_signals[i,:], input_signals[i,:]))
        if False:
            pred_signals = np.where(pred_signals>0.5, 1, 0)
            acc_balanced.append(f1_score(input_signals[i,:], pred_signals[i,:]))
            if input_signals[i,:].sum().sum()>1e-5:
                roc_auc.append(roc_auc_score(input_signals[i,:], pred_signals[i,:]))
    return acc, correlation, l1

def compute_accuracy(logits, targets, pct_cut=0.05):
    """
    Compute prediction accuracy
    :param targets:
    :param pct_cut:
    :return:
    """
    equality = np.less_equal(
        np.abs(np.subtract(logits, targets)), pct_cut
    )
    accuracy = equality.astype(np.float32).mean()
    return accuracy

def compute_correlation(logits, targets):
    """
    Calculate the pearson correlation coefficient
    :param logits:
    :param targets:
    :return:
    """
    mx = np.mean(logits)
    my = np.mean(targets)
    xm, ym = logits - mx, targets - my
    r_num = np.sum(np.multiply(xm, ym))
    r_den = np.sqrt(
        np.multiply(
            np.sum(np.square(xm)),
            np.sum(np.square(ym)),
        )
    )
    if r_den==0:
        #print("correlation Nan return 0")
        return 0
    r = np.divide(r_num, (r_den+ 1e-5))
    r = np.maximum(np.minimum(r, 1.0), -1.0)
    return r

def compute_l1(logits, targets, pct_cut=0.05):
    """
    Compute prediction accuracy
    :param targets:
    :param pct_cut:
    :return:
    """
    l1 = np.mean(np.abs(targets-logits))
    return l1

def using_mpl_scatter_density(x, y, fig, cols=1, rows=1, index=1):
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
                    (0, '#ffffff'),
                    (1e-20, '#440053'),
                    (0.2, '#404388'),
                    (0.4, '#2a788e'),
                    (0.6, '#21a784'),
                    (0.8, '#78d151'),
                    (1, '#fde624'),
                    ], N=256)
    ax = fig.add_subplot(rows, cols, index, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')

def density_scatter(x, y, ax = None, 
                    sort = True, bins = 20, fig=None, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    z = np.log(z/z.max() + 1e-7)

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    cmap ="Reds"
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(1.0,0.0,ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=cmap+'_alpha',colors=color_array)

    # register this new colormap with matplotlib
    #plt.register_cmap(cmap=map_object)

    ax.scatter( x, y, c=z,s=20,cmap = cmap,
                    **kwargs )
    

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm, cmap=cmap), ax=ax)
    cbar.ax.set_ylabel('Density')

    return ax

def plot_heatmap_r2_score(target_sources, pred_sources, savepath, celltypes, name, annotations, sort_by=None):
    r_square_array = np.zeros((target_sources.shape[1],
                                target_sources.shape[2]))
    for i, ct in enumerate(celltypes):
        for j in range(pred_sources.shape[2]):
            sp = r2_score(target_sources[:,i,j], pred_sources[:,i, j])
            r_square_array[i,j] = sp #r2_score(y_true, y_pred, multioutput='raw_values')
    #df_.columns= annotations["nearestGene"].values
    try:
        ax = sns.clustermap(r_square_array, 
                        yticklabels=celltypes, vmin=0 )#, ax=ax)
    except:
        fig, ax = plt.subplots(figsize=(28,8))
            #ax = sns.heatmap(data=df_, vmin=0, vmax=1) 
        sns.heatmap(r_square_array, ax=ax,
                yticklabels=celltypes, vmin=0 )#, ax=ax)
    #ax.set_yticklabels(celltypes)
    #ax.set_title("R square")
    plt.savefig(savepath + "/_heatmap_" + ct + "_" + name+ ".png")
    plt.close("all")
    #if sort_by is not None:
    #    if sort_by != "distToTSS":
    #        for key in annotations[sort_by].unique(): 
    #            index_genes = annotations[annotations[sort_by] == key]["nearestGene"]
    #            ddf_ = df_[index_genes]
    #            if ddf_.shape[0]>0: 
    #                fig, ax = plt.subplots(figsize=(28,8))
    #                ax = sns.heatmap(data=ddf_, vmin=0, vmax=1) 
    #                ax.set_yticklabels(celltypes)
    #                ax.set_title("R square")
    #                plt.savefig(savepath + "/_heatmap_R_square" + ct + "_"
    #                            + name+ "_"+ key + ".png")
    #                plt.close("all")
    #    else:
    #        dff_ = df_.T
    #        fig, axes = plt.subplots(2, 4, figsize=(18,14))
    #        axes=axes.ravel()
    #        dff_["distToTSS"] = annotations["distToTSS"].values
    #        if False:
    #            for it, ct in enumerate(celltypes):
    #                axes[it].scatter(dff_["distToTSS"].values[:1000], 
    #                        dff_[ct][:1000])
    #                axes[it].set_ylabel("R square")
    #                axes[it].set_xlabel("distToTSS")
    #                axes[it].legend(loc="upper left")
    #                axes[it].set_title(ct,fontsize=16)
    #            fig.tight_layout()
    #            plt.savefig(savepath + "/distance_tss_rsquare" + name + ".png", bbox_inches="tight")
    #            plt.close("all")

def spearmancorrAnalysis(true, pred, celltypes, savedir, 
                        name="test", thres_fill_nan=1):
    from scipy import stats
    import math
    spearman_correlation = np.zeros((pred.shape[1], pred.shape[2]))
    nb_nan = 0
    for it in range(pred.shape[1]):
        for j in range(pred.shape[2]):
            sp, pvalue = stats.spearmanr( pred[:,it,j],true[:,it, j])
            if np.isnan(sp):
                nb_nan +=1
                if np.all(pred[:,it, j]<thres_fill_nan) and np.all(true[:,it, j]==0):
                    sp=1
                else:
                    sp=0
                                                    #print("ok")
                                                              
            spearman_correlation[it,j] = sp
    print("number of nan : %d"%nb_nan)
    ax = sns.clustermap(spearman_correlation,figsize=(22,8),yticklabels=celltypes, vmin=0.4 )#, ax=ax)
    plt.savefig(savedir + "/clustermap_spearman_" + name + ".png", bbox_inches="tight")
    plt.close("all")
    fig, axes=plt.subplots(2,4, figsize=(15,6))
    axes = axes.flatten()
    for i, it in enumerate(celltypes):
        idx = [j for j in range(len(celltypes)) if j!=i] 
        hue = spearman_correlation[idx,:].mean(0)
        g = sns.scatterplot(x = np.log(true.mean(0)[i]),
                            y=spearman_correlation[i], ax=axes[i],
                            hue=hue)
            
        #g.axes.legend_.set_title("spearman correlation %s"%celltypes[1-i])
                    #ax.axvline(x=np.log(1) , ymin=0.01, ymax=1, color="k", label="mean > 1", )
        axes[i].set_xlabel("log mean count per type of cells")
        axes[i].set_ylabel("Spearman correlation")
        axes[i].set_title(it)
    plt.savefig(savedir + "/spearman_coorelation_mean_count_" + name +  ".png", bbox_inches="tight")
    plt.close("all")

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

    plot_pred_gt_reg(input_signals, 
                pred_signals*mask,savedir, celltypes, 
                "filtered_masked")
    return mask

