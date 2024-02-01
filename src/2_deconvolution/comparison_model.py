import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
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

#from keras.models import Sequential
#from keras.layers import Dense

def nmf_decomposition(X, X_gt):

    model = NMF(n_components=X_gt.shape[1],
                init='random',
                random_state=0)
    W = model.fit_transform(X)#n_samples, n_component
    H = model.components_#n_component, n_features
    pred = W[:,:,np.newaxis]*H[np.newaxis,:,:]
    return pred 

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(32, input_dim=n_inputs, 
            kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
    model.compile(loss='mse', optimizer='adam')
    return model

def multiOut_regression(X, X_gt, groups, clf_name):

    X_pred = np.zeros_like(X_gt)
    for i in range(X_gt.shape[1]):
        x_ = X
        y_ = X_gt[:,i,:]
        logo = LeaveOneGroupOut()
        for train_index,test_index in logo.split(x_, y_, groups):
            X_train, X_test = x_[train_index], x_[test_index]
            y_train, y_test = y_[train_index], y_[test_index]
            clf = get_clf(clf_name, 
                            n_inputs=X_train.shape[1], 
                            n_outputs=y_train.shape[1])
            clf.fit(X_train, y_train)
            X_pred[test_index,i,:] = clf.predict(X_test)
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
                metrics=["spearman", "pearson", "rmse", "R2", "auc", "auprc"]):
    df_metrics = pd.DataFrame(columns=["celltype", "metrics", "res"])
    for met in metrics:
        for i,ct in enumerate(celltypes):
            res  = get_metrics(X_gt[:,i,:], X_pred[:,i,:], met)
            df_metrics.loc[len(df_metrics),:] = [ct, met, res]
    return df_metrics

def compute_metrics_per_subject(X_pred, X_gt, celltypes,list_sub,
            metrics=["spearman", "pearson", "rmse",  "auc", "auprc"]):
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
    path="/home/eloiseb/stanford_drive/data/ATAC-seq_sc_detailed/scData_6_ct_LAST/"
    gt= np.load(path + "all_sc_celltype_pseudobulks.npz")["mat"]
    celltypes = ["AST", "EXC", "INH", "MIC", "OPCs", "OLD"]
    inp= pd.read_csv(path + "all_sc_pseudobulks.csv",index_col=0)
    #index = ~inp.Sample_num.str.contains("SUNI")
    #gt = gt[index, :, :]
    #inp = inp[index]

    df_metrics_["method"] = "sep"
    list_df.append(df_metrics_)
    path_s = "/home/eloiseb/experiments/deconv_peak/L49K_13_1226_redo/"
    savepath = path_s
    pred = np.load(path_s + "pseudobulk/all_sc_pseudobulks/pseudobulk_sample_decon/all_sc_pseudobulks_out.npy")
    mask = np.load(path_s + "MASK.npy")
    df_metrics_ = compute_metrics(pred,
                                     gt, #*mask[np.newaxis,:,:],
                                     celltypes) 
    df_metrics_["method"] = "sep"
    list_df.append(df_metrics_)

    list_df = []
    df_metrics_tot = pd.read_csv(savename)
    list_df.append(df_metrics_tot)

    path_s = "/home/eloiseb/experiments/deconv_peak/L49K_13_1226_redo/"
    savepath = path_s
    pred = np.load(path_s + "pseudobulk/all_sc_pseudobulks/pseudobulk_sample_decon/all_sc_pseudobulks_out.npy")
    mask = np.load(path_s + "MASK.npy")
    df_metrics_ = compute_metrics(pred,
                                     gt, #*mask[np.newaxis,:,:],
                                     celltypes) 

    df_metrics_tot = pd.concat(list_df, axis=0)
    df_metrics_tot.to_csv(savepath + "comp_with_other_loss.csv")
        
    g = sns.FacetGrid(data=df_metrics_tot[df_metrics_tot.metrics!="R2"], 
            col="metrics",
            col_wrap=2, sharey=False)
    g.map_dataframe(sns.scatterplot, x="celltype", y="res",
            hue="method")
    g.add_legend()
    plt.savefig(savepath + "comp_with_otherloss.png")



if __name__ == "__main__":
    main()





        

