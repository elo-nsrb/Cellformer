import os
import numpy as np
import math
import torch
import logging
import yaml
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

def overlap_and_add(signal, frame_step):
    """
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length
    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long() # signal may in GPU or CPU
    if signal.device.type == "cuda":
        frame = frame.cuda()
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: # [B, C, T]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results

def get_logger(name, format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format='%Y-%m-%d %H:%M:%S', file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger(__name__)

def parse(opt_path, is_tain=True):
    '''
       opt_path: the path of yml file
       is_train: True
    '''
    logger.info('Reading .yml file .......')
    with open(opt_path,mode='r') as f:
        opt = yaml.load(f,Loader=yaml.FullLoader)
    # Export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    #os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    #logger.info('Export CUDA_VISIBLE_DEVICES = {}'.format(gpu_list))

    # is_train into option
    opt['is_train'] = is_tain

    return opt

def setLogger(log_path):
    """
    Define log file
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        #Logging to file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        #Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_opt(opt, opt_path):
    '''
       opt_path: the path of yml file
       is_train: True
    '''
    logger.info('Saving .yml file .......')
    with open(opt_path,mode='w') as f:
        yaml.dump(opt, f)

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
    #g = sns.catplot(x="Threshold_std", y="Spearman", hue='celltypes',
    #                data=dfm, kind='point', palette="husl")
    #plt.savefig(savedir + "/Optimal_std_thrs_spearman_" + name +  ".png", bbox_inches="tight")
    #plt.close("all")
    df_sp = pd.DataFrame(mat_optimal_r2.T, columns = celltypes)
    df_sp["Threshold_std"] = thrs
    dfm = df_sp.melt('Threshold_std', var_name='celltypes',
            value_name='R2')
    g = sns.catplot(x="Threshold_std", y="R2", hue='celltypes',
                    data=dfm, kind='point', palette="husl")
    #plt.savefig(savedir + "/Optimal_std_thrs_R2_" + name +  ".png", bbox_inches="tight")
    optimal_std = []
    tmp = np.zeros_like(mask)
    for it in range(len(celltypes)): 
        optth = np.argmax(mat_optimal_sp[it])
        tmp[it, mask[it]<mask[it].mean() - thrs[optth]*mask[it].std()] = 1
        optimal_std.append(thrs[optth])
    mask = tmp
    return mask
