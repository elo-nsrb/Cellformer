import os
import random
import torch
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import torch.nn as nn 

import asteroid
from asteroid.metrics import get_metrics
from asteroid.data.librimix_dataset import LibriMix
from asteroid.data.wsj0_mix import Wsj0mixDataset
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device
from comparison_model import *

from asteroid.models import DPRNNTasNet
from data import *
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc,precision_recall_curve, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
from test_functions import *
from analysis import *
from utils import get_logger, parse #device, 
from src_ssl.models import *
from src_ssl.models.sepformer_tasnet import SepFormerTasNet, SepFormer2TasNet
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_mse, pairwise_mse, singlesrc_neg_sisdr
from losses import singlesrc_bcewithlogit, combinedpairwiseloss, combinedsingleloss, fpplusmseloss
from network import *

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument("--model", default="ConvTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet", "SepFormerTasNet", "SepFormer2TasNet", "FC_MOR", "NL_MOR"])
parser.add_argument("--gpu", default="2")
parser.add_argument("--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution")
parser.add_argument("--out_dir", type=str, default="results/best_model", help="Directory in exp_dir where the eval results will be stored")
parser.add_argument("--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all")
parser.add_argument('--peak_count_matrix', type=str,
                        default="/home/eloiseb/stanford_drive/data/ATAC-seq/count_from_sc_peaks/AD/CTRL/CTRL_CAUD_AD.peak_countMatrix.txt",
                        help='Bulk sample')

parser.add_argument('--groundtruth',
                        default=None,
                        help='Bulk sample')
parser.add_argument('--type', type=str,
                        default="bulk",
                        )

parser.add_argument("--ckpt_path", default="best_model.pth", help="Experiment checkpoint path")
parser.add_argument("--publishable", action="store_true", help="Save publishable.")
parser.add_argument('--binarize', action='store_true',
                        help='binarize output')
parser.add_argument('--mask', action='store_true',
                        help='binarize output')
parser.add_argument('--pure', action='store_true',
                        help='Test on pure')
parser.add_argument('--save', action='store_true',
                        help='Test on pure')

#compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(args):
    parent_dir = args.parent_dir
    list_ids = ["13_1226","13_0038","13_0419", 
                "11_0393", "09_1589", "14_0586",
                "14_1018", "06_0615", "09_35", "03_39",
                "04_38", "11_0311"]
    #list_val_ids = ["13_0038", "13_1226"]
    res_cv_raw = None
    res_cv_mask = None
    df_metrics_raw_list = []
    df_metrics_mask_list = []
    gt_m = None
    for s_id in list_ids:
        args.model_path = os.path.join(parent_dir,
                                    "exp_%s/"%(s_id))
        model_path = os.path.join(args.model_path, args.ckpt_path)
        savedir = os.path.join(args.model_path, args.type)
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        if args.ckpt_path != "best_model.pth":
            savedir = os.path.join(savedir,
                    args.ckpt_path.split("/")[-1].split(".")[0])
            if not os.path.exists(savedir):
                os.mkdir(savedir)

        opt = parse(args.model_path + "train.yml", is_tain=True)
        annotations = pd.read_csv(opt["datasets"]["dataset_dir"] 
                                + opt["datasets"]["name"] + "_annotations.csv")
        if annotations.isna().sum().sum() !=0:
            annotations.fillna("Unknown",inplace=True)
        annot=annotations
        annot["Geneid"] = (annot["chrm"] + "." 
                          + annot["start"].astype("str")
                          + "."+annot["end"].astype("str"))
        if args.type == "bulk":
            mixtures = pd.read_csv(args.peak_count_matrix, sep="\t",header=1)
            name = args.peak_count_matrix.split("/")[-1].split(".")[0]
            print(name)
            mixtures = mixtures[mixtures.Geneid.isin(annot.Geneid)]
            mixtures = mixtures.iloc[:,6:].T
            mixtures_tt = mixtures/50000.
            mixtures_tt["Sample_num"] = mixtures_tt.index.tolist()
            mixtures["Sample_num"] = mixtures.index.tolist()
            savedir += "/bulk_sample_decon/"
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        elif args.type == "pseudobulk":
            mixtures = pd.read_csv(args.peak_count_matrix, index_col=0)
            name = args.peak_count_matrix.split("/")[-1].split(".")[0]
            print(name)
            if "Sample_num" not in mixtures.columns.tolist():
                mixtures["Sample_num"] = mixtures.index.tolist()
                mixtures.reset_index(inplace=True, drop=True)
            mixtures_tt = mixtures
            savedir = os.path.join(savedir, name)
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            savedir += "/pseudobulk_sample_decon/"
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        print("Savedir : " + savedir)
        if opt["training"]["loss"] == "neg_sisdr":
            loss_func = pairwise_neg_sisdr 
            loss = PITLossWrapper(loss_func, pit_from="pw_mtx")
        elif opt["training"]["loss"] == "single_neg_sisdr":
            loss_func = singlesrc_neg_sisdr
            loss = PITLossWrapper(loss_func, pit_from="pw_pt")
        elif opt["training"]["loss"] == "mse":
            loss_func = singlesrc_mse
            loss = PITLossWrapper(loss_func, pit_from="pw_pt")
        elif opt["training"]["loss"] == "mse_no_pit":
            loss = nn.MSELoss()
        elif opt["training"]["loss"] == "bce":
            loss_func = singlesrc_bcewithlogit
            loss = PITLossWrapper(loss_func, pit_from="pw_pt")
        elif opt["training"]["loss"] == "pair_mse":
            loss_func = pairwise_mse
            loss = PITLossWrapper(loss_func, pit_from="pw_mtx")
        elif opt["training"]["loss"] == "pair_bce":
            loss_func = pairwise_bce
            loss = PITLossWrapper(loss_func, pit_from="pw_mtx")
        elif opt["training"]["loss"] == "combined_sc":
            loss_func = combinedsingleloss
            loss = PITLossWrapper(loss_func, pit_from="pw_pt")
        elif opt["training"]["loss"] == "fp_mse":
            loss_func = fpplusmseloss
            loss = PITLossWrapper(loss_func, pit_from="pw_pt")

        celltypes = opt["datasets"]["celltype_to_use"]
        num_spk = len(celltypes)

        if args.mask:
            mask = np.load(args.model_path + "MASK.npy")
            savedir += "/masked/"
            if not os.path.exists(savedir):
                os.mkdir(savedir)
        if args.ckpt_path == "best_model.pth":
            # serialized checkpoint
            if args.model == "FC_MOR":
                model = MultiOutputRegression(**opt["net"])
                model = model.from_pretrained(model_path, **opt["net"])
            elif args.model == "NL_MOR":
                model = NonLinearMultiOutputRegression(**opt["net"])
                model = model.from_pretrained(model_path, **opt["net"])
            else:
                model = getattr(asteroid.models, args.model).from_pretrained(model_path)
        else:
            model = getattr(asteroid.models, args.model)(**opt["filterbank"], **opt["masknet"])
            all_states = torch.load(args.ckpt_path, map_location="cpu")
            state_dict = {k.split('.', 1)[1]: all_states["state_dict"][k] for k in all_states["state_dict"]}
            model.load_state_dict(state_dict)
                # model.load_state_dict(all_states["state_dict"], strict=False)

            # Handle device placement
        if args.use_gpu:
            model.cuda()
        model_device = next(model.parameters()).device
        series_list = []
        binary = False
        perfs = []
        torch.no_grad().__enter__()
        mix = torch.from_numpy(mixtures_tt.iloc[:,
                                         :-1].values.astype(
                                             np.float32))
        if opt['datasets']["normalizeMax"]:
            mix = mix/mix.max()
        mix = tensors_to_device(mix, device=model_device)

        sources_res = model(mix)
        sources_res = torch.clamp(sources_res, min=0)
        if opt['datasets']["normalizeMax"]:
            sources_res = sources_res*mix.max()
            mix = mix*mix.max()
        sources_res_np = sources_res.detach().cpu().numpy()
        if args.mask:
            sources_res_np = sources_res_np*mask
        if res_cv_raw is None:
            res_cv_raw = np.zeros_like(sources_res_np)
            res_cv_mask = np.zeros_like(sources_res_np)
            gt_m = np.zeros_like(sources_res_np)
        test_sp = sources_res_np[mixtures["Sample_num"].str.contains(s_id)]
        testset_idx = mixtures[mixtures.Sample_num.str.contains(s_id)].index.values.tolist() 
        trainset_idx = mixtures[~mixtures.Sample_num.str.contains(s_id)].index.values.tolist() 
        res_cv_raw[testset_idx,:,:] = test_sp
        df_res = [] 
        for k, ct in enumerate(celltypes):
            df_ = pd.DataFrame(sources_res_np[:,k,:], 
                                index=mixtures.index, 
                                columns=mixtures.columns.tolist()[:-1])
            df_["celltype"] = ct
            df_res.append(df_)
        df_res_f = pd.concat(df_res, axis=0)
        df_res_f.to_csv(savedir + name +".csv")

        if args.groundtruth is not None:
            print(savedir)
            separate = np.load(args.groundtruth)["mat"]
            if len(separate.shape)<2:
                separate = np.expand_dims(separate, 0)
            print(separate.shape)
            _, separate = gatherCelltypes(celltypes, 
                                        separate, opt["datasets"]["celltypes"] )
            df_met = compute_metrics(sources_res_np, 
                            separate, 
                            celltypes) 
            df_met["test_id"] = s_id
            df_metrics_raw_list.append(df_met)
            print(separate.shape)
            if opt['datasets']["normalizeMax"]:
                #separate = separate/separate.max()
                thres = 0
                strict = True
            sources = torch.from_numpy(separate.astype(np.float32))
            sources = tensors_to_device(sources, device=model_device)
            print(trainset_idx)
            mask = defineMask(separate[trainset_idx,:,:],
                                sources_res_np[trainset_idx, : :],
                                celltypes,
                                savedir,
                                name + "MASK_TRAINSET")
            np.save(args.model_path + "MASK.npy",mask)
            (_, pred_thres,
                optimal_thrs) = plot_aurc_from_sig(separate[trainset_idx, :,:], 
                            sources_res_np[trainset_idx, :, :],
                                celltypes, savedir,
                                binarize_type="threshold", threshold=thres,
                                binary=False, 
                                name=name + "before_thres_Train_only", 
                                strict=strict,
                                normalize=opt['datasets']["normalizeMax"])
            sources_res_np = sources_res_np*mask
            test_sp = sources_res_np[mixtures["Sample_num"].str.contains(s_id)]
            res_cv_mask[testset_idx,:,:] = test_sp
            df_met = compute_metrics(sources_res_np, 
                            separate*mask, 
                            celltypes) 
            df_met["test_id"] = s_id
            df_metrics_mask_list.append(df_met)

            print("optimal thresholds:")
            print(optimal_thrs)
            #np.save(args.model_path + "optimal_thres.npy",optimal_thrs)

            for k, ct in enumerate(celltypes):
                df_ = pd.DataFrame(sources_res_np[:,k,:], 
                                    index=mixtures.index, 
                                    columns=mixtures.columns.tolist()[:-1])
                df_["celltype"] = ct
                df_res.append(df_)
            df_res_f = pd.concat(df_res, axis=0)
            df_res_f.to_csv(savedir + name +"MASKED_thres.csv")
            np.save(savedir + name +"MASKED.npy", sources_res_np)
            test_sp = sources_res_np[mixtures["Sample_num"].str.contains(s_id)]
            res_cv_mask[testset_idx,:,:] = test_sp
            test_gt = separate[mixtures["Sample_num"].str.contains(s_id)]
            gt_m[testset_idx,:,:] = test_gt*mask[np.newaxis, :,:]

    df_metrics_tot_mask = pd.concat(df_metrics_mask_list,axis=0)
    df_metrics_tot_mask.to_csv(os.path.join(parent_dir,name + "ALL_runs_mask_metrics_cv.csv"))
    df_metrics_tot_raw = pd.concat(df_metrics_raw_list,axis=0)
    df_metrics_tot_raw.to_csv(os.path.join(parent_dir,name + "ALL_runs_raw_metrics_cv.csv"))
    df_metrics_mask= compute_metrics(res_cv_mask, 
                            gt_m, 
                            celltypes) 
    df_metrics_= compute_metrics(res_cv_raw, 
                            separate, 
                            celltypes) 
    df_metrics_.to_csv(os.path.join(parent_dir,name + "metrics_cv.csv"))
    df_metrics_mask.to_csv(os.path.join(parent_dir,name + "mask_metrics_cv.csv"))
    np.save(os.path.join(parent_dir,name+"gt_.npy"), gt_m)
    np.save(os.path.join(parent_dir,name+"pred_.npy"), res_cv_mask)


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    main(args)
