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
from asteroid.utils import tensors_to_device
from data import *
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc,precision_recall_curve, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
from test_functions import *
from utils import get_logger, parse #device, 
from src_ssl.models import *
from src_ssl.models.sepformer_tasnet import SepFormerTasNet, SepFormer2TasNet
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_mse, pairwise_mse, singlesrc_neg_sisdr
from losses import *
from network import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='final.pth.tar',
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

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(args):
    model_path = os.path.join(args.model_path, args.ckpt_path)
    savedir = os.path.join(args.model_path, args.type)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    if args.ckpt_path != "best_model.pth":
        savedir = os.path.join(savedir,
                args.ckpt_path.split("/")[-1].split(".")[0])
        if not os.path.exists(savedir):
            os.mkdir(savedir)

    opt = parse(os.path.join(args.model_path , "train.yml"), is_tain=True)
    #annotations = pd.read_csv(opt["datasets"]["dataset_dir"] + opt["datasets"]["name"] + "_annotations.csv")
    #if annotations.isna().sum().sum() !=0:
    #    annotations.fillna("Unknown",inplace=True)
    #annot=annotations
    #annot["Geneid"] = (annot["chrm"] + "." 
    #                  + annot["start"].astype("str")
    #                  + "."+annot["end"].astype("str"))
    if args.type == "bulk":
        mixtures = pd.read_csv(args.peak_count_matrix, sep="\t",header=1)
        name = args.peak_count_matrix.split("/")[-1].split(".")[0]
        #mixtures = mixtures[mixtures.Geneid.isin(annot.Geneid)]
        mixtures = mixtures.iloc[:,6:].T
        mixtures_tt = mixtures/50000.
        norm_per_cell = mixtures_tt.sum(1)
        mixtures_tt = mixtures_tt*100/norm_per_cell[:, np.newaxis]
        mixtures_tt["Sample_num"] = mixtures_tt.index.tolist()
        mixtures["Sample_num"] = mixtures.index.tolist()
        savedir += "/bulk_sample_decon/"
        if not os.path.exists(savedir):
            os.mkdir(savedir)
    elif args.type == "pseudobulk":
        mixtures = pd.read_csv(args.peak_count_matrix, index_col=0)
        name = args.peak_count_matrix.split("/")[-1].split(".")[0]
        mixtures_tt = mixtures
        if "Sample_num" not in mixtures.columns.tolist():
            mixtures["Sample_num"] = mixtures.index.tolist()
            mixtures = mixtures.reset_index(drop=True)
        savedir = os.path.join(savedir, name)
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        savedir += "/pseudobulk_experiments/"
        if not os.path.exists(savedir):
            os.mkdir(savedir)
    print("Savedir : " + savedir)
    
    celltypes = opt["datasets"]["celltype_to_use"]
    num_spk = len(celltypes)

    #mask = np.zeros_like(mixtures_tt.iloc[:,:-1])
    #mask[mixtures_tt.iloc[:,:-1] >0] = 1
    #mask = mask[:, np.newaxis,:]
    #print(" mask shape " + str(mask.shape))
    if args.mask:
        mask = np.load(os.path.join(args.model_path , "MASK.npy"))
        savedir += "/masked/"
        if not os.path.exists(savedir):
            os.mkdir(savedir)
    if True:
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
            # non-serialized checkpoint, _ckpt_epoch_{i}.ckpt, keys would start with
            # "model.", which need to be removed
            model = getattr(asteroid.models, args.model)(**opt["filterbank"], **opt["masknet"])
            all_states = torch.load(args.ckpt_path, map_location="cpu")
            state_dict = {k.split('.', 1)[1]: all_states["state_dict"][k] for k in all_states["state_dict"]}
            model.load_state_dict(state_dict, strict=False)
            # model.load_state_dict(all_states["state_dict"], strict=False)

        # Handle device placement
        if args.use_gpu:
            model.cuda()
        model_device = next(model.parameters()).device

        # Randomly choose the indexes of sentences to save.

        series_list = []
        binary = False
        perfs = []
        torch.no_grad().__enter__()
        mix = mixtures_tt.iloc[:, :-1].values.astype(
                                             np.float32)
        if opt['datasets']["normalizeMax"]:
            max_val = np.max(mix,1)
            mix = mix/max_val[:, np.newaxis]
        mix = torch.from_numpy(mix)
        
        mix = tensors_to_device(mix, device=model_device)

        sources_res = model(mix)
        sources_res = torch.clamp(sources_res, min=0)
        if opt['datasets']["normalizeMax"]:
            sources_res = sources_res*max_val[:, np.newaxis,np.newaxis]
            mix = mix*max_val[:, np.newaxis]
        sources_res_np = sources_res.detach().cpu().numpy()
        np.save(savedir + name +"_out.npy", sources_res_np)
        if args.mask:
            sources_res_np = sources_res_np*mask
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
        print(separate.shape)
        if opt['datasets']["normalizeMax"]:
            separate = separate/separate.max()
            thres = 0
            strict = True
        sources = torch.from_numpy(separate.astype(np.float32))
        sources = tensors_to_device(sources, device=model_device)
        testset_idx = mixtures[mixtures.Sample_num == SAMPLE_ID_TEST].index.values.tolist() 
        trainset_idx = mixtures[mixtures.Sample_num != SAMPLE_ID_TEST].index.values.tolist() 
        print(trainset_idx)
        #plot_pred_gt_reg(separate[trainset_idx,:,:], 
        #        sources_res_np[trainset_idx, :,:], savedir,
        #                celltypes, name + "before_thresh_train_only",  
        #                normalizeMax= opt['datasets']["normalizeMax"],
        #                pure=False, binary=opt["datasets"]["binarize"])
        #if not os.path.exists(os.path.join(args.model_path, "MASK_OK.npy")):
        #    mask = defineMask(separate[trainset_idx,:,:],
        #                        sources_res_np[trainset_idx, : :],
        #                        celltypes,
        #                        savedir,
        #                        name + "MASK_TRAINSET")
        #    np.save(args.model_path + "MASK.npy",mask)
        #else:
        #    mask = np.load(args.model_path + "MASK.npy")

        #(_, pred_thres,
        #    optimal_thrs) = plot_aurc_from_sig(separate[trainset_idx, :,:], 
        #                sources_res_np[trainset_idx, :, :],
        #                    celltypes, savedir,
        #                    binarize_type="threshold", threshold=thres,
        #                    binary=opt["datasets"]["binarize"], 
        #                    name=name + "before_thres_Train_only", 
        #                    strict=strict,
        #                    normalize=opt['datasets']["normalizeMax"])
        #sources_res_np = sources_res_np*mask

        #(_, pred_thres,
        #    optimal_thrs) = plot_aurc_from_sig(separate[trainset_idx, :,:], 
        #                sources_res_np[trainset_idx, :, :],
        #                    celltypes, savedir,
        #                    binarize_type="threshold", threshold=thres,
        #                    binary=opt["datasets"]["binarize"], 
        #                    name=name + "Masked_train_only", 
        #                    strict=strict,
        #                    normalize=opt['datasets']["normalizeMax"])
        ##print("optimal thresholds:")
        ##print(optimal_thrs)
        #for k, ct in enumerate(celltypes):
        #    df_ = pd.DataFrame(sources_res_np[:,k,:], 
        #                        index=mixtures.index, 
        #                        columns=mixtures.columns.tolist()[:-1])
        #    df_["celltype"] = ct
        #    df_res.append(df_)
        #df_res_f = pd.concat(df_res, axis=0)
        #df_res_f.to_csv(savedir + name +"MASKED_optimalthres.csv")
        ##np.save(savedir + name +"MASKED_optimalthres.npy", pred_thres)
        #np.save(savedir + name +"MASKED.npy", sources_res_np)
        #plot_pred_gt_reg(separate[testset_idx, :,:], 
        #                sources_res_np[testset_idx, :, :], 
        #                    savedir,
        #                celltypes, 
        #                name + "TESTSET_filter_masked", 
        #                normalizeMax= opt['datasets']["normalizeMax"],
        #                    binary=opt["datasets"]["binarize"], 
        #                pure=False )
        #(_, _, _) = plot_aurc_from_sig(separate[testset_idx, :,:], 
        #                    sources_res_np[testset_idx, :,:],
        #                    celltypes, savedir,
        #                    binarize_type="threshold", threshold=thres,
        #                    binary=opt["datasets"]["binarize"], 
        #                    name=name + "TESTSET_filtered_masked", 
        #                    strict=strict,
        #                    normalize=opt['datasets']["normalizeMax"])
        #(_, _, _) = plot_aurc_from_sig(separate, 
        #                    sources_res_np,
        #                    celltypes, savedir,
        #                    binarize_type="threshold", threshold=thres,
        #                    binary=opt["datasets"]["binarize"], 
        #                    name=name + "ALL_filtered_masked", 
        #                    strict=strict,
        #                    normalize=opt['datasets']["normalizeMax"])
        keys = ["peakType", "distToTSS", "distToGeneStart", "chrm"]
        #for kk in keys:
        #    plot_heatmap_r2_score(separate, sources_res, savedir,
        #                celltypes, name, annot, sort_by=kk)
        #spearmancorrAnalysis(separate[0,:,:].T[:,:,np.newaxis], 
        #                    sources_res[0,:,:].T[:,:,np.newaxis], celltypes, savedir, 
         #               name=name, thres_fill_nan=thres)
        #annot=annotations
        #prefix = args.type
        #plot_pred_gt(separate, sources_res_np*mask, savedir,celltypes, name + "FINAL", annot=annot, keys=keys)
        #(mixtures, 
        # array_color_ba, map_colors_ba,
        # array_color_perso, 
        # map_colors_perso) = prepare_mixtures(mixtures_tt, separate,
        #                                     annot,
        #                                        type="pseudobulk")
        #plot_scatter_plot_per_metadata_color(separate, 
        #        sources_res_np*mask, celltypes, savedir,
        #                                    prefix,
        #                                    array_color_ba, map_colors_ba, 
        #                                    "brain_area")
        #plot_scatter_plot_per_metadata_color(separate, sources_res_np*mask, 
        #                                celltypes, savedir,
        #                                    prefix,
        #                                    array_color_perso, map_colors_perso, 
        #                                    "Subject")


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    main(args)
