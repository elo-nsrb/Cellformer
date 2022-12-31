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
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_mse, pairwise_mse, singlesrc_neg_sisdr
from losses import singlesrc_bcewithlogit, combinedpairwiseloss, combinedsingleloss, fpplusmseloss

from asteroid.models import save_publishable
from asteroid.utils import tensors_to_device

from asteroid.models import DPRNNTasNet
from data import make_dataloader,gatherCelltypes
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

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument("--model", default="SepFormerTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet", "SepFormerTasNet", "SepFormer2TasNet"])
parser.add_argument("--gpu", default="2")
parser.add_argument("--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution")
parser.add_argument("--out_dir", type=str, default="results/best_model", help="Directory in exp_dir where the eval results will be stored")
parser.add_argument("--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all")
parser.add_argument('--testset', default="test",
                        help='partition to use')
parser.add_argument('--limit', default=None)
parser.add_argument('--custom_testset', default=None,
                        help='Use a custom testset')


parser.add_argument("--ckpt_path", default="best_model.pth", help="Experiment checkpoint path")
parser.add_argument("--publishable", action="store_true", help="Save publishable.")
parser.add_argument('--binarize', action='store_true',
                        help='binarize output')
parser.add_argument('--masking', action='store_true',
                        help='binarize output')
parser.add_argument('--pure', action='store_true',
                        help='Test on pure')
parser.add_argument('--save', action='store_true',
                        help='Test on pure')

compute_metrics = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(args):
    model_path = os.path.join(args.model_path, args.ckpt_path)
    savedir = os.path.join(args.model_path, args.testset )
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    if args.pure:
        savedir = os.path.join(savedir,"test_pure")
        if not os.path.exists(savedir):
            os.mkdir(savedir)
    if args.ckpt_path != "best_model.pth":
        savedir = os.path.join(savedir,
                args.ckpt_path.split("/")[-1].split(".")[0])
        if not os.path.exists(savedir):
            os.mkdir(savedir)
    opt = parse(args.model_path + "train.yml", is_tain=True)
    annotations = pd.read_csv(opt["datasets"]["dataset_dir"] + opt["datasets"]["name"] + "_annotations.csv")
    if annotations.isna().sum().sum() !=0:
        annotations.fillna("Unknown",inplace=True)
    if not "promoter" in opt["datasets"]["name"]:
        if not "all" in opt["datasets"]["name"]:
            annot = annotations[annotations.chrm == opt["datasets"]["name"]]
    else:
        annot=annotations

    celltypes = opt["datasets"]["celltype_to_use"]
    num_spk = len(celltypes)

    # all resulting files would be saved in eval_save_dir
    eval_save_dir = os.path.join(args.model_path, args.out_dir)
    if args.custom_testset is not None:
        eval_save_dir = os.path.join(eval_save_dir, args.custom_testset)
    os.makedirs(eval_save_dir, exist_ok=True)
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


    if args.masking:
        bin_matrix = pd.read_csv("/home/eloiseb/data/scatac-seq/Binary_Matrix_no_doublet_unknown_no_suni_fdr_005_logfc_1_after_binarization.csv", index_col=0).iloc[:,:-1].T.values
        bin_matrix = np.asarray(bin_matrix)
        _, bin_matrix = gatherCelltypes(celltypes, 
                                    bin_matrix[np.newaxis,:,:],
                                    opt["datasets"]["celltypes"])#[0,:,:]
        bin_matrix /= bin_matrix.max()
        
    #limit = 1000
    if args.limit is not None:
        limit = eval(args.limit)
    else:
        limit = None
    if not os.path.exists(os.path.join(eval_save_dir, "final_metrics.json")):
        if args.ckpt_path == "best_model.pth":
            # serialized checkpoint
            model = getattr(asteroid.models, args.model).from_pretrained(model_path)
            print(model_path)
        else:
            # non-serialized checkpoint, _ckpt_epoch_{i}.ckpt, keys would start with
            # "model.", which need to be removed
            model = getattr(asteroid.models, args.model)(**opt["filterbank"], **opt["masknet"])
            all_states = torch.load(args.ckpt_path, map_location="cpu")
            print(args.ckpt_path)
            state_dict = {k.split('.', 1)[1]: all_states["state_dict"][k] for k in all_states["state_dict"]}
            model.load_state_dict(state_dict)
            # model.load_state_dict(all_states["state_dict"], strict=False)

        # Handle device placement
        if args.use_gpu:
            model.cuda()
        model_device = next(model.parameters()).device
        print(model)
        if args.testset == "train":
            use_train = True
        else:
            use_train = False
        _, mixtures, separate = make_dataloader("test",
                             is_train=False,
                                data_kwargs=opt['datasets'], 
                                num_workers=opt['datasets'] ['num_workers'],
                                   batch_size=1,
                                   limit=limit,
                                   use_train=use_train,
                                   custom_testset=args.custom_testset,
                                   pure=args.pure)#.data_loader
        # Used to reorder sources only
        #loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

        # Randomly choose the indexes of sentences to save.
#        ex_save_dir = os.path.join(eval_save_dir, "examples/")
#        if args.n_save_ex == -1:
#            args.n_save_ex = len(mixtures)
#        save_idx = random.sample(range(len(mixtures)), args.n_save_ex)
#
        series_list = []
        binary = False
        perfs = []
        torch.no_grad().__enter__()

        if limit is None:
            limit=len(mixtures)
        for idx in tqdm(range(mixtures.shape[0])):
            # Forward the network on the mixture.
            if idx<limit:
                mask = np.zeros_like(mixtures.iloc[idx, :-1])
                mask[mixtures.iloc[idx, :-1] >0] = 1
                mix = torch.from_numpy(
                            mixtures.iloc[idx,
                                         :-1].values.astype(
                                             np.float32))
                mix = tensors_to_device(mix, device=model_device)
                sources = torch.from_numpy(separate[idx].astype(np.float32))
                sources = tensors_to_device(sources, device=model_device)

                est_sources = model(mix.unsqueeze(0))

                # When inferencing separation for multi-task training,
                # exclude the last channel. Does not effect single-task training
                # models (from_scratch, pre+FT).
                if False:
                    est_sources = est_sources[:, :sources.shape[0]]
                sig = [0]*len(celltypes)
                #loss_val, reordered_sources = loss(est_sources,
                #                                    sources.unsqueeze(0), #sources[None], 
                #                                    return_est=True)
                #print("loss : " + str(loss_val))
                
                mix_np = mix.cpu().data.numpy()
                sources_np = sources.cpu().data.numpy()
                #est_sources_np = reordered_sources.cpu().data.numpy()
                est_sources_np = est_sources.cpu().data.numpy()
                #print("MASKING !!!")
                if args.masking:
                    est_sources_np = est_sources_np*bin_matrix #mask
                # For each utterance, we get a dictionary with the mixture path,
                # the input and output metrics
                if False:
                    for ex in range(mix_np.shape[0]):
                        utt_metrics = get_metrics(
                            mix_np, #[ex],
                            sources_np[0], #[ex],
                            est_sources_np[0], #[ex],
                            sample_rate=1,
                            metrics_list=compute_metrics,
                        )
                        series_list.append(pd.Series(utt_metrics))
                        #ACC, CORR, L1 = get_metrics2(sources_np[0], 
                        #                            est_sources_np[0], 
                        #                            binary=binary)
                        #for i, name in enumerate(celltypes):
                        #    sig[i] = {"ACC" : ACC[i], "CORR" : CORR[i], "L1" : L1[i]}
                        #perfs.append(sig)

                #if hasattr(test_set, "mixture_path"):
                #    utt_metrics["mix_path"] = test_set.mixture_path

                # Save some examples in a folder. Wav files and metrics as text.
                    # Write local metrics to the example folder.
                    #with open(local_save_dir + "metrics.json", "w") as f:
                    #    json.dump(utt_metrics, f, indent=0)
                if idx ==0:
                    sources_res = est_sources_np
                else:
                    sources_res = np.concatenate([sources_res,est_sources_np], axis=0) 
        del model
        torch.cuda.empty_cache()
        name = args.testset + "_" + str(args.pure)
        if args.custom_testset is not None:
            name += args.custom_testset
        if args.masking:
            name += "_masking_with_input_"
        if args.save:
            print("Saving ...")
            np.savez_compressed(os.path.join(args.model_path, "predictions_" 
                                            + name+ ".npz"),
                                mat=sources_res)
            np.savez_compressed(os.path.join(args.model_path, "true_"
                                            + name+ ".npz"),
                                mat=separate)
            mixtures.to_csv(os.path.join(
                                args.model_path,
                                "mixtures_" + name+ ".csv"))

        if opt['datasets']["normalizeMax"]:
            thres = 0
            strict = True

        else:
            thres = 1
            strict = False
        spearmancorrAnalysis(separate, sources_res, celltypes, savedir, 
                        name=name, thres_fill_nan=thres)
        plot_pred_gt_reg(separate, sources_res, savedir,
                        celltypes, name, 
                        normalizeMax= opt['datasets']["normalizeMax"],
                        pure=args.pure, binary=args.binarize)
        plot_aurc_from_sig(separate, sources_res, celltypes, savedir,
                            binarize_type="threshold", threshold=thres,
                            binary=binary, name=name, strict=strict,
                            normalize=opt['datasets']["normalizeMax"])
        keys = ["peakType", "distToTSS", "distToGeneStart", "chrm"]
        for kk in keys:
            plot_heatmap_r2_score(separate, sources_res, savedir,
                        celltypes, name, annot, sort_by=kk)
        annot=annotations
        plot_pred_gt(separate, sources_res, savedir,celltypes, name, annot=annot, keys=keys)

        # Save all metrics to the experiment folder.
        all_metrics_df = pd.DataFrame(series_list)
        all_metrics_df.to_csv(os.path.join(savedir, "all_metrics.csv"))

        # Print and save summary metrics
        final_results = {}
        for metric_name in compute_metrics:
            input_metric_name = "input_" + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()
        print("Overall metrics :")
        pprint(final_results)
        with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
            json.dump(final_results, f, indent=0)
        if "promoter" in opt["datasets"]["name"]:
            keys = ["chrm","gene_type", "Distance_TSS"]
        else:
            keys = ["peak_type","gene_type", "Distance_TSS"]
    else:
        with open(os.path.join(eval_save_dir, "final_metrics.json"), "r") as f:
            final_results = json.load(f)

    if args.publishable:
        assert args.ckpt_path == "best_model.pth"
        model_dict = torch.load(model_path, map_location="cpu")
        os.makedirs(os.path.join(args.model_path, "publish_dir"), exist_ok=True)
        publishable = save_publishable(
            os.path.join(args.model_path, "publish_dir"),
            model_dict,
            metrics=final_results,
            train_conf=train_conf,
        )


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    main(args)
