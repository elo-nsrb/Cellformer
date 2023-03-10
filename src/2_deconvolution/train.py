#steroid is based on PyTorch and PyTorch-Lightning.
#import comet_ml
import torch
import torch.nn as nn
from torch import optim
from utils import get_logger, parse #device, 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from asteroid.engine.schedulers import DPTNetScheduler
import pytorch_lightning as pl
#https://github.com/SungFeng-Huang/SSL-pretraining-separation
from src_ssl.models import *
import asteroid
import json
import yaml
from network import *
from asteroid.models import DPRNNTasNet
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_mse, pairwise_mse, singlesrc_neg_sisdr, SinkPITLossWrapper
from losses import *
from data import *
import argparse
from asteroid.engine import System
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument("--model", default="SepFormerTasNet", choices=["ConvTasNet", "DPRNNTasNet", "DPTNet", "SepFormerTasNet", "SepFormer2TasNet", "FC_MOR", "NL_MOR"])
parser.add_argument("--gpu", default="2")
parser.add_argument("--resume_ckpt", default="last.ckpt", help="Checkpoint path to load for resume-training")
parser.add_argument("--resume", action="store_true", help="Resume-training")

def main(args):
    seed_everything(42, workers=True)
    args.save_folder = args.model_path
    opt = parse(args.save_folder + "train.yml", is_tain=True)
    annotations = pd.read_csv(opt["datasets"]["dataset_dir"] + opt["datasets"]["name"] + "_annotations.csv")
    if annotations.isna().sum().sum() !=0:
        annotations.fillna("Unknown",inplace=True)
    if not "promoter" in opt["datasets"]["name"]:
        if not "all" in opt["datasets"]["name"]:
            annot = annotations[annotations.chrm == opt["datasets"]["name"]]
    else:
        annot=annotations

    logger = get_logger(__name__)

    logger.info('Building the model of %s'%args.model)
    train_loader = make_dataloader("train", 
                                is_train=True,
                                data_kwargs=opt['datasets'],
                                num_workers=opt['datasets']
                               ['num_workers'],
                               batch_size=opt["training"]["batch_size"],
                               ratio=opt['datasets']["ratio"])
    val_loader = make_dataloader("val",
                                is_train=True,
                                data_kwargs=opt['datasets'], 
                                num_workers=opt['datasets'] ['num_workers'],
                                batch_size=opt["training"]["batch_size"],
                                ratio=opt['datasets']["ratio"])


    n_src = len(opt["datasets"]["celltype_to_use"])

    if args.model == "FC_MOR":
        model = MultiOutputRegression(**opt["net"])
        print(model)
    elif args.model == "NL_MOR":
        model = NonLinearMultiOutputRegression(**opt["net"])
        print(model)
    else:
        model = getattr(asteroid.models, args.model)(**opt["filterbank"], **opt["masknet"])
    print(model)

    learnable_params = list(model.parameters())
    if opt["training"]["loss"] == "neg_sisdr":
        loss_func = pairwise_neg_sisdr 
        loss = PITLossWrapper(loss_func, pit_from="pw_mtx")
    elif opt["training"]["loss"] == "mse_no_pit":
        #loss = nn.MSELoss()
        if opt["training"]["weights"] is not None:
            weights = np.asarray(opt["training"]["weights"])
        else:
            weights = None
        loss = weightedloss(src=n_src, weights=weights)
    elif opt["training"]["loss"] == "mse_weighted":
        #loss = nn.MSELoss()
        loss = weightedloss(src=n_src, method="Uncertainty")
        learnable_params += list(loss.parameters())
    elif opt["training"]["loss"] == "mixtemse":
        loss = MixteMSE()
    elif opt["training"]["loss"] == "mea":
        loss = nn.L1Loss()
    elif opt["training"]["loss"] == "bce":
        loss = nn.BCEWithLogitsLoss() #singlesrc_bcewithlogit
        #loss = PITLossWrapper(loss_func, pit_from="pw_pt")
    elif opt["training"]["loss"] == "bcemse":
        loss = BCEMSE_loss() #singlesrc_bcewithlogit
    elif opt["training"]["loss"] == "pearson":
        loss = Pearson_loss() #singlesrc_bcewithlogit
    elif opt["training"]["loss"] == "pearson_mse":
        loss = PearsonMSE_loss()
    elif opt["training"]["loss"] == "pair_mse":
        loss_func = pairwise_mse
        loss = PITLossWrapper(loss_func, pit_from="pw_mtx")
    elif opt["training"]["loss"] == "pair_bce":
        loss_func = pairwise_bce
        loss = PITLossWrapper(loss_func, pit_from="pw_mtx")
    elif opt["training"]["loss"] == "combined":
        loss_func = combinedpairwiseloss
        loss = PITLossWrapper(loss_func, pit_from="pw_mtx")
    elif opt["training"]["loss"] == "combined_sc_no_pit":
        loss = combinedsingleloss
        #loss = PITLossWrapper(loss_func, pit_from="pw_mtx")
    elif opt["training"]["loss"] == "fp_mse":
        loss_func = fpplusmseloss
        loss = PITLossWrapper(loss_func, pit_from="pw_pt")

    optimizer = optim.AdamW(learnable_params, lr=1e-3)
    # Define scheduler
    scheduler = None
    if args.model in ["DPTNet", "SepFormerTasNet", "SepFormer2TasNet"]:
        steps_per_epoch = len(train_loader) // opt["training"]["accumulate_grad_batches"]
        opt["scheduler"]["steps_per_epoch"] = steps_per_epoch
        scheduler = {
                    "scheduler": DPTNetScheduler(
                    optimizer=optimizer,
                    steps_per_epoch=steps_per_epoch,
                     d_model=model.masker.mha_in_dim,
                     ),
                         "interval": "batch",
                        }
    else: 
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                      factor=0.8,
                                      patience=opt["training"]["patience"]
                                      )

    system = System(model, 
            optimizer, 
            loss, 
            train_loader
            , val_loader,
            scheduler=scheduler)


    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(args.model_path, "checkpoints/")
    if opt["datasets"]["only_training"]:
            monitor = "loss"
    else:
            monitor = "val_loss"
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, 
                            filename='{epoch}-{step}',
                            monitor=monitor, mode="min",
                        save_top_k=opt["training"]["save_epochs"],
                        save_last=True, verbose=True,
                         )
    callbacks.append(checkpoint)
    if opt["training"]["early_stop"]:

        callbacks.append(EarlyStopping(monitor=monitor, 
                            mode="min", 
                            patience=opt["training"]["patience"],
                            verbose=True,
                            min_delta=0.0))

    loggers = []
    tb_logger = pl.loggers.TensorBoardLogger(
                        os.path.join(args.model_path, "tb_logs/"),
                        )
    loggers.append(tb_logger)
    if opt["training"]["comet"]:
        comet_logger = pl.loggers.CometLogger(
            save_dir=os.path.join(args.model_path, "comet_logs/"),
            experiment_key=opt["training"].get("comet_exp_key", None),
            log_code=True,
            log_graph=True,
            parse_args=True,
            log_env_details=True,
            log_git_metadata=True,
            log_git_patch=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_host=True,
            )
        comet_logger.log_hyperparams(opt)
        loggers.append(comet_logger)

    if args.resume:
        resume_from = os.path.join(checkpoint_dir, args.resume_ckpt)
    else:
        resume_from = None
    trainer = Trainer(max_epochs=opt["training"]["epochs"],
                    #batch_size =opt["training"]["batch_size"],
                    logger=loggers,
                    callbacks=callbacks,
                    default_root_dir=args.model_path,
            accumulate_grad_batches=opt[ "training"]["accumulate_grad_batches"],
                resume_from_checkpoint=resume_from,
                    #deterministic=True,
                    gpus=4,
                    auto_select_gpus=True)
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(args.model_path, "best_k_models.json"), "w") as f:
            json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    train_set_infos = dict()
    train_set_infos["dataset"] = "Brain"

    to_save = system.model.serialize()
    to_save.update(train_set_infos)
    torch.save(to_save, os.path.join(args.model_path, "best_model.pth"))

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
