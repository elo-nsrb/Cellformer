import torch
import torch.nn as nn
from torch import optim
from pytorch_lightning import seed_everything
import argparse




class MultiOutputRegression(torch.nn.Module):

    def __init__(self,
                n_src=6,
                n_feat=21140,
                ff_hid=64,
                **kwargs
                ):
        super(MultiOutputRegression, self).__init__()
        self.n_src = n_src
        self.n_feat = n_feat
        self.ff_hid = ff_hid
        self.linear1 = torch.nn.Linear(n_feat, ff_hid)
        layers = []
        for it in range(n_src):
            layers.append(torch.nn.Linear(ff_hid, n_feat))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.linear1(x)
        outs = []
        for it in self.layers:
            outs.append(it(x))
        y = torch.stack(outs, dim=-1).permute(0,2,1)
        return y
    def serialize(self):
        """Serialize model and output dictionary.
        Returns:
        dict, serialized model with keys `model_args` and `state_dict`.
        """
        import pytorch_lightning as pl  # Not used in torch.hub


        model_conf = dict(
            model_name="FC_MOR",
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
            )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
                torch_version=torch.__version__,
                pytorch_lightning_version=pl.__version__,
                )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        config = {
                "n_src": self.n_src,
                "feature_dim": self.n_feat,
                "hidden_dim": self.ff_hid}
        return config
    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path,
                        *args, **kwargs):

        if isinstance(pretrained_model_conf_or_path, str):
            conf = torch.load(pretrained_model_conf_or_path, map_location="cpu")
        else:
            conf = pretrained_model_conf_or_path

        if "model_name" not in conf.keys():
            raise ValueError(
            "Expected config dictionary to have field "
            "model_name`. Found only: {}".format(conf.keys())
            )
        if "state_dict" not in conf.keys():
            raise ValueError(
            "Expected config dictionary to have field "
            "state_dict`. Found only: {}".format(conf.keys())
            )
        if "model_args" not in conf.keys():
            raise ValueError(
            "Expected config dictionary to have field "
            "model_args`. Found only: {}".format(conf.keys())
            )
        conf["model_args"].update(kwargs)  # kwargs overwrite config.
        # Attempt to find the model and instantiate it.
        model = cls(*args, **conf["model_args"])  # Child class.
        model.load_state_dict(conf["state_dict"])
        return model

def trainMultiOuput(X, X_gt):

    model = MultiOutputRegression()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(5):
        for i in range(1, 100, 2):
            x_train = torch.tensor(X)
            y_train = torch.tensor(X_gt).float()

            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            print(loss.detach().numpy())

def main(args):
    seed_everything(42, workers=True)
    args.save_folder = args.model_path
    opt = parse(args.save_folder + "train.yml", is_tain=True)
    annotations = pd.read_csv(opt["datasets"]["dataset_dir"] + opt["datasets"]["name"] + "_annotations.csv")
    if annotations.isna().sum().sum() !=0:
        annotations.fillna("Unknown",inplace=True)

    train_loader = make_dataloader("train", 
                                    is_train=True,
                                    data_kwargs=opt['datasets'],
                                    num_workers=opt['datasets']
                                   ['num_workers'],
                               batch_size=opt["training"]["batch_size"],
                               ratio=opt['datasets']["ratio"])#.data_loader
    val_loader = make_dataloader("val",
                                is_train=True,
                                data_kwargs=opt['datasets'], 
                            num_workers=opt['datasets'] ['num_workers'],
                                batch_size=opt["training"]["batch_size"],
                                ratio=opt['datasets']["ratio"])
    n_src = len(opt["datasets"]["celltype_to_use"])
    checkpoint_dir = os.path.join(args.model_path, "checkpoints/")
    if opt["datasets"]["only_training"]:
            monitor = "loss"
    else:
            monitor = "val_loss"
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, 
                            filename='{epoch}-{step}',
                            monitor=monitor, mode="min",
                        save_top_k=opt["training"]["epochs"],
                        save_last=True, verbose=True,
                         )
    callbacks.append(checkpoint)
    if opt["training"]["early_stop"]:

        callbacks.append(EarlyStopping(monitor=monitor, 
                            mode="min", 
                            patience=opt["training"]["patience"],
                            verbose=True,
                            min_delta=0.0))


class NonLinearMultiOutputRegression(torch.nn.Module):

    def __init__(self,
                n_src=6,
                n_feat=21140,
                ff_hid=64,
                **kwargs
                ):
        super(NonLinearMultiOutputRegression, self).__init__()
        self.n_src = n_src
        self.n_feat = n_feat
        self.ff_hid = ff_hid
        self.linear1 = nn.Sequential(torch.nn.Linear(n_feat, ff_hid),
                                        nn.ReLU())
        layers = []
        for it in range(n_src):
            layers.append(nn.Sequential(torch.nn.Linear(ff_hid, n_feat),
                                        nn.ReLU()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.linear1(x)
        outs = []
        for it in self.layers:
            outs.append(it(x))
        y = torch.stack(outs, dim=-1).permute(0,2,1)
        return y
    def serialize(self):
        """Serialize model and output dictionary.
        Returns:
        dict, serialized model with keys `model_args` and `state_dict`.
        """
        import pytorch_lightning as pl  # Not used in torch.hub


        model_conf = dict(
            model_name="FC_MOR",
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
            )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
                torch_version=torch.__version__,
                pytorch_lightning_version=pl.__version__,
                )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        config = {
                "n_src": self.n_src,
                "feature_dim": self.n_feat,
                "hidden_dim": self.ff_hid}
        return config
    @classmethod
    def from_pretrained(cls, pretrained_model_conf_or_path,
                        *args, **kwargs):

        if isinstance(pretrained_model_conf_or_path, str):
            conf = torch.load(pretrained_model_conf_or_path, map_location="cpu")
        else:
            conf = pretrained_model_conf_or_path

        if "model_name" not in conf.keys():
            raise ValueError(
            "Expected config dictionary to have field "
            "model_name`. Found only: {}".format(conf.keys())
            )
        if "state_dict" not in conf.keys():
            raise ValueError(
            "Expected config dictionary to have field "
            "state_dict`. Found only: {}".format(conf.keys())
            )
        if "model_args" not in conf.keys():
            raise ValueError(
            "Expected config dictionary to have field "
            "model_args`. Found only: {}".format(conf.keys())
            )
        conf["model_args"].update(kwargs)  # kwargs overwrite config.
        # Attempt to find the model and instantiate it.
        model = cls(*args, **conf["model_args"])  # Child class.
        model.load_state_dict(conf["state_dict"])
        return model
