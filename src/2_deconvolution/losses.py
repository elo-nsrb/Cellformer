import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper, singlesrc_mse, pairwise_mse, singlesrc_neg_sisdr


class SingleSrcBCE(_Loss):
    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 2:
            raise TypeError( f"Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead")
        loss = -(est_targets.log()*targets + (1-targets)*(1-est_targets).log())
        mean_over = list(range(1, loss.ndim))
        return loss.mean(dim=mean_over)

class Pearson_loss(nn.Module):
    def __init__(self,
                **kwargs):
        super(Pearson_loss,self).__init__()
        self.name = "Pearson_loss"

    def __call__(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-4)
        return 1-cost

class SingleSrcBCEWithLogit(_Loss):
    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 2:
            raise TypeError( f"Inputs must be of shape [batch, *], got {targets.size()} and {est_targets.size()} instead")
        est_targets = self.sigmoid(est_targets)
        loss = -(est_targets.log()*targets + (1-targets)*(1-est_targets).log())
        mean_over = list(range(1, loss.ndim))
        return loss.mean(dim=mean_over)

    def sigmoid(self,x):
        return (1 + (-x).exp()).reciprocal()

class FPScMSEFunction(_Loss):
    def __init__(self, size_average=None,
                reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        self.mse = singlesrc_mse

    def forward(self, est_targets, targets):
        mask = torch.clamp(targets.detach().copy(), min=0,max=1)
        return 10*(1-mask)*self.mse(est_targets, targets)

class FPPlusScMSEFunction(_Loss):
    def __init__(self, size_average=None,
                reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        self.mse = singlesrc_mse

    def forward(self, est_targets, targets):
        mask = torch.clamp(targets.detach().copy(), min=0,max=1)
        return 10*(1-mask)*self.mse(est_targets, targets) + self.mse(est_targets, targets) 


class CombinedPairwiseFunction(_Loss):
    def __init__(self, size_average=None,
                reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        self.mse = pairwise_mse
        self.neg_sisdr = pairwise_neg_sisdr

    def forward(self, est_targets, targets):
        return self.mse(est_targets, targets) + self.neg_sisdr(est_targets, targets)

class SI_SNR_loss(nn.Module):
    def __init__(self,
                **kwargs):
        super(SI_SNR_loss,self).__init__()
        self.name = "SI_SNR_loss"

    def __call__(self, x, s,eps=1e-8):
        """
        calculate training loss
        input:
              x: separated signal, N x S tensor
              s: reference signal, N x S tensor
        Return:
              sisnr: N tensor
        """
        def l2norm(mat, keepdim=False):
            return torch.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - torch.mean(x, dim=-1, keepdim=True)
        s_zm = s - torch.mean(s, dim=-1, keepdim=True)
        t = torch.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return -torch.sum(20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps)))


class CombinedSingleFunction(_Loss):
    def __init__(self, size_average=None,
                reduce=None, reduction: str = 'mean') -> None:
            super(_Loss, self).__init__()
            self.mse = nn.MSELoss() #singlesrc_mse
            self.neg_sisdr = SI_SNR_loss() #singlesrc_neg_sisdr

    def forward(self, est_targets, targets):
        return self.mse(est_targets, targets) + self.neg_sisdr(est_targets, targets)

class WeightedMSE(_Loss):
    def __init__(self, beta=0.999, gamma=0.5,
                method="Equal", weights=None,
                src=2,
                nFeats=14584) -> None:
        super(_Loss, self).__init__()
        if (weights is not None) and (weights.tolist() !="None"):
            self.weights = nn.Parameter(torch.tensor(weights,
                                            dtype=torch.float),
                                            requires_grad=False)
        else:
            print("Unweighted MSE")
            self.weights = None
        self.weights

        self.register_buffer("w", self.weights)
        self.method = method
        if self.method =="Uncertainty":
            self.log_vars = nn.Parameter(torch.zeros((src)))
    def forward(self, est_targets, targets):
        if targets.size() != est_targets.size() or targets.ndim < 3:
            raise TypeError(
                     f"Inputs must be of shape [batch, n_src, *], got {targets.size()} and {est_targets.size()} instead"
        )

        #targets = targets.unsqueeze(1)
        #est_targets = est_targets.unsqueeze(2)
        pw_loss = (targets - est_targets) ** 2
        # Need to return [batch, nsrc, nsrc]
        #print(pw_loss.shape)
        pw_loss = pw_loss.mean(-1)
        if self.method =="Uncertainty":
            weights = torch.square(torch.exp(-self.log_vars))
            pw_loss *= weights
        if self.weights is not None:
            #print(pw_loss.shape) (4,4)
            pw_loss *=self.w
        #for it in range(pw_loss.shape[1]):
         #   print(str(it) + str(pw_loss[:,it].mean()))

        return pw_loss.mean()*100

class L1PearMSE_loss(nn.Module):
    def __init__(self,
                **kwargs):
        super(L1PearMSE_loss,self).__init__()
        self.name = "MIX_loss"
        self.mea = nn.L1Loss()
        self.pearson = Pearson_loss()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        return self.mea(x,y) + self.pearson(x,y) + self.mse(x,y)

class BCEMSE_loss(nn.Module):
    def __init__(self,
                **kwargs):
        super(BCEMSE_loss,self).__init__()
        self.name = "BCEMSE_loss"
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x, target):
        return 100*self.mse(x,target) + self.bce(x,target)
class PearsonMSE_loss(nn.Module):
    def __init__(self,
                **kwargs):
        super(PearsonMSE_loss,self).__init__()
        self.name = "BCEMSE_loss"
        self.mse = nn.MSELoss()
        self.pear = Pearson_loss()

    def forward(self, x, target):
        return self.mse(x,target) + self.pear(x,target)

class MixteMSE(nn.Module):
    def __init__(self, 
                **kwargs):
        super(MixteMSE, self).__init__()
        self.mse = nn.MSELoss()
    def forward(self, x, target):
        return self.mse(x,target) + 100*self.mse(1-x,1-target)

singlesrc_bcewithlogit = SingleSrcBCEWithLogit()
combinedpairwiseloss = CombinedPairwiseFunction()
combinedsingleloss = CombinedSingleFunction()
fpplusmseloss = FPPlusScMSEFunction()
weightedloss = WeightedMSE
