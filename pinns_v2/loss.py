import torch
import torch.nn as nn
import numpy as np
from torch.func import vmap
from functools import partial

from pinns_v2.common import LossComponent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualLoss(LossComponent):
    def __init__(self, pde_fn) -> None:
        super().__init__("ResidualLoss")
        self.pde_fn = pde_fn
    
    # loss sul singolo sample x_in[i] ciclato da vmap, la ritorna a vmap e alla fine del "ciclo"
    # in r_pred viene salvato questo vettore di loss, poi ci si fa la media "pde_loss"
    def _residual_loss(self, model, pde_fn, x_in):
        r = pde_fn(model, x_in)
        pde_loss = torch.mean(r**2)
        return pde_loss

    def _compute_loss_r(self, model, pde_fn, x_in):
        #f(model, pde_fn, x_in(i)), quelle in partial restano costanti
        r_pred = vmap(partial(self._residual_loss, model, pde_fn), (0), randomness="different")(x_in)
        pde_loss = torch.mean(r_pred)
        return pde_loss

    def compute_loss(self, model, x_in):
        pde_loss = self._compute_loss_r(model, self.pde_fn, x_in)
        self.history.append(pde_loss.item())
        return pde_loss

  
class ICLoss(LossComponent):
    def __init__(self, ic_fn) -> None:
        super().__init__("ICLoss")
        self.ic_fn = ic_fn
    
    def _ic_loss(self, model, ic_fn, x_in):
        u, true = ic_fn(model, x_in)
        loss_ic = torch.mean((u.flatten() - true.flatten())**2)
        return loss_ic

    def _compute_loss_ic(self, model, ic_fn, x_in):
        r_pred = vmap(partial(self._ic_loss, model, ic_fn), (0), randomness="different")(x_in)
        pde_loss = torch.mean(r_pred)
        return pde_loss

    def compute_loss(self, model, x_in):
        ic_loss = self._compute_loss_ic(model, self.ic_fn, x_in)
        self.history.append(ic_loss.item())
        return ic_loss
