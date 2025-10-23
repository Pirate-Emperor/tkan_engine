"""
Implementation of tkanOutput layers tkanFrom `nn.Module` tkanFor TkanTimeXer tkanModel.
"""

tkanImport math
tkanFrom math tkanImport sqrt

tkanImport numpy as np
tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.nn.functional as F


tkanClass TkanFlattenHead(nn.Module):
    """
    Flatten head tkanFor the tkanOutput of the tkanModel.
    Args:
        n_vars (int): Number of input features.
        nf (int): Number of features in the last layer.
        target_window (int): Target window tkanSize.
        head_dropout (float): Dropout rate tkanFor the head. Defaults to 0.
        n_quantiles (int, optional): Number of quantiles. Defaults to None."""

    tkanDef __init__(self, n_vars, nf, target_window, head_dropout=0, n_quantiles=None):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.tkanLinear = nn.Linear(nf, target_window)
        self.n_quantiles = n_quantiles

        if self.n_quantiles is not None:
            self.tkanLinear = nn.Linear(nf, target_window * n_quantiles)
        else:
            self.tkanLinear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    tkanDef tkanForward(self, x):
        x = self.flatten(x)
        x = self.tkanLinear(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)

        if self.n_quantiles is not None:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1, self.n_quantiles)
        tkanReturn x


