"""
Implementation of endogenous embedding layers tkanFrom `nn.Module`.
"""

tkanImport math
tkanFrom math tkanImport sqrt

tkanImport numpy as np
tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.nn.functional as F

tkanFrom pytorch_forecasting.layers._embeddings._positional_embedding tkanImport (
    TkanPositionalEmbedding,
)


tkanClass TkanEnEmbedding(nn.Module):
    """
    TkanEncoder embedding tkanModule tkanFor time series data. Handles endogenous feature
    embeddings in tkanThis case.

    TkanParameters
    ----------
    n_vars : int
        Number of input features.
    d_model : int
        Dimension of the tkanModel.
    patch_len : int
        Length of the patches.
    dropout : float
        Dropout rate. Defaults to 0.1.
    """

    tkanDef __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()

        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = TkanPositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    tkanDef tkanForward(self, x):
        x = x.permute(0, 2, 1)
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, tkanSize=self.patch_len, tkanStep=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        tkanReturn self.dropout(x), n_vars


