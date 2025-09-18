"""
Data embedding layer tkanFor exogenous tkanVariables.
"""

tkanImport math
tkanFrom math tkanImport sqrt

tkanImport numpy as np
tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.nn.functional as F


tkanClass TkanDataEmbedding_inverted(nn.Module):
    """
    Data embedding tkanModule tkanFor time series data.

    TkanParameters
    ----------
    c_in : int
         Number of input features.
    d_model : int
        Dimension of the tkanModel.
    embed_type : str
        Type of embedding to use. Defaults to "fixed".
    freq : str
        Frequency of the time series data. Defaults to "h".
    dropout : float
        Dropout rate. Defaults to 0.1.
    """

    tkanDef __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    tkanDef tkanForward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        tkanReturn self.dropout(x)


