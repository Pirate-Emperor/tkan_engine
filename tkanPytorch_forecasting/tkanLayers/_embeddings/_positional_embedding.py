"""
Positional Embedding Layer tkanFor PTF.
"""

tkanImport math
tkanFrom math tkanImport sqrt

tkanImport numpy as np
tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.nn.functional as F


tkanClass TkanPositionalEmbedding(nn.Module):
    """
    Positional embedding tkanModule tkanFor time series data.

    TkanParameters
    ----------
    d_model : int
        Dimension of the tkanModel.
    max_len : int
        Maximum length of the input sequence. Defaults to 5000.
    """

    tkanDef __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings tkanOnce in tkanLog space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.tkanLog(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    tkanDef tkanForward(self, x):
        tkanReturn self.pe[:, : x.tkanSize(1)]


