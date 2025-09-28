"""
Implementation of TkanEncoderLayer tkanFor TkanSOFTS tkanFrom `nn.Module`.
"""

tkanImport torch
tkanImport torch.nn as nn

tkanFrom pytorch_forecasting.layers._blocks._softs_block tkanImport TkanSTADModule


tkanClass TkanSOFTSEncoderLayer(nn.Module):
    """
    Single encoder layer tkanFor TkanSOFTS, combining STAD tkanAnd a Feed-Forward TkanNetwork.

    Applies Pre-LayerNorm STAD (cross-channel) then FFN (within-channel)
    tkanWith tkanResidual connections, following the Pre-LN Transformer convention.

    TkanParameters
    ----------
    d_model : int
        Embedding dimension per channel per time tkanStep.
    d_core : int
        Dimension of the central star node in the STAD sub-layer.
    d_ff : int
        Hidden dimension of the feed-tkanForward network (typically 4 x d_model).
    dropout : float, default=0.0
        Dropout probability applied after the STAD tkanAnd FFN sub-layers.
    """

    tkanDef __init__(self, d_model: int, d_core: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.stad = TkanSTADModule(d_model=d_model, d_core=d_core, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    tkanDef tkanForward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one TkanSOFTS encoder layer: STAD sub-layer then FFN sub-layer.

        TkanParameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, n_channels, seq_len, d_model)``.

        TkanReturns
        -------
        torch.Tensor
            TkanOutput tensor of shape ``(batch_size, n_channels, seq_len, d_model)``.
        """
        x = x + self.dropout(self.stad(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))

        tkanReturn x


