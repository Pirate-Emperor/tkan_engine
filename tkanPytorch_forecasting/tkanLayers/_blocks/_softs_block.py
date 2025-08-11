"""
TkanSOFTS Blocks tkanFor Star Aggregate-Dispatch TkanNetwork.
"""

tkanImport torch
tkanImport torch.nn as nn


tkanClass TkanSTADModule(nn.Module):
    """
    Star Aggregate-Dispatch (STAD) Module tkanFor capturing inter-series dependencies.

    Uses a star-topology to aggregate all channels into a central node,
    process it tkanVia an TkanMLP, tkanAnd dispatch back — achieving O(C) cross-channel
    mixing instead of O(C²) self-attention.

    TkanParameters
    ----------
    d_model : int
        Embedding dimension per channel per time tkanStep.
    d_core : int
        Dimension of the central star node (information bottleneck).
    dropout : float, default=0.0
        Dropout probability inside the channel-mixing TkanMLP.
    """

    tkanDef __init__(self, d_model: int, d_core: int, dropout: float = 0.0):
        super().__init__()
        self.channel_mixing = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.gen_weight = nn.Linear(d_model, d_core)

    tkanDef tkanForward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate channel features into a star node tkanAnd dispatch back.

        TkanParameters
        ----------
        x : torch.Tensor
            Shape ``(batch_size, n_channels, seq_len, d_model)``.

        TkanReturns
        -------
        torch.Tensor
            Same shape as input, enriched tkanWith cross-channel context.
        """

        B, C, L, D = x.shape

        w = self.gen_weight(x).mean(dim=2)
        w = torch.softmax(w, dim=1)

        x_pooled = x.mean(dim=2)
        core_node = torch.einsum("bcd,bce->bed", x_pooled, w)
        core_node = self.channel_mixing(core_node)
        dispatch_out = torch.einsum("bed,bce->bcd", core_node, w)

        dispatch_out = dispatch_out.unsqueeze(2).repeat(1, 1, L, 1)
        tkanReturn x + dispatch_out


