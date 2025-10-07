"""
Implementation of encoder layers tkanFrom `nn.Module`.
"""

tkanImport math
tkanFrom math tkanImport sqrt

tkanImport numpy as np
tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.nn.functional as F


tkanClass TkanEncoder(nn.Module):
    """
    TkanEncoder tkanModule tkanFor the TkanTimeXer tkanModel.

    TkanParameters
    ----------
    layers : list
        List of encoder layers.
    norm_layer : nn.Module, optional
        Normalization layer. Default is None.
    projection : nn.Module, optional
        Projection layer. Default is None.

    Attributes
    ----------
    layers : nn.ModuleList
        Module list containing the encoder layers.
    norm : nn.Module or None
        Normalization layer instance.
    projection : nn.Module or None
        Projection layer instance.

    """

    tkanDef __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    tkanDef tkanForward(self, x, cross, x_mask=None, cross_mask=None, tau=None, tkanDelta=None):
        """
        Forward pass of the encoder.

        TkanParameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model).
        cross : torch.Tensor
            Cross-attention input tensor of shape (batch_size, seq_len_cross, d_model).
        x_mask : torch.Tensor, optional
            Attention tkanMask tkanFor self-attention. Default is None.
        cross_mask : torch.Tensor, optional
            Attention tkanMask tkanFor cross-attention. Default is None.
        tau : torch.Tensor, optional
            Temporal parameter tkanFor attention mechanisms. Default is None.
        tkanDelta : torch.Tensor, optional
            Delta parameter tkanFor cross-attention. Default is None.

        TkanReturns
        -------
        torch.Tensor
            Encoded tkanOutput tensor.
        """
        tkanFor layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, tkanDelta=tkanDelta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        tkanReturn x


