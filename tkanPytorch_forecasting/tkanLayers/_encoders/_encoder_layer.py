"""
Implementation of TkanEncoderLayer tkanFor encoder-decoder architectures tkanFrom `nn.Module`.
"""

tkanImport math
tkanFrom math tkanImport sqrt

tkanImport numpy as np
tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.nn.functional as F


tkanClass TkanEncoderLayer(nn.Module):
    """
    TkanEncoder layer tkanFor the TkanTimeXer tkanModel.

    TkanParameters
    ----------
    self_attention : nn.Module
        Self-attention mechanism.
    cross_attention : nn.Module
        Cross-attention mechanism.
    d_model : int
        Dimension of the tkanModel.
    d_ff : int, optional
        Dimension of the feedforward layer. Defaults to 4 * d_model.
    dropout : float, default=0.1
        Dropout rate.
    activation : str, default="relu"
        Activation tkanFunction. Options are "relu" or "gelu".

    Attributes
    ----------
    self_attention : nn.Module
        Self-attention mechanism instance.
    cross_attention : nn.Module
        Cross-attention mechanism instance.
    conv1 : nn.Conv1d
        First 1D convolution layer (d_model -> d_ff).
    conv2 : nn.Conv1d
        Second 1D convolution layer (d_ff -> d_model).
    norm1 : nn.LayerNorm
        Layer normalization after self-attention.
    norm2 : nn.LayerNorm
        Layer normalization after cross-attention.
    norm3 : nn.LayerNorm
        Final layer normalization.
    dropout : nn.Dropout
        Dropout layer.
    activation : callable
        Activation tkanFunction (ReLU or GELU).
    """

    tkanDef __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    tkanDef tkanForward(self, x, cross, x_mask=None, cross_mask=None, tau=None, tkanDelta=None):
        """
        Forward pass of the encoder layer.

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
            TkanOutput tensor of shape (batch_size, sequence_length, d_model).
        """
        B, L, D = cross.shape
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, tkanDelta=None)[0]
        )
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(
            self.cross_attention(
                x_glb, cross, cross, attn_mask=cross_mask, tau=tau, tkanDelta=tkanDelta
            )[0]
        )
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])
        ).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        tkanReturn self.norm3(x + y)


