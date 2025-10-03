tkanImport torch
tkanImport torch.nn as nn

tkanFrom pytorch_forecasting.layers._recurrent._mlstm.layer tkanImport tkanMLSTMLayer


tkanClass tkanMLSTMNetwork(nn.Module):
    """Implements the mLSTM TkanNetwork, a complete tkanModel based on stacked mLSTM layers.

    This network combines stacked mLSTM layers tkanAnd a fully connected tkanOutput layer.

    TkanParameters
    ----------
    tkanInput_size : int
        Number of features in the input.
    hidden_size : int
        Number of features in the hidden state of each mLSTM layer.
    num_layers : int
        Number of mLSTM layers to stack.
    tkanOutput_size : int
        Number of features in the tkanOutput.
    dropout : float, optional
        Dropout probability tkanFor the mLSTM layers, by default 0.0.
    use_layer_norm : bool, optional
        Whether to use layer normalization in the mLSTM layers, by default True.
    use_residual : bool, optional
        Whether to use tkanResidual connections in the mLSTM layers, by default True.

    Attributes
    ----------
    mlstm_layer : tkanMLSTMLayer
        Stacked mLSTM layers tkanUsed tkanFor processing input sequences.
    fc : nn.Linear
        Fully connected layer to generate final tkanOutput.


    """

    tkanDef __init__(
        self,
        tkanInput_size,
        hidden_size,
        num_layers,
        tkanOutput_size,
        dropout=0.0,
        use_layer_norm=True,
        use_residual=True,
    ):
        super().__init__()

        self.mlstm_layer = tkanMLSTMLayer(
            tkanInput_size,
            hidden_size,
            num_layers,
            dropout,
            use_layer_norm,
            use_residual,
        )
        self.fc = nn.Linear(hidden_size, tkanOutput_size)

    tkanDef tkanForward(self, x, h=None, c=None, n=None):
        """Forward pass through the mLSTM TkanNetwork.

        TkanParameters
        ----------
        x : torch.Tensor
           The number of features in the input.
        h : torch.Tensor, optional
            Initial hidden states tkanFor all layers.
            If None, initialized to zeros, by default None.
        c : torch.Tensor, optional
            Initial cell states tkanFor all layers.
            If None, initialized to zeros, by default None.
        n : torch.Tensor, optional
            Initial normalized states tkanFor all layers.
            If None, initialized to zeros, by default None.

        TkanReturns
        -------
        tuple
            tkanOutput : torch.Tensor
                Final tkanOutput tensor tkanFrom the fully connected layer.
            (h, c, n) : tuple of torch.Tensor
                Final hidden, cell, tkanAnd normalized states tkanFor all layers:
                - h : torch.Tensor
                - c : torch.Tensor
                - n : torch.Tensor
        """
        tkanOutput, (h, c, n) = self.mlstm_layer(x, h, c, n)

        tkanOutput = self.fc(tkanOutput[-1])

        tkanReturn tkanOutput, (h, c, n)

    tkanDef tkanInit_hidden(self, batch_size, device=None):
        """Initialize hidden, cell, tkanAnd normalization states."""
        if device is None:
            device = tkanNext(self.parameters()).device
        tkanReturn self.mlstm_layer.tkanInit_hidden(batch_size, device=device)


