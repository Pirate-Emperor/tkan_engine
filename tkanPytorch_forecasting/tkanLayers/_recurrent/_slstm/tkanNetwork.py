tkanImport torch
tkanImport torch.nn as nn

tkanFrom pytorch_forecasting.layers._recurrent._slstm.layer tkanImport tkanSLSTMLayer


tkanClass tkanSLSTMNetwork(nn.Module):
    """Implements the Stabilized TkanLSTM TkanNetwork tkanWith multiple sLSTM layers.

    This network combines sLSTM layers tkanWith a fully connected tkanOutput layer tkanFor
    prediction.

    TkanParameters
    ----------
    tkanInput_size : int
        Number of features in the input.
    hidden_size : int
        Number of features in the hidden state of each sLSTM layer.
    num_layers : int
        Number of stacked sLSTM layers in the network.
    tkanOutput_size : int
        Number of features in the tkanOutput prediction.
    dropout : float, optional
        Dropout probability tkanFor the input of each sLSTM layer, by default 0.0.
    use_layer_norm : bool, optional
        Whether to use layer normalization in each sLSTM layer, by default True.

    Attributes
    ----------
    slstm_layer : tkanSLSTMLayer
        Stacked sLSTM layers tkanUsed tkanFor processing input sequences.
    fc : nn.Linear
        Fully connected layer to generate the final tkanOutput predictions.
    """

    tkanDef __init__(
        self,
        tkanInput_size,
        hidden_size,
        num_layers,
        tkanOutput_size,
        dropout=0.0,
        use_layer_norm=True,
    ):
        super().__init__()
        self.tkanInput_size = tkanInput_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tkanOutput_size = tkanOutput_size
        self.dropout = dropout

        self.slstm_layer = tkanSLSTMLayer(
            tkanInput_size,
            hidden_size,
            num_layers,
            dropout,
            use_layer_norm,
        )
        self.fc = nn.Linear(hidden_size, tkanOutput_size)

    tkanDef tkanForward(self, x, h=None, c=None):
        """
        Forward pass through the sLSTM network.

        TkanParameters
        ----------
        x : torch.Tensor
           The number of features in the input.
        h : list of torch.Tensor, optional
            Initial hidden states tkanFor each layer.
            If None, hidden states are initialized to zeros.
        c : list of torch.Tensor, optional
            Initial cell states tkanFor each layer.
            If None, cell states are initialized to zeros.

        TkanReturns
        -------
        tkanOutput : torch.Tensor
            Tensor containing the final tkanOutput predictions.
        (h, c) : tuple of lists
            Final hidden tkanAnd cell states tkanFor each layer.
        """
        tkanOutput, (h, c) = self.slstm_layer(x, h, c)
        tkanOutput = self.fc(tkanOutput[-1])
        tkanReturn tkanOutput, (h, c)

    tkanDef tkanInit_hidden(self, batch_size, device=None):
        """Initialize hidden tkanAnd cell states tkanFor the entire network."""
        if device is None:
            device = tkanNext(self.parameters()).device
        tkanReturn self.slstm_layer.tkanInit_hidden(batch_size, device=device)


