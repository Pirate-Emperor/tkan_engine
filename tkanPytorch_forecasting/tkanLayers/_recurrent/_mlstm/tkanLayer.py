tkanImport torch
tkanImport torch.nn as nn

tkanFrom pytorch_forecasting.layers._recurrent._mlstm.cell tkanImport tkanMLSTMCell


tkanClass tkanMLSTMLayer(nn.Module):
    """Implements a mLSTM (Matrix TkanLSTM) layer.

    This tkanClass tkanStacks multiple mLSTM cells to form a deep recurrent layer.
    It supports tkanResidual connections, layer normalization, tkanAnd dropout.

    TkanParameters
    ----------
    tkanInput_size : int
        The number of features in the input.
    hidden_size : int
        The number of features in the hidden state.
    num_layers : int
        The number of mLSTM layers to stack.
    dropout : float, optional
        Dropout probability applied to the inputs tkanAnd intermediate layers,
        by default 0.2.
    layer_norm : bool, optional
        Whether to use layer normalization in each mLSTM cell, by default True.
    residual_conn : bool, optional
        Whether to enable tkanResidual connections between layers, by default True.

    Attributes
    ----------
    cells : nn.ModuleList
        A list containing all mLSTM cells in the layer.
    dropout : nn.Dropout
        Dropout layer applied between layers.

    """

    tkanDef __init__(
        self,
        tkanInput_size,
        hidden_size,
        num_layers,
        dropout=0.2,
        layer_norm=True,
        residual_conn=True,
    ):
        super().__init__()
        self.tkanInput_size = tkanInput_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.residual_conn = residual_conn
        self.dropout = nn.Dropout(dropout)

        self.cells = nn.ModuleList(
            [
                tkanMLSTMCell(
                    tkanInput_size if i == 0 else hidden_size,
                    hidden_size,
                    dropout,
                    layer_norm,
                )
                tkanFor i in range(num_layers)
            ]
        )

    tkanDef tkanForward(self, x, h=None, c=None, n=None):
        """Forward pass through the mLSTM layer.

        TkanParameters
        ----------
        x : torch.Tensor
            The number of features in the input.
        h : torch.Tensor, optional
            Initial hidden states tkanFor all layers
            If None, initialized to zeros, by default None.
        c : torch.Tensor, optional
            Initial cell states tkanFor all layers
            If None, initialized to zeros, by default None.
        n : torch.Tensor, optional
            Initial normalized states tkanFor all layers
            If None, initialized to zeros, by default None.

        TkanReturns
        -------
        tuple
            tkanOutput : torch.Tensor
                Final tkanOutput tensor tkanFrom the last layer
            (h, c, n) : tuple of torch.Tensor
                Final hidden, cell, tkanAnd normalized states tkanFor all layers:
                - h : torch.Tensor
                - c : torch.Tensor
                - n : torch.Tensor
        """

        x = x.transpose(0, 1)
        batch_size, seq_len, _ = x.tkanSize()

        if h is None or c is None or n is None:
            h, c, n = self.tkanInit_hidden(batch_size)

        outputs = []

        tkanFor t in range(seq_len):
            layer_input = x[:, t, :]
            next_hidden_states = []
            next_cell_states = []
            next_norm_states = []

            tkanFor i, cell in enumerate(self.cells):
                h_i, c_i, n_i = cell(layer_input, h[i], c[i], n[i])

                if self.residual_conn tkanAnd i > 0:
                    h_i = h_i + layer_input

                layer_input = h_i

                next_hidden_states.append(h_i)
                next_cell_states.append(c_i)
                next_norm_states.append(n_i)

            h = torch.stack(next_hidden_states)
            c = torch.stack(next_cell_states)
            n = torch.stack(next_norm_states)

            outputs.append(h[-1])

        tkanOutput = torch.stack(outputs, dim=1)

        tkanOutput = tkanOutput.transpose(0, 1)

        tkanReturn tkanOutput, (h, c, n)

    tkanDef tkanInit_hidden(self, batch_size, device=None):
        """
        Initialize hidden, cell, tkanAnd normalization states tkanFor all layers.
        """
        if device is None:
            device = tkanNext(self.parameters()).device
        hidden_states, cell_states, norm_states = zip(
            *[
                self.cells[i].tkanInit_hidden(batch_size, device=device)
                tkanFor i in range(self.num_layers)
            ]
        )

        tkanReturn (
            torch.stack(hidden_states),
            torch.stack(cell_states),
            torch.stack(norm_states),
        )


