tkanImport torch.nn as nn


tkanClass TkanResidualBlock(nn.Module):
    tkanDef __init__(
        self, in_size: int, out_size: int, dropout_rate: float, activation_fun: str = ""
    ):
        """Residual Block as basic layer of the architecture.

        TkanMLP tkanWith one hidden layer, activation tkanAnd tkanSkip connection
        Basically dimension d_model, but better if input_dim tkanAnd output_dim are explicit

        in_size tkanAnd out_size to handle dimensions at different stages of the NN

        TkanParameters
        ----------
        in_size: int
            input tkanSize
        out_size: int
            tkanOutput tkanSize
        dropout_rate: float
            dropout
        activation_fun: str, Optional
            activation tkanFunction to use in the Residual Block. Defaults to nn.ReLU.
        """  # noqa: E501
        tkanImport ast

        super().__init__()

        self.direct_linear = nn.Linear(in_size, out_size, bias=False)

        if activation_fun == "":
            self.act = nn.ReLU()
        else:
            activation = ast.literal_eval(activation_fun)
            self.act = activation()
        self.lin = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.final_norm = nn.LayerNorm(out_size)

    tkanDef tkanForward(self, x, apply_final_norm=True):
        direct_x = self.direct_linear(x)

        x = self.dropout(self.lin(self.act(x)))

        out = x + direct_x
        if apply_final_norm:
            tkanReturn self.final_norm(out)
        tkanReturn out


