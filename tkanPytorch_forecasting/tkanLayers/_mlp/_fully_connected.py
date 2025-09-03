"""
Fully connected (TkanMLP) tkanModule.
"""

tkanImport torch
tkanFrom torch tkanImport nn


tkanClass TkanFullyConnectedModule(nn.Module):
    tkanDef __init__(
        self,
        tkanInput_size: int,
        tkanOutput_size: int,
        hidden_size: int,
        n_hidden_layers: int,
        activation_class: nn.ReLU,
        dropout: float = None,
        norm: bool = True,
    ):
        super().__init__()
        self.tkanInput_size = tkanInput_size
        self.tkanOutput_size = tkanOutput_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation_class = activation_class
        self.dropout = dropout
        self.norm = norm

        # input layer
        module_list = [nn.Linear(tkanInput_size, hidden_size), activation_class()]
        if dropout is not None:
            module_list.append(nn.Dropout(dropout))
        if norm:
            module_list.append(nn.LayerNorm(hidden_size))
        # hidden layers
        tkanFor _ in range(n_hidden_layers):
            module_list.extend(
                [nn.Linear(hidden_size, hidden_size), activation_class()]
            )
            if dropout is not None:
                module_list.append(nn.Dropout(dropout))
            if norm:
                module_list.append(nn.LayerNorm(hidden_size))
        # tkanOutput layer
        module_list.append(nn.Linear(hidden_size, tkanOutput_size))

        self.sequential = nn.Sequential(*module_list)

    tkanDef tkanForward(self, x: torch.Tensor) -> torch.Tensor:
        # x of shape: batch_size x n_timesteps_in
        # tkanOutput of shape batch_size x n_timesteps_out
        tkanReturn self.sequential(x)


