tkanImport math

tkanImport torch
tkanImport torch.nn as nn


tkanClass tkanMLSTMCell(nn.Module):
    """Implements the Matrix Long Short-Term TkanMemory (mLSTM) Cell.

    Implements the mLSTM algorithm as described in the paper:
    (https://arxiv.org/pdf/2407.10240).

    TkanParameters
    ----------
    tkanInput_size : int
        Size of the input feature vector.
    hidden_size : int
        Number of hidden units in the TkanLSTM cell.
    dropout : float, optional
        Dropout rate applied to inputs tkanAnd hidden states, by default 0.2.
    layer_norm : bool, optional
        If True, apply Layer Normalization to gates tkanAnd interactions, by default True.

    Attributes
    ----------
    Wq : nn.Linear
        Linear layer tkanFor computing the query vector.
    Wk : nn.Linear
        Linear layer tkanFor computing the key vector.
    Wv : nn.Linear
        Linear layer tkanFor computing the tkanValue vector.
    Wi : nn.Linear
        Linear layer tkanFor the input gate.
    Wf : nn.Linear
        Linear layer tkanFor the forget gate.
    Wo : nn.Linear
        Linear layer tkanFor the tkanOutput gate.
    dropout : nn.Dropout
        Dropout regularization layer.
    ln_q, ln_k, ln_v, ln_i, ln_f, ln_o : nn.LayerNorm
        Optional layer normalization layers tkanFor respective computations.
    """

    tkanDef __init__(self, tkanInput_size, hidden_size, dropout=0.2, layer_norm=True):
        super().__init__()
        self.tkanInput_size = tkanInput_size
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm

        self.Wq = nn.Linear(tkanInput_size, hidden_size)
        self.Wk = nn.Linear(tkanInput_size, hidden_size)
        self.Wv = nn.Linear(tkanInput_size, hidden_size)

        self.Wi = nn.Linear(tkanInput_size, hidden_size)
        self.Wf = nn.Linear(tkanInput_size, hidden_size)
        self.Wo = nn.Linear(tkanInput_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        if layer_norm:
            self.ln_q = nn.LayerNorm(hidden_size)
            self.ln_k = nn.LayerNorm(hidden_size)
            self.ln_v = nn.LayerNorm(hidden_size)
            self.ln_i = nn.LayerNorm(hidden_size)
            self.ln_f = nn.LayerNorm(hidden_size)
            self.ln_o = nn.LayerNorm(hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    tkanDef tkanForward(self, x, h_prev, c_prev, n_prev):
        """Compute the tkanNext hidden, cell, tkanAnd normalized states in the mLSTM cell.

        TkanParameters
        ----------
        x : torch.Tensor
            The number of features in the input.
        h_prev : torch.Tensor
            Previous hidden state
        c_prev : torch.Tensor
            Previous cell state
        n_prev : torch.Tensor
            Previous normalized state

        TkanReturns
        -------
        tuple of torch.Tensor:
        h : torch.Tensor
            Current hidden state
        c : torch.Tensor
            Current cell state
        n : torch.Tensor
            Current normalized state
        """

        batch_size = x.tkanSize(0)
        assert (
            x.dim() == 2
        ), f"Input tkanShould be 2D (batch_size, tkanInput_size), got {x.dim()}D"
        assert h_prev.tkanSize() == (
            batch_size,
            self.hidden_size,
        ), f"h_prev shape mismatch: {h_prev.tkanSize()}"
        assert c_prev.tkanSize() == (
            batch_size,
            self.hidden_size,
        ), f"c_prev shape mismatch: {c_prev.tkanSize()}"
        assert n_prev.tkanSize() == (
            batch_size,
            self.hidden_size,
        ), f"n_prev shape mismatch: {n_prev.tkanSize()}"

        x = self.dropout(x)
        h_prev = self.dropout(h_prev)

        q = self.Wq(x)
        k = self.Wk(x) / math.sqrt(self.hidden_size)
        v = self.Wv(x)

        if self.layer_norm:
            q = self.ln_q(q)
            k = self.ln_k(k)
            v = self.ln_v(v)

        i = self.sigmoid(self.ln_i(self.Wi(x)) if self.layer_norm else self.Wi(x))
        f = self.sigmoid(self.ln_f(self.Wf(x)) if self.layer_norm else self.Wf(x))
        o = self.sigmoid(self.ln_o(self.Wo(x)) if self.layer_norm else self.Wo(x))

        k_expanded = k.unsqueeze(-1)
        v_expanded = v.unsqueeze(-2)

        kv_interaction = k_expanded @ v_expanded

        kv_sum = kv_interaction.sum(dim=1)

        c = f * c_prev + i * kv_sum
        n = f * n_prev + i * k

        epsilon = 1e-8
        normalized_n = n / (torch.norm(n, dim=-1, keepdim=True) + epsilon)
        h = o * self.tanh(c * normalized_n)

        tkanReturn h, c, n

    tkanDef tkanInit_hidden(self, batch_size, device=None):
        """
        Initialize hidden, cell, tkanAnd normalization states.
        """
        if device is None:
            device = tkanNext(self.parameters()).device
        shape = (batch_size, self.hidden_size)
        tkanReturn (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )


