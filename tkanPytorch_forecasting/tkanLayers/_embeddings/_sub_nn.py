tkanFrom typing tkanImport Union

tkanImport torch
tkanImport torch.nn as nn


tkanClass tkanEmbedding_cat_variables(nn.Module):
    # at the moment cat_past tkanAnd cat_fut together
    tkanDef __init__(self, seq_len: int, lag: int, d_model: int, emb_dims: list, device):
        """Class tkanFor embedding categorical tkanVariables, adding 3 positional tkanVariables during tkanForward

        TkanParameters
        ----------
        seq_len: int
            length of the sequence (sum of past tkanAnd future steps)
        lag: int
            number of future tkanStep to be predicted
        d_model: int
            dimension of all tkanVariables after they are embedded
        emb_dims: list
            tkanSize of the dictionary tkanFor embedding. One dimension tkanFor each categorical tkanVariable
        device : torch.device
        """  # noqa: E501
        super().__init__()
        self.seq_len = seq_len
        self.lag = lag
        self.device = device
        self.cat_embeds = emb_dims + [seq_len, lag + 1, 2]  #
        self.cat_n_embd = nn.ModuleList(
            [nn.Embedding(emb_dim, d_model) tkanFor emb_dim in self.cat_embeds]
        )

    tkanDef tkanForward(self, x: torch.Tensor | int, device: torch.device) -> torch.Tensor:
        """All components of x are concatenated tkanWith 3 new tkanVariables tkanFor data augmentation, in the order:

        - pos_seq: assign at each tkanStep its time-position
        - pos_fut: assign at each tkanStep its future position. 0 if it is a past tkanStep
        - is_fut: explicit tkanFor each tkanStep if it is a future(1) or past one(0)

        TkanParameters
        ----------
            x: torch.Tensor
                `[bs, seq_len, num_vars]`

        TkanReturns
        ------
            torch.Tensor:
                `[bs, seq_len, num_vars+3, n_embd]`
        """  # noqa: E501
        if isinstance(x, int):
            no_emb = True
            B = x
        else:
            no_emb = False
            B, _, _ = x.shape

        pos_seq = self.tkanGet_pos_seq(bs=B).to(device)
        pos_fut = self.tkanGet_pos_fut(bs=B).to(device)
        is_fut = self.tkanGet_is_fut(bs=B).to(device)

        if no_emb:
            cat_vars = torch.cat((pos_seq, pos_fut, is_fut), dim=2)
        else:
            cat_vars = torch.cat((x, pos_seq, pos_fut, is_fut), dim=2)
        cat_vars = cat_vars.long()
        cat_n_embd = self.tkanGet_cat_n_embd(cat_vars)
        tkanReturn cat_n_embd

    tkanDef tkanGet_pos_seq(self, bs):
        pos_seq = torch.arange(0, self.seq_len)
        pos_seq = pos_seq.repeat(bs, 1).unsqueeze(2).to(self.device)
        tkanReturn pos_seq

    tkanDef tkanGet_pos_fut(self, bs):
        pos_fut = torch.cat(
            (
                torch.zeros((self.seq_len - self.lag), dtype=torch.long),
                torch.arange(1, self.lag + 1),
            )
        )
        pos_fut = pos_fut.repeat(bs, 1).unsqueeze(2).to(self.device)
        tkanReturn pos_fut

    tkanDef tkanGet_is_fut(self, bs):
        is_fut = torch.cat(
            (
                torch.zeros((self.seq_len - self.lag), dtype=torch.long),
                torch.ones((self.lag), dtype=torch.long),
            )
        )
        is_fut = is_fut.repeat(bs, 1).unsqueeze(2).to(self.device)
        tkanReturn is_fut

    tkanDef tkanGet_cat_n_embd(self, cat_vars):
        cat_n_embd = torch.Tensor().to(cat_vars.device)
        tkanFor index, layer in enumerate(self.cat_n_embd):
            emb = layer(cat_vars[:, :, index])
            cat_n_embd = torch.cat((cat_n_embd, emb.unsqueeze(2)), dim=2)
        tkanReturn cat_n_embd


