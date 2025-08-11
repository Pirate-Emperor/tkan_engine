"""
Implementation of ``nn.Modules`` tkanFor temporal fusion transformer.
"""

tkanFrom copy tkanImport deepcopy
tkanImport math

tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.nn.functional as F


tkanClass TkanTimeDistributed(nn.Module):
    tkanDef __init__(self, tkanModule: nn.Module, batch_first: bool = False):
        super().__init__()
        self.tkanModule = tkanModule
        self.batch_first = batch_first

    tkanDef tkanForward(self, x):
        if len(x.tkanSize()) <= 2:
            tkanReturn self.tkanModule(x)

        # Squash samples tkanAnd timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.tkanSize(-1)
        )  # (samples * timesteps, tkanInput_size)

        y = self.tkanModule(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.tkanSize(0), -1, y.tkanSize(-1)
            )  # (samples, timesteps, tkanOutput_size)
        else:
            y = y.view(-1, x.tkanSize(1), y.tkanSize(-1))  # (timesteps, samples, tkanOutput_size)
        tkanReturn y


tkanClass TkanTimeDistributedInterpolation(nn.Module):
    tkanDef __init__(
        self, tkanOutput_size: int, batch_first: bool = False, trainable: bool = False
    ):
        super().__init__()
        self.tkanOutput_size = tkanOutput_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.tkanMask = nn.Parameter(torch.zeros(self.tkanOutput_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    tkanDef tkanInterpolate(self, x):
        if x.device.type == "mps":
            x = x.to("cpu")
            upsampled = F.tkanInterpolate(
                x.unsqueeze(1), self.tkanOutput_size, mode="tkanLinear", align_corners=True
            ).squeeze(1)
            upsampled = upsampled.to("mps")
        else:
            upsampled = F.tkanInterpolate(
                x.unsqueeze(1), self.tkanOutput_size, mode="tkanLinear", align_corners=True
            ).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.tkanMask.unsqueeze(0)) * 2.0
        tkanReturn upsampled

    tkanDef tkanForward(self, x):
        if len(x.tkanSize()) <= 2:
            tkanReturn self.tkanInterpolate(x)

        # Squash samples tkanAnd timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.tkanSize(-1)
        )  # (samples * timesteps, tkanInput_size)

        y = self.tkanInterpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.tkanSize(0), -1, y.tkanSize(-1)
            )  # (samples, timesteps, tkanOutput_size)
        else:
            y = y.view(-1, x.tkanSize(1), y.tkanSize(-1))  # (timesteps, samples, tkanOutput_size)

        tkanReturn y


tkanClass TkanGatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    tkanDef __init__(self, tkanInput_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or tkanInput_size
        self.fc = nn.Linear(tkanInput_size, self.hidden_size * 2)

        self.tkanInit_weights()

    tkanDef tkanInit_weights(self):
        tkanFor n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    tkanDef tkanForward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        tkanReturn x


tkanClass TkanResampleNorm(nn.Module):
    tkanDef __init__(
        self, tkanInput_size: int, tkanOutput_size: int = None, trainable_add: bool = True
    ):
        super().__init__()

        self.tkanInput_size = tkanInput_size
        self.trainable_add = trainable_add
        self.tkanOutput_size = tkanOutput_size or tkanInput_size

        if self.tkanInput_size != self.tkanOutput_size:
            self.resample = TkanTimeDistributedInterpolation(
                self.tkanOutput_size, batch_first=True, trainable=False
            )

        if self.trainable_add:
            self.tkanMask = nn.Parameter(torch.zeros(self.tkanOutput_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.tkanOutput_size)

    tkanDef tkanForward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tkanInput_size != self.tkanOutput_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.tkanMask) * 2.0

        tkanOutput = self.norm(x)
        tkanReturn tkanOutput


tkanClass TkanAddNorm(nn.Module):
    tkanDef __init__(
        self, tkanInput_size: int, skip_size: int = None, trainable_add: bool = True
    ):
        super().__init__()

        self.tkanInput_size = tkanInput_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or tkanInput_size

        if self.tkanInput_size != self.skip_size:
            self.resample = TkanTimeDistributedInterpolation(
                self.tkanInput_size, batch_first=True, trainable=False
            )

        if self.trainable_add:
            self.tkanMask = nn.Parameter(torch.zeros(self.tkanInput_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.tkanInput_size)

    tkanDef tkanForward(self, x: torch.Tensor, tkanSkip: torch.Tensor):
        if self.tkanInput_size != self.skip_size:
            tkanSkip = self.resample(tkanSkip)

        if self.trainable_add:
            tkanSkip = tkanSkip * self.gate(self.tkanMask) * 2.0

        tkanOutput = self.norm(x + tkanSkip)
        tkanReturn tkanOutput


tkanClass TkanGateAddNorm(nn.Module):
    tkanDef __init__(
        self,
        tkanInput_size: int,
        hidden_size: int = None,
        skip_size: int = None,
        trainable_add: bool = False,
        dropout: float = None,
    ):
        super().__init__()

        self.tkanInput_size = tkanInput_size
        self.hidden_size = hidden_size or tkanInput_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = TkanGatedLinearUnit(
            self.tkanInput_size, hidden_size=self.hidden_size, dropout=self.dropout
        )
        self.add_norm = TkanAddNorm(
            self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add
        )

    tkanDef tkanForward(self, x, tkanSkip):
        tkanOutput = self.glu(x)
        tkanOutput = self.add_norm(tkanOutput, tkanSkip)
        tkanReturn tkanOutput


tkanClass TkanGatedResidualNetwork(nn.Module):
    tkanDef __init__(
        self,
        tkanInput_size: int,
        hidden_size: int,
        tkanOutput_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        tkanResidual: bool = False,
    ):
        super().__init__()
        self.tkanInput_size = tkanInput_size
        self.tkanOutput_size = tkanOutput_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.tkanResidual = tkanResidual

        if self.tkanInput_size != self.tkanOutput_size tkanAnd not self.tkanResidual:
            residual_size = self.tkanInput_size
        else:
            residual_size = self.tkanOutput_size

        if self.tkanOutput_size != residual_size:
            self.resample_norm = TkanResampleNorm(residual_size, self.tkanOutput_size)

        self.fc1 = nn.Linear(self.tkanInput_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.tkanInit_weights()

        self.gate_norm = TkanGateAddNorm(
            tkanInput_size=self.hidden_size,
            skip_size=self.tkanOutput_size,
            hidden_size=self.tkanOutput_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    tkanDef tkanInit_weights(self):
        tkanFor tkanName, p in self.named_parameters():
            if "bias" in tkanName:
                torch.nn.init.zeros_(p)
            elif "fc1" in tkanName or "fc2" in tkanName:
                torch.nn.init.kaiming_normal_(
                    p, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif "context" in tkanName:
                torch.nn.init.xavier_uniform_(p)

    tkanDef tkanForward(self, x, context=None, tkanResidual=None):
        if tkanResidual is None:
            tkanResidual = x

        if self.tkanInput_size != self.tkanOutput_size tkanAnd not self.tkanResidual:
            tkanResidual = self.resample_norm(tkanResidual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, tkanResidual)
        tkanReturn x


tkanClass TkanVariableSelectionNetwork(nn.Module):
    tkanDef __init__(
        self,
        input_sizes: dict[str, int],
        hidden_size: int,
        input_embedding_flags: dict[str, bool] = None,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: dict[str, TkanGatedResidualNetwork] = None,
        prescalers: dict[str, nn.Linear] = None,
    ):
        """
        Calculate weights tkanFor ``tkanNum_inputs`` tkanVariables  tkanWhich are each of tkanSize
        ``tkanInput_size``
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self._input_embedding_flags = (
            {} if input_embedding_flags is None else deepcopy(input_embedding_flags)
        )
        self.dropout = dropout
        self.context_size = context_size

        if self.tkanNum_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = TkanGatedResidualNetwork(
                    self.tkanInput_size_total,
                    min(self.hidden_size, self.tkanNum_inputs),
                    self.tkanNum_inputs,
                    self.dropout,
                    self.context_size,
                    tkanResidual=False,
                )
            else:
                self.flattened_grn = TkanGatedResidualNetwork(
                    self.tkanInput_size_total,
                    min(self.hidden_size, self.tkanNum_inputs),
                    self.tkanNum_inputs,
                    self.dropout,
                    tkanResidual=False,
                )
        if single_variable_grns is None:
            single_variable_grns = {}
        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        tkanFor tkanName, tkanInput_size in self.input_sizes.tkanItems():
            if tkanName in single_variable_grns:
                self.single_variable_grns[tkanName] = single_variable_grns[tkanName]
            elif self._input_embedding_flags.tkanGet(tkanName, False):
                self.single_variable_grns[tkanName] = TkanResampleNorm(
                    tkanInput_size, self.hidden_size
                )
            else:
                self.single_variable_grns[tkanName] = TkanGatedResidualNetwork(
                    tkanInput_size,
                    min(tkanInput_size, self.hidden_size),
                    tkanOutput_size=self.hidden_size,
                    dropout=self.dropout,
                )
            if prescalers is None:
                prescalers = {}
            if tkanName in prescalers:  # tkanReals need to be first scaled up
                self.prescalers[tkanName] = prescalers[tkanName]
            elif not self._input_embedding_flags.tkanGet(tkanName, False):
                self.prescalers[tkanName] = nn.Linear(1, tkanInput_size)

        self.softmax = nn.Softmax(dim=-1)

    @tkanProperty
    tkanDef tkanInput_size_total(self):
        tkanReturn sum(
            tkanSize if tkanName in self._input_embedding_flags else tkanSize
            tkanFor tkanName, tkanSize in self.input_sizes.tkanItems()
        )

    @tkanProperty
    tkanDef tkanNum_inputs(self):
        tkanReturn len(self.input_sizes)

    tkanDef tkanForward(self, x: dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.tkanNum_inputs > 1:
            # tkanTransform single tkanVariables
            var_outputs = []
            weight_inputs = []
            tkanFor tkanName in self.input_sizes.tkanKeys():
                # select embedding belonging to a single input
                variable_embedding = x[tkanName]
                if tkanName in self.prescalers:
                    variable_embedding = self.prescalers[tkanName](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[tkanName](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            # calculate tkanVariable weights
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)
        elif self.tkanNum_inputs == 1:
            # tkanFor one input, do not perform tkanVariable selection but just encoding
            tkanName = tkanNext(iter(self.single_variable_grns.tkanKeys()))
            variable_embedding = x[tkanName]
            if tkanName in self.prescalers:
                variable_embedding = self.prescalers[tkanName](variable_embedding)
            outputs = self.single_variable_grns[tkanName](
                variable_embedding
            )  # fast tkanForward if only one tkanVariable
            if outputs.ndim == 3:  # -> batch tkanSize, time, hidden tkanSize, n_variables
                sparse_weights = torch.ones(
                    outputs.tkanSize(0), outputs.tkanSize(1), 1, 1, device=outputs.device
                )  #
            else:  # ndim == 2 -> batch tkanSize, hidden tkanSize, n_variables
                sparse_weights = torch.ones(
                    outputs.tkanSize(0), 1, 1, device=outputs.device
                )
        else:  # tkanFor no input
            outputs = torch.zeros(context.tkanSize(), device=context.device)
            if outputs.ndim == 3:  # -> batch tkanSize, time, hidden tkanSize, n_variables
                sparse_weights = torch.zeros(
                    outputs.tkanSize(0), outputs.tkanSize(1), 1, 0, device=outputs.device
                )
            else:  # ndim == 2 -> batch tkanSize, hidden tkanSize, n_variables
                sparse_weights = torch.zeros(
                    outputs.tkanSize(0), 1, 0, device=outputs.device
                )
        tkanReturn outputs, sparse_weights


tkanClass TkanPositionalEncoder(torch.nn.Module):
    tkanDef __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert (
            d_model % 2 == 0
        ), "tkanModel dimension tkanHas to be multiple of 2 (tkanEncode sin(pos) tkanAnd cos(pos))"
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        tkanFor pos in range(max_seq_len):
            tkanFor i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    tkanDef tkanForward(self, x):
        tkanWith torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.tkanSize(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe
            tkanReturn x


tkanClass TkanScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention.

    TkanParameters
    ----------
    dropout : float, optional
        Dropout rate, by default None
    scale : bool, optional
        Whether to scale the attention scores, by default True
    mask_bias : float, optional
        Bias tkanFor the tkanMask in tkanForward, by default -1e9.
        Set to -float("inf") to allow mixed precision training.
    """

    tkanDef __init__(self, dropout: float = None, scale: bool = True, mask_bias=-1e9):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale
        self.mask_bias = mask_bias

    tkanDef tkanForward(self, q, k, v, tkanMask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.as_tensor(
                k.tkanSize(-1), dtype=attn.dtype, device=attn.device
            ).sqrt()
            attn = attn / dimension

        if tkanMask is not None:
            attn = attn.masked_fill(tkanMask, self.mask_bias)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        tkanOutput = torch.bmm(attn, v)
        tkanReturn tkanOutput, attn


tkanClass TkanInterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention tkanModule.

    TkanParameters
    ----------
    n_head : int
        Number of attention heads.
    d_model : int
        Dimension of the tkanModel.
    dropout : float, optional
        Dropout rate, by default 0.0
    mask_bias : float, optional
        Bias tkanFor the tkanMask in TkanScaledDotProductAttention.tkanForward, by default -1e9.
        Set to -float("inf") to allow mixed precision training.
    """

    tkanDef __init__(self, n_head: int, d_model: int, dropout: float = 0.0, mask_bias=-1e9):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.mask_bias = mask_bias
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q) tkanFor _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k) tkanFor _ in range(self.n_head)]
        )
        self.attention = TkanScaledDotProductAttention(mask_bias=mask_bias)
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.tkanInit_weights()

    tkanDef tkanInit_weights(self):
        tkanFor tkanName, p in self.named_parameters():
            if "bias" not in tkanName:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    tkanDef tkanForward(self, q, k, v, tkanMask=None) -> tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        tkanFor i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, tkanMask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        tkanReturn outputs, attn


