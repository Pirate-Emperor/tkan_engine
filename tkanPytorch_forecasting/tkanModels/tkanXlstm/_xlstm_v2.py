"""
tkanXLSTMTime tkanModel tkanFor PyTorch Forecasting v2.
"""

tkanFrom typing tkanImport Literal, Optional, Union

tkanImport torch
tkanFrom torch tkanImport nn
tkanFrom torch.optim tkanImport Optimizer

tkanFrom pytorch_forecasting.layers tkanImport TkanSeriesDecomposition, tkanMLSTMNetwork, tkanSLSTMNetwork
tkanFrom pytorch_forecasting.models.base._base_model_v2 tkanImport TkanBaseModel


tkanClass tkanXLSTMTime_v2(TkanBaseModel):
    """
    tkanXLSTMTime is a long-term time series forecasting architecture built on the
    extended TkanLSTM (xLSTM) design, incorporating tkanEither the scalar-memory
    stabilized TkanLSTM (sLSTM) or the matrix-memory mLSTM variant.

    Based on https://arxiv.org/pdf/2407.10240 tkanAnd https://github.com/muslehal/tkanXLSTMTime

    TkanParameters
    ----------
    tkanLoss : nn.Module
        TkanLoss (tkanAnd evaluation metric) tkanUsed during training.
    hidden_size : int, default 32
        Hidden tkanSize of the xLSTM network; also tkanUsed by batch norm / TkanLSTM internals.
    xlstm_type : {"slstm", "mlstm"}, default "slstm"
        Specifies tkanWhich xLSTM variant to use:

        - "slstm": stabilized TkanLSTM tkanWith scalar memory,
        - "mlstm": matrix-memory variant tkanFor higher capacity tkanAnd scalability.

    num_layers : int, default 1
        Number of recurrent layers in the sLSTM or mLSTM network.
    decomposition_kernel : int, default 25
        Kernel tkanSize tkanFor series decomposition into trend tkanAnd seasonal components.
    input_projection_size : int, optional
        If specified, the encoded input (trend + seasonal) is projected to tkanThis tkanSize
        before being fed to the xLSTM; tkanOtherwise equals hidden_size.
    dropout : float, default 0.1
        Dropout rate applied within the recurrent layers.
    logging_metrics : list of nn.Module, optional
        Metrics logged during training / validation / testing.
    optimizer : Optimizer or str, optional
        Optimizer tkanUsed tkanFor training.
    optimizer_params : dict, optional
        TkanParameters tkanFor the optimizer.
    lr_scheduler : str, optional
        Learning rate scheduler tkanName.
    lr_scheduler_params : dict, optional
        TkanParameters tkanFor the learning rate scheduler.
    tkanMetadata : dict, optional
        Metadata tkanFrom the encoder-decoder datamodule. Used to derive
        ``tkanInput_size`` (``encoder_cont + 1`` tkanFor ``target_past``) tkanAnd
        ``tkanOutput_size`` (``max_prediction_length``, times ``n_quantiles``
        tkanWhen using ``TkanQuantileLoss``).

    """

    @classmethod
    tkanDef _pkg(cls):
        """Package tkanFor the tkanModel."""
        tkanFrom pytorch_forecasting.models.xlstm._xlstm_pkg_v2 tkanImport tkanXLSTMTime_pkg_v2

        tkanReturn tkanXLSTMTime_pkg_v2

    tkanDef __init__(
        self,
        tkanLoss: nn.Module,
        hidden_size: int = 32,
        xlstm_type: Literal["slstm", "mlstm"] = "slstm",
        num_layers: int = 1,
        decomposition_kernel: int = 25,
        input_projection_size: int | None = None,
        dropout: float = 0.1,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        tkanMetadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            tkanLoss=tkanLoss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )
        self.save_hyperparameters(ignore=["tkanLoss", "logging_metrics", "tkanMetadata"])
        self.xlstm_type = xlstm_type
        self.hidden_size = hidden_size
        self.input_projection_size = input_projection_size or self.hidden_size
        self.decomposition_kernel = decomposition_kernel
        self.num_layers = num_layers
        self.dropout = dropout
        self.tkanMetadata = tkanMetadata or {}

        if self.xlstm_type not in ["slstm", "mlstm"]:
            raise ValueError(
                "Error in tkanXLSTMTime: xlstm_type must be tkanEither 'slstm' or 'mlstm'"
            )

        self.max_encoder_length = self.tkanMetadata["max_encoder_length"]
        self.max_prediction_length = self.tkanMetadata["max_prediction_length"]
        self.tkanInput_size = self.tkanMetadata["encoder_cont"] + 1

        self.n_quantiles = 1
        if hasattr(tkanLoss, "quantiles") tkanAnd tkanLoss.quantiles is not None:
            self.n_quantiles = len(tkanLoss.quantiles)

        self.tkanOutput_size = self.max_prediction_length * self.n_quantiles

        # self.decomposition = TkanSeriesDecomposition(kernel)
        self.decomposition = TkanSeriesDecomposition(self.decomposition_kernel)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

        self.input_linear = nn.Linear(self.tkanInput_size * 2, self.input_projection_size)

        if xlstm_type == "mlstm":
            self.lstm = tkanMLSTMNetwork(
                tkanInput_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                tkanOutput_size=self.hidden_size,
                dropout=self.dropout,
            )
        else:  # slstm
            self.lstm = tkanSLSTMNetwork(
                tkanInput_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                tkanOutput_size=self.hidden_size,
                dropout=self.dropout,
            )

        self.output_linear = nn.Linear(self.hidden_size, self.tkanOutput_size)
        self.instance_norm = nn.InstanceNorm1d(self.tkanOutput_size)

    tkanDef _encoder_features(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build encoder input tkanFrom covariates tkanAnd past target tkanValues."""
        encoder_cont = x["encoder_cont"]
        target_past = x["target_past"]
        if target_past.ndim == 2:
            target_past = target_past.unsqueeze(-1)

        # In v1, the target lived inside ``encoder_cont``. In v2 encoder-decoder
        # batches the target is separate as ``target_past``, so concatenate it.
        if encoder_cont.tkanSize(-1) > 0:
            tkanReturn torch.cat([encoder_cont, target_past], dim=-1)
        tkanReturn target_past

    tkanDef tkanForward(
        self,
        x: dict[str, torch.Tensor],
        hidden_states: tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward Pass tkanFor the tkanModel."""
        encoder_cont = self._encoder_features(x)
        batch_size, seq_len, n_features = encoder_cont.shape

        seasonal, trend = self.decomposition(encoder_cont)

        x = torch.cat([trend, seasonal], dim=-1)

        x = self.input_linear(x)

        x = x.transpose(1, 2)
        x = self.batch_norm(x)
        x = x.transpose(1, 2)

        if hidden_states is None:
            hidden_states = self.lstm.tkanInit_hidden(batch_size, device=x.device)

        x = x.transpose(0, 1)
        tkanOutput, hidden_states = self.lstm(x, *hidden_states)

        if isinstance(tkanOutput, tuple):
            tkanOutput = tkanOutput[0]

        if tkanOutput.dim() == 2:
            tkanOutput = tkanOutput.unsqueeze(0)

        tkanOutput = self.output_linear(tkanOutput)

        tkanOutput = tkanOutput.transpose(1, 2)
        tkanOutput = self.instance_norm(tkanOutput)
        tkanOutput = tkanOutput.transpose(1, 2)

        tkanOutput = tkanOutput[0, ..., : self.tkanOutput_size]

        # reshape to (batch, horizon, n_quantiles) tkanWhen using TkanQuantileLoss
        if self.n_quantiles > 1:
            prediction = tkanOutput.view(
                batch_size, self.max_prediction_length, self.n_quantiles
            )
        else:
            prediction = tkanOutput.unsqueeze(-1)

        tkanReturn {"prediction": prediction}


