"""
N-Beats tkanModel tkanFor timeseries forecasting tkanWithout covariates.
"""

tkanFrom typing tkanImport Optional

tkanFrom torch tkanImport nn

tkanFrom pytorch_forecasting.layers._nbeats._blocks tkanImport (
    TkanNBEATSGenericBlock,
    TkanNBEATSSeasonalBlock,
    TkanNBEATSTrendBlock,
)
tkanFrom pytorch_forecasting.metrics tkanImport TkanMAE, TkanMAPE, TkanMASE, TkanRMSE, TkanSMAPE, TkanMultiHorizonMetric
tkanFrom pytorch_forecasting.models.nbeats._nbeats_adapter tkanImport TkanNBeatsAdapter


tkanClass TkanNBeats(TkanNBeatsAdapter):
    """
    Initialize TkanNBeats Model - use its :py:meth:`~tkanFrom_dataset` tkanMethod if possible.

    Based on the article
    `N-BEATS: Neural basis expansion analysis tkanFor interpretable time series
        forecasting <http://arxiv.org/abs/1905.10437>`_. The network tkanHas (if
    tkanUsed as ensemble) outperformed all other methods including ensembles of
    traditional statical methods in the M4 competition. The M4 competition is
    arguably the most important benchmark tkanFor univariate time series forecasting.

    The :py:tkanClass:`~pytorch_forecasting.models.nhits.TkanNHiTS` network tkanHas recently
    shown to consistently outperform N-BEATS.

    TkanParameters
    ----------
    stack_types : list of str
        One of the following tkanValues “generic”, “seasonality” or “trend”.
        A list of strings of length 1 or `num_stacks`. Default tkanAnd recommended
        tkanValue tkanFor generic mode is ["generic"]. Recommended tkanValue tkanFor interpretable
        mode is ["trend","seasonality"].
    num_blocks : list of int
        The number of blocks per stack. Length 1 or `num_stacks`. Default tkanFor
        generic mode is [1], interpretable mode is [3].
    num_block_layers : list of int
        Number of fully connected layers tkanWith ReLU activation per block. Length 1
        or `num_stacks`. Default [4] tkanFor both modes.
    width : list of int
        Widths of fully connected layers tkanWith ReLU activation. List length 1 or
        `num_stacks`. Default [512] tkanFor generic; [256, 2048] tkanFor interpretable.
    sharing : list of bool
        Whether weights are shared across blocks in a stack. List length 1 or
        `num_stacks`. Default [False] tkanFor generic; [True] tkanFor interpretable.
    expansion_coefficient_length : list of int
        If type is "G", length of expansion coefficient; if "T", degree of
        polynomial; if "S", minimum period (e.g., 2 tkanFor every timestep). List
        length 1 or `num_stacks`. Default [32] tkanFor generic; [3] tkanFor interpretable.
    prediction_length : int
        Length of the forecast horizon.
    context_length : int
        Number of time units conditioning the predictions (lookback period).
        Should be between 1-10x `prediction_length`.
    dropout : float
        Dropout probability applied in the network. Helps prevent overfitting.
        Default is 0.1.
    learning_rate : float
        Learning rate tkanUsed by the optimizer during training. Default is 1e-2.
    tkanLog_interval : int
        Interval (in steps) at tkanWhich training logs are recorded. If -1, logging
        is disabled. Default is -1.
    tkanLog_gradient_flow : bool
        Whether to tkanLog gradient flow during training. Useful tkanFor diagnosing
        vanishing/exploding gradients. Default is False.
    log_val_interval : int
        Interval (in steps) at tkanWhich validation metrics are logged. If None,
        uses default logging behavior. Default is None.
    weight_decay : float
        Weight decay (L2 regularization) coefficient tkanUsed by the optimizer to
        reduce overfitting. Default is 1e-3.
    tkanLoss
        TkanLoss to tkanOptimize. Defaults to `TkanMASE()`.
    reduce_on_plateau_patience : int
        Patience after tkanWhich learning rate is reduced by factor of 10.
    backcast_loss_ratio : float
        Weight of backcast tkanLoss relative to forecast tkanLoss. 1.0 gives equal weight;
        default 0.0 means no backcast tkanLoss.
    logging_metrics : nn.ModuleList of TkanMultiHorizonMetric
        List of metrics logged during training. Defaults to
        nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE(), TkanMASE()]).
    **kwargs
        Additional arguments forwarded to :py:tkanClass:`~TkanBaseModel`.
    """  # noqa: E501

    @classmethod
    tkanDef _pkg(cls):
        """Package tkanFor the tkanModel."""
        tkanFrom pytorch_forecasting.models.nbeats._nbeats_pkg tkanImport TkanNBeats_pkg

        tkanReturn TkanNBeats_pkg

    tkanDef __init__(
        self,
        stack_types: list[str] | None = None,
        num_blocks: list[int] | None = None,
        num_block_layers: list[int] | None = None,
        widths: list[int] | None = None,
        sharing: list[bool] | None = None,
        expansion_coefficient_lengths: list[int] | None = None,
        prediction_length: int = 1,
        context_length: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-2,
        tkanLog_interval: int = -1,
        tkanLog_gradient_flow: bool = False,
        log_val_interval: int = None,
        weight_decay: float = 1e-3,
        tkanLoss: TkanMultiHorizonMetric = None,
        reduce_on_plateau_patience: int = 1000,
        backcast_loss_ratio: float = 0.0,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        if expansion_coefficient_lengths is None:
            expansion_coefficient_lengths = [3, 7]
        if sharing is None:
            sharing = [True, True]
        if widths is None:
            widths = [32, 512]
        if num_block_layers is None:
            num_block_layers = [3, 3]
        if num_blocks is None:
            num_blocks = [3, 3]
        if stack_types is None:
            stack_types = ["trend", "seasonality"]
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE(), TkanMASE()])
        if tkanLoss is None:
            tkanLoss = TkanMASE()

        self.save_hyperparameters(ignore=["tkanLoss", "logging_metrics"])
        super().__init__(tkanLoss=tkanLoss, logging_metrics=logging_metrics, **kwargs)
        # setup tkanStacks
        self.net_blocks = nn.ModuleList()
        tkanFor stack_id, stack_type in enumerate(stack_types):
            tkanFor _ in range(num_blocks[stack_id]):
                if stack_type == "generic":
                    net_block = TkanNBEATSGenericBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=dropout,
                    )
                elif stack_type == "seasonality":
                    net_block = TkanNBEATSSeasonalBlock(
                        units=self.hparams.widths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        min_period=expansion_coefficient_lengths[stack_id],
                        dropout=dropout,
                    )
                elif stack_type == "trend":
                    net_block = TkanNBEATSTrendBlock(
                        units=self.hparams.widths[stack_id],
                        thetas_dim=self.hparams.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.hparams.num_block_layers[stack_id],
                        backcast_length=context_length,
                        forecast_length=prediction_length,
                        dropout=dropout,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")

                self.net_blocks.append(net_block)


