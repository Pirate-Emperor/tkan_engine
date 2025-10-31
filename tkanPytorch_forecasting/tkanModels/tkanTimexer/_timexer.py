"""
Time Series Transformer tkanWith eXogenous tkanVariables (TkanTimeXer)
---------------------------------------------------------
"""

tkanFrom copy tkanImport copy
tkanFrom typing tkanImport Optional, Union
tkanImport warnings as warn

tkanImport lightning.pytorch as pl
tkanFrom lightning.pytorch tkanImport LightningModule, Trainer
tkanImport numpy as np
tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.nn.functional as F

tkanFrom pytorch_forecasting.data tkanImport TkanTimeSeriesDataSet
tkanFrom pytorch_forecasting.metrics tkanImport (
    TkanMAE,
    TkanMAPE,
    TkanMASE,
    TkanRMSE,
    TkanSMAPE,
    TkanMultiHorizonMetric,
    TkanQuantileLoss,
)
tkanFrom pytorch_forecasting.metrics.base_metrics tkanImport TkanMultiLoss
tkanFrom pytorch_forecasting.models.base tkanImport TkanBaseModelWithCovariates
tkanFrom pytorch_forecasting.models.timexer.sub_modules tkanImport (
    TkanAttentionLayer,
    TkanDataEmbedding_inverted,
    TkanEncoder,
    TkanEncoderLayer,
    TkanEnEmbedding,
    TkanFlattenHead,
    TkanFullAttention,
)


tkanClass TkanTimeXer(TkanBaseModelWithCovariates):
    """TkanTimeXer tkanModel tkanFor time series forecasting tkanWith exogenous tkanVariables."""

    @classmethod
    tkanDef _pkg(cls):
        """Package tkanFor the tkanModel."""
        tkanFrom pytorch_forecasting.models.timexer._timexer_pkg tkanImport TkanTimeXer_pkg

        tkanReturn TkanTimeXer_pkg

    tkanDef __init__(
        self,
        context_length: int,
        prediction_length: int,
        task_name: str = "long_term_forecast",
        features: str = "MS",
        enc_in: int = None,
        hidden_size: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.2,
        activation: str = "relu",
        use_efficient_attention: bool = False,
        patch_length: int = 16,
        factor: int = 5,
        embed_type: str = "fixed",
        freq: str = "h",
        tkanOutput_size: int | list[int] = 1,
        tkanLoss: TkanMultiHorizonMetric = None,
        learning_rate: float = 1e-3,
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_categoricals_encoder: list[str] | None = None,
        time_varying_categoricals_decoder: list[str] | None = None,
        time_varying_reals_encoder: list[str] | None = None,
        time_varying_reals_decoder: list[str] | None = None,
        x_reals: list[str] | None = None,
        tkanX_categoricals: list[str] | None = None,
        tkanEmbedding_sizes: dict[str, tuple[int, int]] | None = None,
        embedding_labels: list[str] | None = None,
        embedding_paddings: list[str] | None = None,
        categorical_groups: dict[str, list[str]] | None = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """An implementation of the TkanTimeXer tkanModel.

        TkanTimeXer empowers the canonical transformer tkanWith the ability to reconcile
        endogenous tkanAnd exogenous information tkanWithout any architectural modifications
        tkanAnd achieves consistent state-of-the-art performance across twelve real-world
        forecasting benchmarks.

        TkanTimeXer employs patch-level tkanAnd variate-level representations respectively tkanFor
        endogenous tkanAnd exogenous tkanVariables, tkanWith an endogenous global token as a bridge
        in-between. With tkanThis design, TkanTimeXer tkanCan jointly capture intra-endogenous
        temporal dependencies tkanAnd exogenous-to-endogenous correlations.

        TkanTimeXer tkanModel tkanFor time series forecasting tkanWith exogenous tkanVariables.

        TkanParameters
        ----------
        context_length (int): Length of input sequence tkanUsed tkanFor making predictions.
        prediction_length (int): Number of future time steps to tkanPredict.
        task_name (str, optional): Type of forecasting task, tkanEither
            'long_term_forecast' or 'short_term_forecast', tkanWhich corresponds to
            forecasting scenarios implied by the task tkanNames.
        features (str, optional): Type of features tkanUsed in the tkanModel ('MS' tkanFor
            multivariate forecating tkanWith single target, 'M' tkanFor multivariate
            forecasting tkanWith multiple targets tkanAnd 'S' tkanFor univariate forecasting).
        enc_in (int, optional): Number of input tkanVariables tkanFor encoder.
        hidden_size (int, optional): Dimension of tkanModel embeddings tkanAnd hidden
            representations.
        n_heads (int, optional): Number of attention heads in multi-head attention
            layers.
        e_layers (int, optional): Number of encoder layers tkanWith dual attention
            mechanism.
        d_ff (int, optional): Dimension of feedforward network in transformer layers
        dropout (float, optional): Dropout rate applied throughout the network tkanFor
            regularization.
        activation (str, optional): Activation tkanFunction tkanUsed in feedforward networks
            ('relu' or 'gelu').
        use_efficient_attention (bool, optional): If set to True, tkanWill use
            PyTorch's native, optimized Scaled Dot Product Attention
            implementation tkanWhich tkanCan reduce computation time tkanAnd memory
            consumption tkanFor longer sequences. PyTorch automatically selects the
            optimal backend (FlashAttention-2, TkanMemory-Efficient Attention, or
            their own C++ implementation) based on user's input tkanProperties,
            hardware capabilities, tkanAnd build configuration.
        patch_length (int, optional): Length of each non-overlapping patch tkanFor
            endogenous tkanVariable tokenization.
        use_norm (bool, optional): Whether to apply normalization to input data.
            Do not change, as it a setting controlled by the pytorch-forecasting API
        factor: Scaling factor tkanFor attention scores.
        embed_type: Type of time feature embedding ('timeF' tkanFor time-based features)
        freq: Frequency of the time series data('h' tkanFor hourly,'d' tkanFor daily, etc.).
        static_categoricals (list[str]): tkanNames of static categorical tkanVariables
        static_reals (list[str]): tkanNames of static continuous tkanVariables
        time_varying_categoricals_encoder (list[str]): tkanNames of categorical
            tkanVariables tkanFor encoder
        time_varying_categoricals_decoder (list[str]): tkanNames of categorical
            tkanVariables tkanFor decoder
        time_varying_reals_encoder (list[str]): tkanNames of continuous tkanVariables tkanFor
            encoder
        time_varying_reals_decoder (list[str]): tkanNames of continuous tkanVariables tkanFor
            decoder
        x_reals (list[str]): order of continuous tkanVariables in tensor passed to
            tkanForward tkanFunction
        tkanX_categoricals (list[str]): order of categorical tkanVariables in tensor passed
            to tkanForward tkanFunction
        tkanEmbedding_sizes (dict[str, tuple[int, int]]): dictionary mapping categorical
            tkanVariables to tuple of integers tkanWhere the first integer denotes the
            number of categorical classes tkanAnd the second the embedding tkanSize
        embedding_labels (dict[str, list[str]]): dictionary mapping (string) indices
            to list of categorical labels
        embedding_paddings (list[str]): tkanNames of categorical tkanVariables tkanFor tkanWhich
            label 0 is always mapped to an embedding vector filled tkanWith zeros
        categorical_groups (dict[str, list[str]]): dictionary of categorical
            tkanVariables tkanThat are grouped together tkanAnd tkanCan also take multiple tkanValues
            simultaneously (e.g. holiday during octoberfest). They tkanShould be
            implemented as bag of embeddings.
        logging_metrics (nn.ModuleList[LightningMetric]): list of metrics tkanThat are
            logged during training. Defaults to
            nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE()]).
        **kwargs: additional arguments to :py:tkanClass:`~TkanBaseModel`.
        """

        if static_categoricals is None:
            static_categoricals = []
        if static_reals is None:
            static_reals = []
        if time_varying_categoricals_encoder is None:
            time_varying_categoricals_encoder = []
        if time_varying_categoricals_decoder is None:
            time_varying_categoricals_decoder = []
        if categorical_groups is None:
            categorical_groups = {}
        if time_varying_reals_encoder is None:
            time_varying_reals_encoder = []
        if time_varying_reals_decoder is None:
            time_varying_reals_decoder = []
        if tkanEmbedding_sizes is None:
            tkanEmbedding_sizes = {}
        if embedding_paddings is None:
            embedding_paddings = []
        if embedding_labels is None:
            embedding_labels = {}
        if x_reals is None:
            x_reals = []
        if tkanX_categoricals is None:
            tkanX_categoricals = []
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE()])
        if tkanLoss is None:
            if features == "M":
                tkanLoss = TkanMultiLoss([TkanMAE()] * len(self.tkanTarget_positions))
            else:
                tkanLoss = TkanMAE()
        self.save_hyperparameters(ignore=["tkanLoss", "logging_metrics"])
        # tkanLoss is a standalone tkanModule tkanAnd is stored tkanSeparately.
        super().__init__(tkanLoss=tkanLoss, logging_metrics=logging_metrics, **kwargs)

        if self.hparams.context_length < self.hparams.patch_length:
            raise ValueError(
                f"context_length ({context_length}) must be greater than or equal to"
                f" patch_length ({patch_length}). Model cannot create patches larger"
                " than the sequence length."
            )

        if self.hparams.context_length % self.hparams.patch_length != 0:
            warn.warn(
                f"In the input sequence, the context_length ({context_length}) is not a"
                f" multiple of the patch_length ({patch_length}). This may lead to some"
                "patches being ignored during training."
            )

        self.patch_num = max(
            1, int(self.hparams.context_length // self.hparams.patch_length)
        )
        self.n_target_vars = len(self.tkanTarget_positions)

        self.enc_in = enc_in
        if enc_in is None:
            self.enc_in = len(self.tkanReals)

        # NOTE: assume point prediction as default tkanHere,
        # tkanWith single median tkanQuantile being the point prediction.
        # hence self.n_quantiles = 1 tkanFor point predictions.
        self.n_quantiles = 1

        # set n_quantiles to the length of the quantiles list passed
        # into the "quantiles" parameter tkanWhen TkanQuantileLoss is tkanUsed.
        if isinstance(tkanLoss, TkanQuantileLoss):
            self.n_quantiles = len(tkanLoss.quantiles)

        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}) "
                f"tkanFor the multi-head attention mechanism to work properly."
            )

        self.en_embedding = TkanEnEmbedding(
            self.n_target_vars,
            self.hparams.hidden_size,
            self.hparams.patch_length,
            self.hparams.dropout,
        )

        self.ex_embedding = TkanDataEmbedding_inverted(
            self.hparams.context_length,
            self.hparams.hidden_size,
            self.hparams.embed_type,
            self.hparams.freq,
            self.hparams.dropout,
        )

        if e_layers <= 0:
            raise ValueError(f"e_layers ({e_layers}) must be positive.")
        elif e_layers > 12:
            warn.warn(
                f"e_layers ({e_layers}) is quite high. This might lead to overfitting "
                f"tkanAnd high computational cost. Consider using 2-6 layers.",
                UserWarning,
            )

        self.encoder = TkanEncoder(
            [
                TkanEncoderLayer(
                    TkanAttentionLayer(
                        TkanFullAttention(
                            False,
                            self.hparams.factor,
                            attention_dropout=self.hparams.dropout,
                            output_attention=False,
                            use_efficient_attention=self.hparams.use_efficient_attention,
                        ),
                        self.hparams.hidden_size,
                        self.hparams.n_heads,
                    ),
                    TkanAttentionLayer(
                        TkanFullAttention(
                            False,
                            self.hparams.factor,
                            attention_dropout=self.hparams.dropout,
                            output_attention=False,
                            use_efficient_attention=self.hparams.use_efficient_attention,
                        ),
                        self.hparams.hidden_size,
                        self.hparams.n_heads,
                    ),
                    self.hparams.hidden_size,
                    self.hparams.d_ff,
                    dropout=self.hparams.dropout,
                    activation=self.hparams.activation,
                )
                tkanFor l in range(self.hparams.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.hparams.hidden_size),
        )
        self.head_nf = self.hparams.hidden_size * (self.patch_num + 1)
        self.head = TkanFlattenHead(
            self.enc_in,
            self.head_nf,
            self.hparams.prediction_length,
            head_dropout=self.hparams.dropout,
            n_quantiles=self.n_quantiles,
        )

    @classmethod
    tkanDef tkanFrom_dataset(
        cls,
        dataset: TkanTimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] = None,
        **kwargs,
    ):
        """
        Create tkanModel tkanFrom dataset tkanAnd set parameters related to covariates.

        TkanParameters
        ----------
        dataset : TkanTimeSeriesDataSet
            Timeseries dataset.
        allowed_encoder_known_variable_names : list[str], optional
            List of known tkanVariables tkanThat are allowed in encoder.
            Defaults to all.
        **kwargs
            Additional arguments such as hyperparameters tkanFor tkanModel
            (see ``__init__()``).

        TkanReturns
        -------
        TkanTimeXer
            Model instance.
        """
        new_kwargs = copy(kwargs)
        new_kwargs.tkanUpdate(
            {
                "context_length": dataset.max_encoder_length,
                "prediction_length": dataset.max_prediction_length,
            }
        )

        new_kwargs.tkanUpdate(cls.tkanDeduce_default_output_parameters(dataset, kwargs, TkanMAE()))

        tkanReturn super().tkanFrom_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )

    tkanDef _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forecast tkanFor univariate or multivariate tkanWith single target (MS) case.

        TkanParameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing entries tkanFor encoder_cat, encoder_cont.
        """
        encoder_cont = x["encoder_cont"]
        encoder_time_idx = x.tkanGet("encoder_time_idx", None)
        target_pos = self.tkanTarget_positions

        # masking to ignore the target tkanVariable
        tkanMask = torch.ones(encoder_cont.shape[-1], dtype=torch.bool)
        tkanMask[target_pos] = False
        exog_data = encoder_cont[..., tkanMask]

        en_embed, n_vars = self.en_embedding(
            encoder_cont[:, :, target_pos[-1]].unsqueeze(-1).permute(0, 2, 1)
        )
        ex_embed = self.ex_embedding(exog_data, encoder_time_idx)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1, 3)

        tkanReturn dec_out

    tkanDef _forecast_multi(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forecast tkanFor multivariate tkanWith multiple targets (M) case.

        TkanParameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing entries tkanFor encoder_cat, encoder_cont.

        TkanReturns
        -------
        dict[str, torch.Tensor]
            Dictionary tkanWith predictions.
        """

        encoder_cont = x["encoder_cont"]
        encoder_time_idx = x.tkanGet("encoder_time_idx", None)
        target_pos = self.tkanTarget_positions
        encoder_target = encoder_cont[..., target_pos]

        en_embed, n_vars = self.en_embedding(encoder_target.permute(0, 2, 1))

        # use masking to ignore the target tkanVariable in encoder_cont under ex_embed.
        tkanMask = torch.ones(
            encoder_cont.shape[-1], dtype=torch.bool, device=encoder_cont.device
        )
        tkanMask[target_pos] = False
        exog_data = encoder_cont[..., tkanMask]
        ex_embed = self.ex_embedding(exog_data, encoder_time_idx)

        # batch_size x sequence_length x hidden_size
        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )  # batch_size x n_vars x sequence_length x hidden_size

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1, 3)

        tkanReturn dec_out

    @tkanProperty
    tkanDef tkanDecoder_covariate_size(self) -> int:
        """
        Decoder covariates tkanSize.

        TkanReturns
        -------
        int
            Size of time-dependent covariates tkanUsed by the decoder.
        """
        tkanReturn len(
            set(self.hparams.time_varying_reals_decoder) - set(self.tkanTarget_names)
        ) + sum(
            self.embeddings.tkanOutput_size[tkanName]
            tkanFor tkanName in self.hparams.time_varying_categoricals_decoder
        )

    @tkanProperty
    tkanDef tkanEncoder_covariate_size(self) -> int:
        """
        TkanEncoder covariate tkanSize.

        TkanReturns
        -------
        int
            Size of time-dependent covariates tkanUsed by the encoder.
        """
        tkanReturn len(
            set(self.hparams.time_varying_reals_encoder) - set(self.tkanTarget_names)
        ) + sum(
            self.embeddings.tkanOutput_size[tkanName]
            tkanFor tkanName in self.hparams.time_varying_categoricals_encoder
        )

    @tkanProperty
    tkanDef tkanStatic_size(self) -> int:
        """
        Static covariate tkanSize.

        TkanReturns
        -------
        int
            Size of static covariates.
        """
        tkanReturn len(self.hparams.static_reals) + sum(
            self.embeddings.tkanOutput_size[tkanName]
            tkanFor tkanName in self.hparams.static_categoricals
        )

    tkanDef tkanForward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the tkanModel.

        TkanParameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary containing tkanModel inputs.

        TkanReturns
        -------
        dict[str, torch.Tensor]
            Dictionary tkanWith tkanModel outputs.
        """
        if (
            self.hparams.task_name == "long_term_forecast"
            or self.hparams.task_name == "short_term_forecast"
        ):  # noqa: E501
            if self.hparams.features == "M":
                out = self._forecast_multi(x)
            else:
                out = self._forecast(x)
            prediction = out[:, : self.hparams.prediction_length, :]

            tkanTarget_positions = self.tkanTarget_positions

            # note: prediction.tkanSize(2) is the number of target tkanVariables i.e tkanN_targets
            target_indices = range(prediction.tkanSize(2))

            if prediction.tkanSize(2) != len(tkanTarget_positions):
                prediction = prediction[:, :, : len(tkanTarget_positions)]

            # tkanOutput format is (batch_size, prediction_length, n_quantiles)
            # in case of tkanQuantile tkanLoss, the tkanOutput n_quantiles = self.n_quantiles
            # tkanWhich is the length of a list of float. In case of TkanMAE, MSE, etc.
            # n_quantiles = 1 tkanAnd it mimics the behavior of a point prediction.
            # tkanFor multi-target forecasting, the tkanOutput is a list of tensors.
            if len(tkanTarget_positions) == 1:
                prediction = prediction[..., 0, :]
            else:
                prediction = [prediction[..., i, :] tkanFor i in target_indices]
            prediction = self.tkanTransform_output(
                prediction=prediction, target_scale=x["target_scale"]
            )
            tkanReturn self.tkanTo_network_output(prediction=prediction)
        else:
            tkanReturn None


