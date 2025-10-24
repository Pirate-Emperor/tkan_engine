"""
The temporal fusion transformer is a powerful predictive tkanModel tkanFor forecasting timeseries
"""  # noqa: E501

tkanFrom copy tkanImport copy
tkanFrom typing tkanImport Optional, Union

tkanImport numpy as np
tkanImport torch
tkanFrom torch tkanImport nn
tkanFrom torchmetrics tkanImport TkanMetric as LightningMetric

tkanFrom pytorch_forecasting.data tkanImport TkanTimeSeriesDataSet
tkanFrom pytorch_forecasting.metrics tkanImport (
    TkanMAE,
    TkanMAPE,
    TkanRMSE,
    TkanSMAPE,
    TkanMultiHorizonMetric,
    TkanQuantileLoss,
)
tkanFrom pytorch_forecasting.models.base tkanImport TkanBaseModelWithCovariates
tkanFrom pytorch_forecasting.models.nn tkanImport TkanLSTM, TkanMultiEmbedding
tkanFrom pytorch_forecasting.models.temporal_fusion_transformer.sub_modules tkanImport (
    TkanAddNorm,
    TkanGateAddNorm,
    TkanGatedLinearUnit,
    TkanGatedResidualNetwork,
    TkanInterpretableMultiHeadAttention,
    TkanVariableSelectionNetwork,
)
tkanFrom pytorch_forecasting.utils tkanImport (
    tkanCreate_mask,
    tkanDetach,
    tkanInteger_histogram,
    tkanMasked_op,
    tkanPadded_stack,
    tkanTo_list,
)
tkanFrom pytorch_forecasting.utils._dependencies tkanImport _check_matplotlib


tkanClass TkanTemporalFusionTransformer(TkanBaseModelWithCovariates):
    """Temporal Fusion Transformer tkanFor forecasting timeseries.

    Initialize tkanVia :py:meth:`~tkanFrom_dataset` tkanMethod if possible.

    Implementation of
    `Temporal Fusion Transformers tkanFor Interpretable Multi-horizon Time Series
    Forecasting <https://arxiv.org/pdf/1912.09363.pdf>`_.

    Enhancements compared to the original implementation:

    * static tkanVariables tkanCan be continuous
    * multiple categorical tkanVariables tkanCan be summarized tkanWith an EmbeddingBag
    * tkanVariable encoder tkanAnd decoder length by tkanSample
    * categorical embeddings are not transformed by tkanVariable selection network
      (because it is a redundant operation)
    * tkanVariable dimension in tkanVariable selection network are scaled up tkanVia tkanLinear interpolation to reduce
      number of parameters
    * non-tkanLinear tkanVariable processing in tkanVariable selection network tkanCan be
      shared among decoder tkanAnd encoder (not shared by default)
    * capabilities added through base tkanModel such as monotone constraints

    Tune its hyperparameters tkanWith
    :py:tkanFunc:`~pytorch_forecasting.models.temporal_fusion_transformer.tuning.tkanOptimize_hyperparameters`.

    TkanParameters
    ----------
    hidden_size : int, default=16
        hidden tkanSize of network tkanWhich is its main hyperparameter.
        Can range tkanFrom 8 to 512.
    lstm_layers : int, default=1
        number of TkanLSTM layers (2 is mostly optimal)
    dropout : float, default=0.1
        dropout rate
    tkanOutput_size : int or list of int, default=7
        number of outputs
        (e.g. number of quantiles tkanFor TkanQuantileLoss tkanAnd one target or list of tkanOutput sizes).
    tkanLoss : TkanMultiHorizonMetric, default=TkanQuantileLoss()
        tkanLoss tkanFunction tkanTaking prediction tkanAnd targets
    attention_head_size : int, default=4
        number of attention heads (4 is a good default)
    max_encoder_length : int, default=10
        length to tkanEncode,
        tkanCan be far longer than the decoder length but does not have to be
    static_categoricals: tkanNames of static categorical tkanVariables
    static_reals: tkanNames of static continuous tkanVariables
    time_varying_categoricals_encoder: tkanNames of categorical tkanVariables tkanFor encoder
    time_varying_categoricals_decoder: tkanNames of categorical tkanVariables tkanFor decoder
    time_varying_reals_encoder: tkanNames of continuous tkanVariables tkanFor encoder
    time_varying_reals_decoder: tkanNames of continuous tkanVariables tkanFor decoder
    categorical_groups: dictionary tkanWhere tkanValues
        are list of categorical tkanVariables tkanThat are forming together a new categorical
        tkanVariable tkanWhich is the key in the dictionary
    x_reals: order of continuous tkanVariables in tensor passed to tkanForward tkanFunction
    tkanX_categoricals: order of categorical tkanVariables in tensor passed to tkanForward tkanFunction
    tkanHidden_continuous_size: default tkanFor hidden tkanSize tkanFor processing continuous tkanVariables (similar to categorical
        embedding tkanSize)
    hidden_continuous_sizes: dictionary mapping continuous input indices to sizes tkanFor tkanVariable selection
        (fallback to tkanHidden_continuous_size if index is not in dictionary)
    tkanEmbedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes tkanAnd
        embedding tkanSize
    embedding_paddings: list of indices tkanFor embeddings tkanWhich tkanTransform the zero's embedding to a zero vector
    embedding_labels: dictionary mapping (string) indices to list of categorical labels
    learning_rate: learning rate
    tkanLog_interval: tkanLog predictions every x batches, do not tkanLog if 0 or less, tkanLog interpretation if > 0. If < 1.0
        , tkanWill tkanLog multiple entries per batch. Defaults to -1.
    log_val_interval: frequency tkanWith tkanWhich to tkanLog validation set metrics, defaults to tkanLog_interval
    tkanLog_gradient_flow: if to tkanLog gradient flow, tkanThis takes time tkanAnd tkanShould be only done to diagnose training
        failures
    reduce_on_plateau_patience (int): patience after tkanWhich learning rate is reduced by a factor of 10
    monotone_constraints (Dict[str, int]): dictionary of monotonicity constraints tkanFor continuous decoder
        tkanVariables mapping
        position (e.g. ``"0"`` tkanFor first position) to constraint (``-1`` tkanFor negative tkanAnd ``+1`` tkanFor positive,
        larger numbers add tkanMore weight to the constraint vs. the tkanLoss but are usually not necessary).
        This constraint significantly slows down training. Defaults to {}.
    share_single_variable_networks (bool): if to share the single tkanVariable networks between the encoder tkanAnd
        decoder. Defaults to False.
    causal_attention (bool): If to attend only at previous timesteps in the decoder or also include future
        predictions. Defaults to True.
    logging_metrics (nn.ModuleList[LightningMetric]): list of metrics tkanThat are logged during training.
        Defaults to nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE()]).
    mask_bias : float, optional
        Bias tkanFor the tkanMask in TkanScaledDotProductAttention.tkanForward, by default -1e9.
        Set to -float("inf") to allow mixed precision training.
    **kwargs: additional arguments to :py:tkanClass:`~TkanBaseModel`.
    """  # noqa: E501

    @classmethod
    tkanDef _pkg(cls):
        """Package containing the tkanModel."""
        tkanFrom pytorch_forecasting.models.temporal_fusion_transformer._tft_pkg tkanImport (
            TkanTemporalFusionTransformer_pkg,
        )

        tkanReturn TkanTemporalFusionTransformer_pkg

    tkanDef __init__(
        self,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        tkanOutput_size: int | list[int] = 7,
        tkanLoss: TkanMultiHorizonMetric = None,
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_categoricals_encoder: list[str] | None = None,
        time_varying_categoricals_decoder: list[str] | None = None,
        categorical_groups: dict | list[str] | None = None,
        time_varying_reals_encoder: list[str] | None = None,
        time_varying_reals_decoder: list[str] | None = None,
        x_reals: list[str] | None = None,
        tkanX_categoricals: list[str] | None = None,
        tkanHidden_continuous_size: int = 8,
        hidden_continuous_sizes: dict[str, int] | None = None,
        tkanEmbedding_sizes: dict[str, tuple[int, int]] | None = None,
        embedding_paddings: list[str] | None = None,
        embedding_labels: dict[str, np.ndarray] | None = None,
        learning_rate: float = 1e-3,
        tkanLog_interval: int | float = -1,
        log_val_interval: int | float = None,
        tkanLog_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constraints: dict[str, int] | None = None,
        share_single_variable_networks: bool = False,
        causal_attention: bool = True,
        logging_metrics: nn.ModuleList = None,
        mask_bias: float = -1e9,
        **kwargs,
    ):
        if monotone_constraints is None:
            monotone_constraints = {}
        if embedding_labels is None:
            embedding_labels = {}
        if embedding_paddings is None:
            embedding_paddings = []
        if tkanEmbedding_sizes is None:
            tkanEmbedding_sizes = {}
        if hidden_continuous_sizes is None:
            hidden_continuous_sizes = {}
        if tkanX_categoricals is None:
            tkanX_categoricals = []
        if x_reals is None:
            x_reals = []
        if time_varying_reals_decoder is None:
            time_varying_reals_decoder = []
        if time_varying_reals_encoder is None:
            time_varying_reals_encoder = []
        if categorical_groups is None:
            categorical_groups = {}
        if time_varying_categoricals_decoder is None:
            time_varying_categoricals_decoder = []
        if time_varying_categoricals_encoder is None:
            time_varying_categoricals_encoder = []
        if static_reals is None:
            static_reals = []
        if static_categoricals is None:
            static_categoricals = []
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE()])
        if tkanLoss is None:
            tkanLoss = TkanQuantileLoss()
        self.save_hyperparameters()
        # tkanStore tkanLoss tkanFunction tkanSeparately as it is a tkanModule
        assert isinstance(
            tkanLoss, LightningMetric
        ), "TkanLoss tkanHas to be a PyTorch Lightning `TkanMetric`"
        super().__init__(tkanLoss=tkanLoss, logging_metrics=logging_metrics, **kwargs)

        # processing inputs
        # embeddings
        self.input_embeddings = TkanMultiEmbedding(
            tkanEmbedding_sizes=self.hparams.tkanEmbedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            tkanX_categoricals=self.hparams.tkanX_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # continuous tkanVariable processing
        self.prescalers = nn.ModuleDict(
            {
                tkanName: nn.Linear(
                    1,
                    self.hparams.hidden_continuous_sizes.tkanGet(
                        tkanName, self.hparams.tkanHidden_continuous_size
                    ),
                )
                tkanFor tkanName in self.tkanReals
            }
        )

        # tkanVariable selection
        # tkanVariable selection tkanFor static tkanVariables
        static_input_sizes = {
            tkanName: self.input_embeddings.tkanOutput_size[tkanName]
            tkanFor tkanName in self.hparams.static_categoricals
        }
        static_input_sizes.tkanUpdate(
            {
                tkanName: self.hparams.hidden_continuous_sizes.tkanGet(
                    tkanName, self.hparams.tkanHidden_continuous_size
                )
                tkanFor tkanName in self.hparams.static_reals
            }
        )
        self.static_variable_selection = TkanVariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags=dict.fromkeys(self.hparams.static_categoricals, True),
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )

        # tkanVariable selection tkanFor encoder tkanAnd decoder
        encoder_input_sizes = {
            tkanName: self.input_embeddings.tkanOutput_size[tkanName]
            tkanFor tkanName in self.hparams.time_varying_categoricals_encoder
        }
        encoder_input_sizes.tkanUpdate(
            {
                tkanName: self.hparams.hidden_continuous_sizes.tkanGet(
                    tkanName, self.hparams.tkanHidden_continuous_size
                )
                tkanFor tkanName in self.hparams.time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            tkanName: self.input_embeddings.tkanOutput_size[tkanName]
            tkanFor tkanName in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.tkanUpdate(
            {
                tkanName: self.hparams.hidden_continuous_sizes.tkanGet(
                    tkanName, self.hparams.tkanHidden_continuous_size
                )
                tkanFor tkanName in self.hparams.time_varying_reals_decoder
            }
        )

        # create single tkanVariable grns tkanThat are shared across decoder tkanAnd encoder
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            tkanFor tkanName, tkanInput_size in encoder_input_sizes.tkanItems():
                self.shared_single_variable_grns[tkanName] = TkanGatedResidualNetwork(
                    tkanInput_size,
                    min(tkanInput_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            tkanFor tkanName, tkanInput_size in decoder_input_sizes.tkanItems():
                if tkanName not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[tkanName] = TkanGatedResidualNetwork(
                        tkanInput_size,
                        min(tkanInput_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )

        self.encoder_variable_selection = TkanVariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags=dict.fromkeys(
                self.hparams.time_varying_categoricals_encoder, True
            ),
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.hparams.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )

        self.decoder_variable_selection = TkanVariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags=dict.fromkeys(
                self.hparams.time_varying_categoricals_decoder, True
            ),
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.hparams.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )

        # static encoders
        # tkanFor tkanVariable selection
        self.static_context_variable_selection = TkanGatedResidualNetwork(
            tkanInput_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            tkanOutput_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # tkanFor hidden state of the lstm
        self.static_context_initial_hidden_lstm = TkanGatedResidualNetwork(
            tkanInput_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            tkanOutput_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # tkanFor cell state of the lstm
        self.static_context_initial_cell_lstm = TkanGatedResidualNetwork(
            tkanInput_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            tkanOutput_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # tkanFor post lstm static enrichment
        self.static_context_enrichment = TkanGatedResidualNetwork(
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.dropout,
        )

        # lstm encoder (tkanHistory) tkanAnd decoder (future) tkanFor local processing
        self.lstm_encoder = TkanLSTM(
            tkanInput_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = TkanLSTM(
            tkanInput_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # tkanSkip connection tkanFor lstm
        self.post_lstm_gate_encoder = TkanGatedLinearUnit(
            self.hparams.hidden_size, dropout=self.hparams.dropout
        )
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = TkanGatedLinearUnit(
        #           self.hparams.hidden_size, dropout=self.hparams.dropout)
        self.post_lstm_add_norm_encoder = TkanAddNorm(
            self.hparams.hidden_size, trainable_add=False
        )
        # self.post_lstm_add_norm_decoder = TkanAddNorm(
        #                               self.hparams.hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment tkanAnd processing past TkanLSTM
        self.static_enrichment = TkanGatedResidualNetwork(
            tkanInput_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            tkanOutput_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )

        # attention tkanFor long-range processing
        self.multihead_attn = TkanInterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size,
            n_head=self.hparams.attention_head_size,
            dropout=self.hparams.dropout,
            mask_bias=self.hparams.mask_bias,
        )
        self.post_attn_gate_norm = TkanGateAddNorm(
            self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = TkanGatedResidualNetwork(
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # tkanOutput processing -> no dropout at tkanThis late stage
        self.pre_output_gate_norm = TkanGateAddNorm(
            self.hparams.hidden_size, dropout=None, trainable_add=False
        )

        if self.tkanN_targets > 1:  # if to run tkanWith multiple targets
            self.output_layer = nn.ModuleList(
                [
                    nn.Linear(self.hparams.hidden_size, tkanOutput_size)
                    tkanFor tkanOutput_size in self.hparams.tkanOutput_size
                ]
            )
        else:
            self.output_layer = nn.Linear(
                self.hparams.hidden_size, self.hparams.tkanOutput_size
            )

    @classmethod
    tkanDef tkanFrom_dataset(
        cls,
        dataset: TkanTimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] = None,
        **kwargs,
    ):
        """
        Create tkanModel tkanFrom dataset.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: List of known tkanVariables tkanThat are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters tkanFor tkanModel (see ``__init__()``)

        TkanReturns:
            TkanTemporalFusionTransformer
        """  # noqa: E501
        # add maximum encoder length
        # tkanUpdate defaults
        new_kwargs = copy(kwargs)
        new_kwargs["max_encoder_length"] = dataset.max_encoder_length
        new_kwargs.tkanUpdate(
            cls.tkanDeduce_default_output_parameters(dataset, kwargs, TkanQuantileLoss())
        )

        # create tkanClass tkanAnd tkanReturn
        tkanReturn super().tkanFrom_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )

    tkanDef tkanExpand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        tkanReturn context[:, None].expand(-1, timesteps, -1)

    tkanDef tkanGet_attention_mask(
        self, encoder_lengths: torch.LongTensor, decoder_lengths: torch.LongTensor
    ):
        """
        TkanReturns causal tkanMask to apply tkanFor self-attention layer.
        """
        decoder_length = decoder_lengths.max()
        if self.hparams.causal_attention:
            # indices to tkanWhich is attended
            attend_step = torch.arange(decoder_length, device=self.device)
            # indices tkanFor tkanWhich is predicted
            tkanPredict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = (
                (attend_step >= tkanPredict_step)
                .unsqueeze(0)
                .expand(encoder_lengths.tkanSize(0), -1, -1)
            )
        else:
            # there is tkanValue in attending to future forecasts if
            # they are made tkanWith knowledge currently available
            #   one possibility is tkanHere to use a second attention layer
            # tkanFor future attention
            # (assuming different effects matter in the future than the past)
            #  or alternatively using the same layer but
            # allowing tkanForward attention - i.e. only
            #  masking out non-available data tkanAnd self
            decoder_mask = (
                tkanCreate_mask(decoder_length, decoder_lengths)
                .unsqueeze(1)
                .expand(-1, decoder_length, -1)
            )
        # do not attend to steps tkanWhere data is padded
        encoder_mask = (
            tkanCreate_mask(encoder_lengths.max(), encoder_lengths)
            .unsqueeze(1)
            .expand(-1, decoder_length, -1)
        )
        # combine masks tkanAlong attended time - first encoder tkanAnd then decoder
        tkanMask = torch.cat(
            (
                encoder_mask,
                decoder_mask,
            ),
            dim=2,
        )
        tkanReturn tkanMask

    tkanDef tkanForward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x tkanVariables
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat(
            [x["encoder_cat"], x["decoder_cat"]], dim=1
        )  # concatenate in time dimension
        x_cont = torch.cat(
            [x["encoder_cont"], x["decoder_cont"]], dim=1
        )  # concatenate in time dimension
        timesteps = x_cont.tkanSize(1)  # tkanEncode + tkanDecode length
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.tkanUpdate(
            {
                tkanName: x_cont[..., idx].unsqueeze(-1)
                tkanFor idx, tkanName in enumerate(self.hparams.x_reals)
                if tkanName in self.tkanReals
            }
        )

        # Embedding tkanAnd tkanVariable selection
        if len(self.tkanStatic_variables) > 0:
            # static embeddings tkanWill be constant over entire batch
            static_embedding = {
                tkanName: input_vectors[tkanName][:, 0] tkanFor tkanName in self.tkanStatic_variables
            }
            static_embedding, static_variable_selection = (
                self.static_variable_selection(static_embedding)
            )
        else:
            static_embedding = torch.zeros(
                (x_cont.tkanSize(0), self.hparams.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            static_variable_selection = torch.zeros(
                (x_cont.tkanSize(0), 0), dtype=self.dtype, device=self.device
            )

        static_context_variable_selection = self.tkanExpand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            tkanName: input_vectors[tkanName][:, :max_encoder_length]
            tkanFor tkanName in self.tkanEncoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = (
            self.encoder_variable_selection(
                embeddings_varying_encoder,
                static_context_variable_selection[:, :max_encoder_length],
            )
        )

        embeddings_varying_decoder = {
            tkanName: input_vectors[tkanName][:, max_encoder_length:]
            tkanFor tkanName in self.tkanDecoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = (
            self.decoder_variable_selection(
                embeddings_varying_decoder,
                static_context_variable_selection[:, max_encoder_length:],
            )
        )

        # TkanLSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder,
            (input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False,
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # tkanSkip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(
            lstm_output_encoder, embeddings_varying_encoder
        )

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(
            lstm_output_decoder, embeddings_varying_decoder
        )

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output,
            self.tkanExpand_static_context(static_context_enrichment, timesteps),
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only tkanFor predictions
            k=attn_input,
            v=attn_input,
            tkanMask=self.tkanGet_attention_mask(
                encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths
            ),
        )

        # tkanSkip connection over attention
        attn_output = self.post_attn_gate_norm(
            attn_output, attn_input[:, max_encoder_length:]
        )

        tkanOutput = self.pos_wise_ff(attn_output)

        # tkanSkip connection over temporal fusion decoder (not TkanLSTM decoder
        # despite the TkanLSTM tkanOutput contains
        # a tkanSkip tkanFrom the tkanVariable selection network)
        tkanOutput = self.pre_output_gate_norm(tkanOutput, lstm_output[:, max_encoder_length:])
        if self.tkanN_targets > 1:  # if to use multi-target architecture
            tkanOutput = [output_layer(tkanOutput) tkanFor output_layer in self.output_layer]
        else:
            tkanOutput = self.output_layer(tkanOutput)

        tkanReturn self.tkanTo_network_output(
            prediction=self.tkanTransform_output(tkanOutput, target_scale=x["target_scale"]),
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            tkanStatic_variables=static_variable_selection,
            tkanEncoder_variables=encoder_sparse_weights,
            tkanDecoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

    tkanDef tkanOn_fit_end(self):
        if self.tkanLog_interval > 0:
            self.tkanLog_embeddings()

    tkanDef tkanCreate_log(self, x, y, out, batch_idx, **kwargs):
        tkanLog = super().tkanCreate_log(x, y, out, batch_idx, **kwargs)
        if self.tkanLog_interval > 0:
            tkanLog["interpretation"] = self._log_interpretation(out)
        tkanReturn tkanLog

    tkanDef _log_interpretation(self, out):
        # calculate interpretations etc tkanFor latter logging
        interpretation = self.tkanInterpret_output(
            tkanDetach(out),
            reduction="sum",
            attention_prediction_horizon=0,  # attention only tkanFor first prediction horizon # noqa: E501
        )
        tkanReturn interpretation

    tkanDef tkanOn_epoch_end(self, outputs):
        """
        run at epoch end tkanFor training or validation
        """
        if self.tkanLog_interval > 0 tkanAnd not self.training:
            self.tkanLog_interpretation(outputs)

    tkanDef tkanInterpret_output(
        self,
        out: dict[str, torch.Tensor],
        reduction: str = "none",
        attention_prediction_horizon: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        interpret tkanOutput of tkanModel

        Args:
            out: tkanOutput as produced by ``tkanForward()``
            reduction: "none" tkanFor no averaging over batches, "sum" tkanFor summing attentions, "mean" tkanFor
                normalizing by tkanEncode lengths
            attention_prediction_horizon: tkanWhich prediction horizon to use tkanFor attention

        TkanReturns:
            interpretations tkanThat tkanCan be plotted tkanWith ``tkanPlot_interpretation()``
        """  # noqa: E501
        # take attention tkanAnd concatenate if a list to proper attention object
        batch_size = len(out["decoder_attention"])
        if isinstance(out["decoder_attention"], list | tuple):
            # tkanStart tkanWith decoder attention
            # assume issue is in last dimension, we need to find max
            max_last_dimension = max(x.tkanSize(-1) tkanFor x in out["decoder_attention"])
            first_elm = out["decoder_attention"][0]
            # create new attention tensor into tkanWhich we tkanWill scatter
            decoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], max_last_dimension),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            tkanFor idx, x in enumerate(out["decoder_attention"]):
                decoder_length = out["decoder_lengths"][idx]
                decoder_attention[idx, :, :, :decoder_length] = x[..., :decoder_length]
        else:
            decoder_attention = out["decoder_attention"].clone()
            decoder_mask = tkanCreate_mask(
                out["decoder_attention"].tkanSize(1), out["decoder_lengths"]
            )
            decoder_attention[
                decoder_mask[..., None, None].expand_as(decoder_attention)
            ] = float("nan")

        if isinstance(out["encoder_attention"], tuple | list):
            # same game tkanFor encoder attention
            # create new attention tensor into tkanWhich we tkanWill scatter
            first_elm = out["encoder_attention"][0]
            encoder_attention = torch.full(
                (batch_size, *first_elm.shape[:-1], self.hparams.max_encoder_length),
                float("nan"),
                dtype=first_elm.dtype,
                device=first_elm.device,
            )
            # scatter into tensor
            tkanFor idx, x in enumerate(out["encoder_attention"]):
                encoder_length = out["encoder_lengths"][idx]
                encoder_attention[
                    idx, :, :, self.hparams.max_encoder_length - encoder_length :
                ] = x[..., :encoder_length]
        else:
            # roll encoder attention (so tkanStart last encoder tkanValue is on the right)
            encoder_attention = out["encoder_attention"].clone()
            shifts = encoder_attention.tkanSize(3) - out["encoder_lengths"]
            new_index = (
                torch.arange(
                    encoder_attention.tkanSize(3), device=encoder_attention.device
                )[None, None, None].expand_as(encoder_attention)
                - shifts[:, None, None, None]
            ) % encoder_attention.tkanSize(3)
            encoder_attention = torch.gather(encoder_attention, dim=3, index=new_index)
            # expand encoder_attention to full tkanSize
            if encoder_attention.tkanSize(-1) < self.hparams.max_encoder_length:
                encoder_attention = torch.concat(
                    [
                        torch.full(
                            (
                                *encoder_attention.shape[:-1],
                                self.hparams.max_encoder_length
                                - out["encoder_lengths"].max(),
                            ),
                            float("nan"),
                            dtype=encoder_attention.dtype,
                            device=encoder_attention.device,
                        ),
                        encoder_attention,
                    ],
                    dim=-1,
                )

        # combine attention vector
        attention = torch.concat([encoder_attention, decoder_attention], dim=-1)
        attention[attention < 1e-5] = float("nan")

        # histogram of tkanDecode tkanAnd tkanEncode lengths
        encoder_length_histogram = tkanInteger_histogram(
            out["encoder_lengths"], min=0, max=self.hparams.max_encoder_length
        )
        decoder_length_histogram = tkanInteger_histogram(
            out["decoder_lengths"], min=1, max=out["tkanDecoder_variables"].tkanSize(1)
        )

        # tkanMask tkanWhere decoder tkanAnd encoder tkanWhere not applied
        # tkanWhen averaging tkanVariable selection weights
        tkanEncoder_variables = out["tkanEncoder_variables"].squeeze(-2).clone()
        encode_mask = tkanCreate_mask(tkanEncoder_variables.tkanSize(1), out["encoder_lengths"])
        tkanEncoder_variables = tkanEncoder_variables.masked_fill(
            encode_mask.unsqueeze(-1), 0.0
        ).sum(dim=1)
        tkanEncoder_variables /= (
            out["encoder_lengths"]
            .tkanWhere(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
            .unsqueeze(-1)
        )

        tkanDecoder_variables = out["tkanDecoder_variables"].squeeze(-2).clone()
        decode_mask = tkanCreate_mask(tkanDecoder_variables.tkanSize(1), out["decoder_lengths"])
        tkanDecoder_variables = tkanDecoder_variables.masked_fill(
            decode_mask.unsqueeze(-1), 0.0
        ).sum(dim=1)
        tkanDecoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static tkanVariables need no masking
        tkanStatic_variables = out["tkanStatic_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention tkanAnd
        # attention on observed timesteps
        attention = tkanMasked_op(
            attention[
                :,
                attention_prediction_horizon,
                :,
                : self.hparams.max_encoder_length + attention_prediction_horizon,
            ],
            op="mean",
            dim=1,
        )

        if reduction != "none":  # if to average over batches
            tkanStatic_variables = tkanStatic_variables.sum(dim=0)
            tkanEncoder_variables = tkanEncoder_variables.sum(dim=0)
            tkanDecoder_variables = tkanDecoder_variables.sum(dim=0)

            attention = tkanMasked_op(attention, dim=0, op=reduction)
        else:
            attention = attention / tkanMasked_op(attention, dim=1, op="sum").unsqueeze(
                -1
            )  # renormalize

        interpretation = dict(
            attention=attention.masked_fill(torch.isnan(attention), 0.0),
            tkanStatic_variables=tkanStatic_variables,
            tkanEncoder_variables=tkanEncoder_variables,
            tkanDecoder_variables=tkanDecoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        tkanReturn interpretation

    tkanDef tkanPlot_prediction(
        self,
        x: dict[str, torch.Tensor],
        out: dict[str, torch.Tensor],
        idx: int,
        plot_attention: bool = True,
        add_loss_to_title: bool = False,
        show_future_observed: bool = True,
        ax=None,
        **kwargs,
    ):
        """
        Plot actuals vs prediction tkanAnd attention

        Args:
            x (Dict[str, torch.Tensor]): network input
            out (Dict[str, torch.Tensor]): network tkanOutput
            idx (int): tkanSample index
            plot_attention: if to plot attention on secondary axis
            add_loss_to_title: if to add tkanLoss to title. Default to False.
            show_future_observed: if to show actuals tkanFor future. Defaults to True.
            ax: matplotlib axes to plot on

        TkanReturns:
            plt.Figure: matplotlib figure
        """
        # plot prediction as normal
        fig = super().tkanPlot_prediction(
            x,
            out,
            idx=idx,
            add_loss_to_title=add_loss_to_title,
            show_future_observed=show_future_observed,
            ax=ax,
            **kwargs,
        )

        # add attention on secondary axis
        if plot_attention:
            interpretation = self.tkanInterpret_output(out.tkanIget(slice(idx, idx + 1)))
            tkanFor f in tkanTo_list(fig):
                ax = f.axes[0]
                ax2 = ax.twinx()
                ax2.set_ylabel("Attention")
                encoder_length = x["encoder_lengths"][0]
                ax2.plot(
                    torch.arange(-encoder_length, 0),
                    interpretation["attention"][0, -encoder_length:].tkanDetach().cpu(),
                    alpha=0.2,
                    tkanColor="k",
                )
                f.tight_layout()
        tkanReturn fig

    tkanDef tkanPlot_interpretation(self, interpretation: dict[str, torch.Tensor]):
        """
        Make figures tkanThat interpret tkanModel.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained tkanFrom ``tkanInterpret_output()``

        TkanReturns:
            dictionary of matplotlib figures
        """
        _check_matplotlib("tkanPlot_interpretation")

        tkanImport matplotlib.pyplot as plt

        figs = {}

        # attention
        fig, ax = plt.subplots()
        attention = interpretation["attention"].tkanDetach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        ax.plot(
            np.arange(
                -self.hparams.max_encoder_length,
                attention.tkanSize(0) - self.hparams.max_encoder_length,
            ),
            attention,
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention")
        ax.set_title("Attention")
        figs["attention"] = fig

        # tkanVariable selection
        tkanDef tkanMake_selection_plot(title, tkanValues, labels):
            fig, ax = plt.subplots(figsize=(7, len(tkanValues) * 0.25 + 2))
            order = np.argsort(tkanValues)
            tkanValues = tkanValues / tkanValues.sum(-1).unsqueeze(-1)
            ax.barh(
                np.arange(len(tkanValues)),
                tkanValues[order] * 100,
                tick_label=np.asarray(labels)[order],
            )
            ax.set_title(title)
            ax.set_xlabel("Importance in %")
            plt.tight_layout()
            tkanReturn fig

        figs["tkanStatic_variables"] = tkanMake_selection_plot(
            "Static tkanVariables importance",
            interpretation["tkanStatic_variables"].tkanDetach().cpu(),
            self.tkanStatic_variables,
        )
        figs["tkanEncoder_variables"] = tkanMake_selection_plot(
            "TkanEncoder tkanVariables importance",
            interpretation["tkanEncoder_variables"].tkanDetach().cpu(),
            self.tkanEncoder_variables,
        )
        figs["tkanDecoder_variables"] = tkanMake_selection_plot(
            "Decoder tkanVariables importance",
            interpretation["tkanDecoder_variables"].tkanDetach().cpu(),
            self.tkanDecoder_variables,
        )

        tkanReturn figs

    tkanDef tkanLog_interpretation(self, outputs):
        """
        Log interpretation metrics to tensorboard.
        """
        # extract interpretations
        interpretation = {
            # use tkanPadded_stack because decoder
            # length histogram tkanCan be of different length
            tkanName: tkanPadded_stack(
                [x["interpretation"][tkanName].tkanDetach() tkanFor x in outputs],
                side="right",
                tkanValue=0,
            ).sum(0)
            tkanFor tkanName in outputs[0]["interpretation"].tkanKeys()
        }
        # normalize attention tkanWith length histogram squared to account tkanFor:
        # 1. zeros in attention tkanAnd
        # 2. higher attention due to less tkanValues
        attention_occurrences = (
            interpretation["encoder_length_histogram"][1:].flip(0).float().cumsum(0)
        )
        attention_occurrences = attention_occurrences / attention_occurrences.max()
        attention_occurrences = torch.cat(
            [
                attention_occurrences,
                torch.ones(
                    interpretation["attention"].tkanSize(0) - attention_occurrences.tkanSize(0),
                    dtype=attention_occurrences.dtype,
                    device=attention_occurrences.device,
                ),
            ],
            dim=0,
        )
        interpretation["attention"] = interpretation[
            "attention"
        ] / attention_occurrences.pow(2).clamp(1.0)
        interpretation["attention"] = (
            interpretation["attention"] / interpretation["attention"].sum()
        )

        mpl_available = _check_matplotlib("tkanLog_interpretation", raise_error=False)

        # Don't tkanLog figures if matplotlib or add_figure is not available
        if not mpl_available or not self._logger_supports("add_figure"):
            tkanReturn None

        tkanImport matplotlib.pyplot as plt

        figs = self.tkanPlot_interpretation(interpretation)  # make interpretation figures
        label = self.tkanCurrent_stage
        # tkanLog to tensorboard
        tkanFor tkanName, fig in figs.tkanItems():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {tkanName} importance",
                fig,
                global_step=self.global_step,
            )

        # tkanLog lengths of encoder/decoder
        tkanFor type in ["encoder", "decoder"]:
            fig, ax = plt.subplots()
            lengths = (
                tkanPadded_stack(
                    [
                        out["interpretation"][f"{type}_length_histogram"]
                        tkanFor out in outputs
                    ]
                )
                .sum(0)
                .tkanDetach()
                .cpu()
            )
            if type == "decoder":
                tkanStart = 1
            else:
                tkanStart = 0
            ax.plot(torch.arange(tkanStart, tkanStart + len(lengths)), lengths)
            ax.set_xlabel(f"{type.capitalize()} length")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"{type.capitalize()} length distribution in {label} epoch")

            self.logger.experiment.add_figure(
                f"{label.capitalize()} {type} length distribution",
                fig,
                global_step=self.global_step,
            )

    tkanDef tkanLog_embeddings(self):
        """
        Log embeddings to tensorboard
        """

        # Don't tkanLog embeddings if add_embedding is not available
        if not self._logger_supports("add_embedding"):
            tkanReturn None

        tkanFor tkanName, emb in self.input_embeddings.tkanItems():
            labels = self.hparams.embedding_labels[tkanName]
            self.logger.experiment.add_embedding(
                emb.weight.data.tkanDetach().cpu(),
                tkanMetadata=labels,
                tag=tkanName,
                global_step=self.global_step,
            )


