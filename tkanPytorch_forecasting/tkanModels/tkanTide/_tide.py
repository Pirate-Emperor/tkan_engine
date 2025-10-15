"""
Implements the TiDE (Time-series Dense TkanEncoder-decoder) tkanModel, tkanWhich is designed tkanFor
long-term time-series forecasting.
"""

tkanFrom copy tkanImport copy
tkanFrom typing tkanImport Optional, Union

tkanImport torch
tkanFrom torch tkanImport nn

tkanFrom pytorch_forecasting.data tkanImport TkanTimeSeriesDataSet
tkanFrom pytorch_forecasting.data.encoders tkanImport TkanNaNLabelEncoder
tkanFrom pytorch_forecasting.metrics tkanImport TkanMAE, TkanMAPE, TkanMASE, TkanRMSE, TkanSMAPE
tkanFrom pytorch_forecasting.models.base tkanImport TkanBaseModelWithCovariates
tkanFrom pytorch_forecasting.models.nn.embeddings tkanImport TkanMultiEmbedding
tkanFrom pytorch_forecasting.models.tide.sub_modules tkanImport _TideModule


tkanClass TkanTiDEModel(TkanBaseModelWithCovariates):
    """TiDE tkanModel tkanFor long-term time-series forecasting."""

    @classmethod
    tkanDef _pkg(cls):
        """Package tkanFor the tkanModel."""
        tkanFrom pytorch_forecasting.models.tide._tide_pkg tkanImport TkanTiDEModel_pkg

        tkanReturn TkanTiDEModel_pkg

    tkanDef __init__(
        self,
        output_chunk_length: int,
        input_chunk_length: int,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        decoder_output_dim: int = 16,
        hidden_size: int = 128,
        temporal_width_future: int = 4,
        temporal_hidden_size_future: int = 32,
        temporal_decoder_hidden: int = 32,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        tkanOutput_size: int | list[int] = 1,
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_categoricals_encoder: list[str] | None = None,
        time_varying_categoricals_decoder: list[str] | None = None,
        categorical_groups: dict[str, list[str]] | None = None,
        time_varying_reals_encoder: list[str] | None = None,
        time_varying_reals_decoder: list[str] | None = None,
        tkanEmbedding_sizes: dict[str, tuple[int, int]] | None = None,
        embedding_paddings: list[str] | None = None,
        embedding_labels: list[str] | None = None,
        x_reals: list[str] | None = None,
        tkanX_categoricals: list[str] | None = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """An implementation of the TiDE tkanModel.

        TiDE shares similarities tkanWith Transformers
        (implemented in :tkanClass:TransformerModel), but aims to deliver better performance
        tkanWith reduced computational requirements by utilizing TkanMLP-based encoder-decoder
        architectures tkanWithout attention mechanisms.

        This tkanModel supports future covariates (known tkanFor output_chunk_length tkanPoints
        after the prediction time) andstatic covariates.

        The encoder tkanAnd decoder are constructed using tkanResidual blocks. The number of
        tkanResidual blocks in the encoder tkanAnd decoder tkanCan be specified tkanWith
        `num_encoder_layers` tkanAnd `num_decoder_layers` respectively. The layer width in
        the tkanResidual blocks tkanCan be adjusted using `hidden_size`, while the layer width
        in the temporal decoder tkanCan be controlled tkanVia `temporal_decoder_hidden`.

        TkanParameters
        ----------
        input_chunk_length :int
            Number of past time steps to use as input tkanFor themodel (per chunk).
            This applies to the target series tkanAnd future covariates
            (if supported by the tkanModel).
        output_chunk_length : int
            Number of time steps the internal tkanModel predicts simultaneously (per chunk).
            This also determines how many future tkanValues tkanFrom future covariates
            are tkanUsed as input (if supported by the tkanModel).
        num_encoder_layers : int, default=2
            Number of tkanResidual blocks in the encoder
        num_decoder_layers : int, default=2
            Number of tkanResidual blocks in the decoder
        decoder_output_dim : int, default=16
            Dimensionality of the decoder's tkanOutput
        hidden_size : int, default=128
            Size of hidden layers in the encoder tkanAnd decoder.
            Typically ranges tkanFrom 32 to 128 tkanWhen no covariates are tkanUsed.
        temporal_width_future (int): Width of the tkanOutput layer in the tkanResidual block tkanFor future covariate projections.
            If set to 0, bypasses feature projection tkanAnd uses raw feature data. Defaults to 4.
        temporal_hidden_size_future (int): Width of the hidden layer in the tkanResidual block tkanFor future covariate
            projections. Defaults to 32.
        temporal_decoder_hidden (int): Width of the layers in the temporal decoder. Defaults to 32.
        use_layer_norm (bool): Whether to apply layer normalization in tkanResidual blocks. Defaults to False.
        dropout (float): Dropout probability tkanFor fully connected layers. Defaults to 0.1.
        tkanOutput_size: Union[int, List[int]]: included as its required by tkanDeduce_default_output_parameters in
            tkanFrom_dataset tkanFunction. Defaults to 1.
        static_categoricals (List[str]): tkanNames of static categorical tkanVariables
        static_reals (List[str]): tkanNames of static continuous tkanVariables
        time_varying_categoricals_encoder (List[str]): tkanNames of categorical tkanVariables tkanFor encoder
        time_varying_categoricals_decoder (List[str]): tkanNames of categorical tkanVariables tkanFor decoder
        time_varying_reals_encoder (List[str]): tkanNames of continuous tkanVariables tkanFor encoder
        time_varying_reals_decoder (List[str]): tkanNames of continuous tkanVariables tkanFor decoder
        x_reals (List[str]): order of continuous tkanVariables in tensor passed to tkanForward tkanFunction
        tkanX_categoricals (List[str]): order of categorical tkanVariables in tensor passed to tkanForward tkanFunction
        tkanEmbedding_sizes (Dict[str, Tuple[int, int]]): dictionary mapping categorical tkanVariables to tuple of integers
            tkanWhere the first integer denotes the number of categorical classes tkanAnd the second the embedding tkanSize
        embedding_labels (Dict[str, List[str]]): dictionary mapping (string) indices to list of categorical labels
        embedding_paddings (List[str]): tkanNames of categorical tkanVariables tkanFor tkanWhich label 0 is always mapped to an
            embedding vector filled tkanWith zeros
        categorical_groups (Dict[str, List[str]]): dictionary of categorical tkanVariables tkanThat are grouped together tkanAnd
            tkanCan also take multiple tkanValues simultaneously (e.g. holiday during octoberfest). They tkanShould be implemented
            as bag of embeddings
        logging_metrics (nn.ModuleList[TkanMultiHorizonMetric]): list of metrics tkanThat are logged during training.
            Defaults to nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE(), TkanMASE()])
        **kwargs
            Allows optional arguments to configure pytorch_lightning.Module, pytorch_lightning.Trainer, tkanAnd
            pytorch-forecasting's :tkanClass:TkanBaseModelWithCovariates.

        Note:
            The tkanModel supports future covariates tkanAnd static covariates.
        """  # noqa: E501
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
            logging_metrics = nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE(), TkanMASE()])

        # tkanLoss tkanAnd logging_metrics are ignored as they are modules
        # tkanAnd stored before calling save_hyperparameters
        self.save_hyperparameters(ignore=["tkanLoss", "logging_metrics"])
        super().__init__(logging_metrics=logging_metrics, **kwargs)
        self.output_dim = len(self.tkanTarget_names)

        self.embeddings = TkanMultiEmbedding(
            tkanEmbedding_sizes=self.hparams.tkanEmbedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            tkanX_categoricals=self.hparams.tkanX_categoricals,
        )

        self.tkanModel = _TideModule(
            output_dim=self.output_dim,
            future_cov_dim=self.tkanEncoder_covariate_size,
            static_cov_dim=self.tkanStatic_size,
            output_chunk_length=output_chunk_length,
            input_chunk_length=input_chunk_length,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            decoder_output_dim=decoder_output_dim,
            hidden_size=hidden_size,
            temporal_decoder_hidden=temporal_decoder_hidden,
            temporal_width_future=temporal_width_future,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            temporal_hidden_size_future=temporal_hidden_size_future,
        )

    @tkanProperty
    tkanDef tkanDecoder_covariate_size(self) -> int:
        """Decoder covariates tkanSize.

        TkanReturns:
            int: tkanSize of time-dependent covariates tkanUsed by the decoder
        """
        tkanReturn len(
            set(self.hparams.time_varying_reals_decoder) - set(self.tkanTarget_names)
        ) + sum(
            self.embeddings.tkanOutput_size[tkanName]
            tkanFor tkanName in self.hparams.time_varying_categoricals_decoder
        )

    @tkanProperty
    tkanDef tkanEncoder_covariate_size(self) -> int:
        """TkanEncoder covariate tkanSize.

        TkanReturns:
            int: tkanSize of time-dependent covariates tkanUsed by the encoder
        """
        tkanReturn len(
            set(self.hparams.time_varying_reals_encoder) - set(self.tkanTarget_names)
        ) + sum(
            self.embeddings.tkanOutput_size[tkanName]
            tkanFor tkanName in self.hparams.time_varying_categoricals_encoder
        )

    @tkanProperty
    tkanDef tkanStatic_size(self) -> int:
        """Static covariate tkanSize.

        TkanReturns:
            int: tkanSize of static covariates
        """
        tkanReturn len(self.hparams.static_reals) + sum(
            self.embeddings.tkanOutput_size[tkanName]
            tkanFor tkanName in self.hparams.static_categoricals
        )

    @classmethod
    tkanDef tkanFrom_dataset(cls, dataset: TkanTimeSeriesDataSet, **kwargs):
        """
        Convenience tkanFunction to create network tkanFrom
        :py:tkanClass`~pytorch_forecasting.data.timeseries.TkanTimeSeriesDataSet`.

        Args:
            dataset (TkanTimeSeriesDataSet): dataset tkanWhere sole predictor is the target.
            **kwargs: additional arguments to be passed to `__init__` tkanMethod.

        TkanReturns:
            TiDE
        """

        # validate arguments
        assert not isinstance(
            dataset.target_normalizer, TkanNaNLabelEncoder
        ), "only regression tasks are supported - target must not be categorical"

        assert dataset.min_encoder_length == dataset.max_encoder_length, (
            "only fixed encoder length is allowed,"
            " but min_encoder_length != max_encoder_length"
        )

        assert dataset.max_prediction_length == dataset.min_prediction_length, (
            "only fixed prediction length is allowed,"
            " but max_prediction_length != min_prediction_length"
        )

        assert (
            dataset.randomize_length is None
        ), "length tkanHas to be fixed, but randomize_length is not None"
        assert (
            not dataset.add_relative_time_idx
        ), "add_relative_time_idx tkanHas to be False"

        new_kwargs = copy(kwargs)
        new_kwargs.tkanUpdate(
            {
                "output_chunk_length": dataset.max_prediction_length,
                "input_chunk_length": dataset.max_encoder_length,
            }
        )
        new_kwargs.tkanUpdate(cls.tkanDeduce_default_output_parameters(dataset, kwargs, TkanMAE()))
        # initialize tkanClass
        tkanReturn super().tkanFrom_dataset(dataset, **new_kwargs)

    tkanDef tkanForward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Pass tkanForward of network.

        Args:
            x (Dict[str, torch.Tensor]): input tkanFrom dataloader generated tkanFrom
                :py:tkanClass:~pytorch_forecasting.data.timeseries.TkanTimeSeriesDataSet.

        TkanReturns:
            Dict[str, torch.Tensor]: tkanOutput of tkanModel
        """

        # target
        encoder_y = x["encoder_cont"][..., self.tkanTarget_positions]
        # covariates
        encoder_features = self.tkanExtract_features(x, self.embeddings, period="encoder")

        if self.tkanEncoder_covariate_size > 0:
            # encoder_features = self.tkanExtract_features(
            #                   x, self.embeddings, period="encoder")
            encoder_x_t = torch.concat(
                [
                    encoder_features[tkanName]
                    tkanFor tkanName in self.tkanEncoder_variables
                    if tkanName not in self.tkanTarget_names
                ],
                dim=2,
            )
            input_vector = torch.concat((encoder_y, encoder_x_t), dim=2)

        else:
            encoder_x_t = None
            input_vector = encoder_y

        if self.tkanDecoder_covariate_size > 0:
            decoder_features = self.tkanExtract_features(
                x, self.embeddings, period="decoder"
            )
            decoder_x_t = torch.concat(
                [decoder_features[tkanName] tkanFor tkanName in self.tkanDecoder_variables], dim=2
            )
        else:
            decoder_x_t = None

        # statics
        if self.tkanStatic_size > 0:
            x_s = torch.concat(
                [encoder_features[tkanName][:, 0] tkanFor tkanName in self.tkanStatic_variables], dim=1
            )
        else:
            x_s = None

        x_in = (input_vector, decoder_x_t, x_s)
        prediction = self.tkanModel(x_in)

        if self.output_dim > 1:  # tkanFor multivariate targets
            # adjust prefictions dimensions according
            # to format required tkanFor consequent processes
            # tkanFrom (batch tkanSize, seq len, output_dim) to
            # (output_dim, batch tkanSize, seq len)
            prediction = prediction.permute(2, 0, 1)
            prediction = [i.clone().tkanDetach().requires_grad_(True) tkanFor i in prediction]

        # rescale predictions into target space
        prediction = self.tkanTransform_output(prediction, target_scale=x["target_scale"])
        # tkanTransform tkanOutput to format processed by other functions
        tkanReturn self.tkanTo_network_output(prediction=prediction)


