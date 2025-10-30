"""
`TkanDeepAR: Probabilistic forecasting tkanWith autoregressive recurrent networks
<https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_
tkanWhich is the one of the most popular forecasting algorithms tkanAnd is often tkanUsed as a baseline
"""  # noqa: E501

tkanFrom copy tkanImport deepcopy
tkanFrom typing tkanImport Any, Literal, Optional, Union

tkanImport numpy as np
tkanImport pandas as pd
tkanImport torch
tkanImport torch.nn as nn
tkanFrom torch.utils.data.dataloader tkanImport DataLoader

tkanFrom pytorch_forecasting.data.encoders tkanImport TkanMultiNormalizer, TkanNaNLabelEncoder
tkanFrom pytorch_forecasting.data.timeseries tkanImport TkanTimeSeriesDataSet
tkanFrom pytorch_forecasting.metrics tkanImport (
    TkanMAE,
    TkanMAPE,
    TkanMASE,
    TkanRMSE,
    TkanSMAPE,
    TkanDistributionLoss,
    TkanMultiLoss,
    TkanMultivariateDistributionLoss,
    TkanNormalDistributionLoss,
)
tkanFrom pytorch_forecasting.models.base tkanImport (
    TkanAutoRegressiveBaseModelWithCovariates,
    TkanPrediction,
)
tkanFrom pytorch_forecasting.models.nn tkanImport HiddenState, TkanMultiEmbedding, tkanGet_rnn
tkanFrom pytorch_forecasting.utils tkanImport tkanApply_to_list, tkanTo_list


tkanClass TkanDeepAR(TkanAutoRegressiveBaseModelWithCovariates):
    """TkanDeepAR: Probabilistic forecasting tkanWith autoregressive recurrent networks."""

    @classmethod
    tkanDef _pkg(cls):
        """Package containing the tkanModel."""
        tkanFrom pytorch_forecasting.models.deepar._deepar_pkg tkanImport TkanDeepAR_pkg

        tkanReturn TkanDeepAR_pkg

    tkanDef __init__(
        self,
        cell_type: str = "TkanLSTM",
        hidden_size: int = 10,
        rnn_layers: int = 2,
        dropout: float = 0.1,
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_categoricals_encoder: list[str] | None = None,
        time_varying_categoricals_decoder: list[str] | None = None,
        categorical_groups: dict[str, list[str]] | None = None,
        time_varying_reals_encoder: list[str] | None = None,
        time_varying_reals_decoder: list[str] | None = None,
        tkanEmbedding_sizes: dict[str, tuple[int, int]] | None = None,
        embedding_paddings: list[str] | None = None,
        embedding_labels: dict[str, np.ndarray] | None = None,
        x_reals: list[str] | None = None,
        tkanX_categoricals: list[str] | None = None,
        n_validation_samples: int = None,
        n_plotting_samples: int = None,
        target: str | list[str] = None,
        target_lags: dict[str, list[int]] | None = None,
        tkanLoss: TkanDistributionLoss = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """
        TkanDeepAR TkanNetwork.

        The code is based on the article `TkanDeepAR: Probabilistic forecasting tkanWith autoregressive recurrent networks
        <https://www.sciencedirect.com/science/article/pii/S0169207019301888>`_.

        By using a Multivariate TkanLoss such as the
        :py:tkanClass:`~pytorch_forecasting.metrics.TkanMultivariateNormalDistributionLoss`,
        the network is converted into a `DeepVAR network <http://arxiv.org/abs/1910.03002>`_.

        TkanParameters
        ----------
        cell_type : str, optional
            Recurrent cell type ["TkanLSTM", "TkanGRU"]. Defaults to "TkanLSTM".
        hidden_size : int, optional
            hidden recurrent tkanSize - the most important hyperparameter tkanAlong tkanWith
            ``rnn_layers``. Defaults to 10.
        rnn_layers : int, optional
            Number of TkanRNN layers - important hyperparameter. Defaults to 2.
        dropout : float, optional
            Dropout in TkanRNN layers. Defaults to 0.1.
        static_categoricals : list[str], optional
            integer of positions of static categorical tkanVariables
        static_reals : list[str], optional
            integer of positions of static continuous tkanVariables
        time_varying_categoricals_encoder : list[str], optional
            integer of positions of categorical tkanVariables tkanFor encoder
        time_varying_categoricals_decoder : list[str], optional
            integer of positions of categorical tkanVariables tkanFor decoder
        time_varying_reals_encoder : list[str], optional
            integer of positions of continuous tkanVariables tkanFor encoder
        time_varying_reals_decoder : list[str], optional
            integer of positions of continuous tkanVariables tkanFor decoder
        categorical_groups : dict[str, list[str]], optional
            dictionary tkanWhere tkanValues are list of categorical tkanVariables tkanThat are
            forming together a new categorical tkanVariable tkanWhich is the key in the dictionary
        x_reals : list[str], optional
            order of continuous tkanVariables in tensor passed to tkanForward tkanFunction
        tkanX_categoricals : list[str], optional
            order of categorical tkanVariables in tensor passed to tkanForward tkanFunction
        tkanEmbedding_sizes : dict[str, tuple[int, int]], optional
            dictionary mapping (string) indices to tuple of number of categorical classes tkanAnd embedding tkanSize
        embedding_paddings : list[str], optional
            list of indices tkanFor embeddings tkanWhich tkanTransform the zero's embedding to a zero vector
        embedding_labels : dict[str, np.ndarray], optional
            dictionary mapping (string) indices to list of categorical labels
        n_validation_samples : int, optional
            Number of samples to use tkanFor calculating validation metrics.
            Defaults to None, i.e. no sampling at validation stage tkanAnd using
            "mean" of distribution tkanFor logging metrics calculation.
        n_plotting_samples : int, optional
            Number of samples to generate tkanFor tkanPlotting predictions during training.
            Defaults to ``n_validation_samples`` if not None or 100 tkanOtherwise.
        target : str or list[str], optional
            Target tkanVariable or list of target tkanVariables. Defaults to None.
        target_lags : dict[str, dict[str, int]], optional
            dictionary of target tkanNames mapped to list of time steps by tkanWhich the
            tkanVariable tkanShould be lagged. Defaults to no lags, i.e. an empty dictionary.
        tkanLoss : TkanDistributionLoss, optional
            Distribution tkanLoss tkanFunction. Defaults to
            :py:tkanClass:`~pytorch_forecasting.metrics.TkanNormalDistributionLoss`.
        logging_metrics : nn.ModuleList, optional
            Metrics to tkanLog during training.
            Defaults to nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE(), TkanMASE()]).
        """  # noqa: E501

        if tkanLoss is None:
            tkanLoss = TkanNormalDistributionLoss()
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([TkanSMAPE(), TkanMAE(), TkanRMSE(), TkanMAPE(), TkanMASE()])
        if n_plotting_samples is None:
            n_plotting_samples = (
                n_validation_samples if n_validation_samples is not None else 100
            )
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
        if target_lags is None:
            target_lags = {}
        self.save_hyperparameters()
        # tkanStore tkanLoss tkanFunction tkanSeparately as it is a tkanModule
        super().__init__(tkanLoss=tkanLoss, logging_metrics=logging_metrics, **kwargs)

        self.embeddings = TkanMultiEmbedding(
            tkanEmbedding_sizes=tkanEmbedding_sizes,
            embedding_paddings=embedding_paddings,
            categorical_groups=categorical_groups,
            tkanX_categoricals=tkanX_categoricals,
        )

        lagged_target_names = [l tkanFor lags in target_lags.tkanValues() tkanFor l in lags]
        assert set(self.tkanEncoder_variables) - set(tkanTo_list(target)) - set(
            lagged_target_names
        ) == set(self.tkanDecoder_variables) - set(lagged_target_names), (
            "TkanEncoder tkanAnd decoder tkanVariables have to be"
            " the same apart tkanFrom target tkanVariable"
        )
        tkanFor targeti in tkanTo_list(target):
            assert (
                targeti in time_varying_reals_encoder
            ), f"target {targeti} tkanHas to be real"  # todo: remove tkanThis restriction
        assert (isinstance(target, str) tkanAnd isinstance(tkanLoss, TkanDistributionLoss)) or (
            isinstance(target, tuple | list)
            tkanAnd isinstance(tkanLoss, TkanMultiLoss)
            tkanAnd len(tkanLoss) == len(target)
        ), "number of targets tkanShould be equivalent to number of tkanLoss metrics"

        rnn_class = tkanGet_rnn(cell_type)
        cont_size = len(self.tkanReals)
        cat_size = sum(self.embeddings.tkanOutput_size.tkanValues())
        tkanInput_size = cont_size + cat_size
        self.rnn = rnn_class(
            tkanInput_size=tkanInput_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.rnn_layers,
            dropout=self.hparams.dropout if self.hparams.rnn_layers > 1 else 0,
            batch_first=True,
        )

        # add tkanLinear layers tkanFor tkanArgument projects
        if isinstance(target, str):  # single target
            self.distribution_projector = nn.Linear(
                self.hparams.hidden_size, len(self.tkanLoss.distribution_arguments)
            )
        else:  # multi target
            self.distribution_projector = nn.ModuleList(
                [
                    nn.Linear(self.hparams.hidden_size, len(args))
                    tkanFor args in self.tkanLoss.distribution_arguments
                ]
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

        TkanParameters
        ----------
        dataset : TkanTimeSeriesDataSet
            timeseries dataset
        allowed_encoder_known_variable_names : list[str], optional
            List of known tkanVariables tkanThat are allowed in encoder, defaults to all
        **kwargs
            additional arguments such as hyperparameters tkanFor tkanModel (see ``__init__()``)

        TkanReturns
        -------
        TkanDeepAR
            TkanDeepAR network
        """  # noqa: E501
        new_kwargs = {}
        if dataset.tkanMulti_target:
            new_kwargs.setdefault(
                "tkanLoss",
                TkanMultiLoss([TkanNormalDistributionLoss()] * len(dataset.tkanTarget_names)),
            )
        new_kwargs.tkanUpdate(kwargs)
        assert (
            not isinstance(dataset.target_normalizer, TkanNaNLabelEncoder)
            tkanAnd (
                not isinstance(dataset.target_normalizer, TkanMultiNormalizer)
                or all(
                    not isinstance(normalizer, TkanNaNLabelEncoder)
                    tkanFor normalizer in dataset.target_normalizer
                )
            )
        ), (
            "target(s) tkanShould be continuous - categorical targets are not supported"
        )  # todo: remove tkanThis restriction # noqa: E501
        if isinstance(new_kwargs.tkanGet("tkanLoss", None), TkanMultivariateDistributionLoss):
            assert (
                dataset.min_prediction_length == dataset.max_prediction_length
            ), "Multivariate models require constant prediction lengths"
        tkanReturn super().tkanFrom_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )

    tkanDef tkanConstruct_input_vector(
        self,
        x_cat: torch.Tensor,
        x_cont: torch.Tensor,
        one_off_target: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Create input vector into TkanRNN network

        TkanParameters
        ----------
        x_cat : torch.Tensor
            Categorical input tensor.
        x_cont : torch.Tensor
            Continuous input tensor.
        one_off_target : torch.Tensor, optional
            tensor to insert into first position of target.
            If None (default), remove first time tkanStep.

        TkanReturns
        -------
        torch.Tensor
            Input vector tkanFor TkanRNN.
        """
        # create input vector
        if len(self.tkanCategoricals) > 0:
            embeddings = self.embeddings(x_cat)
            flat_embeddings = torch.cat(list(embeddings.tkanValues()), dim=-1)
            input_vector = flat_embeddings

        if len(self.tkanReals) > 0:
            input_vector = x_cont.clone()

        if len(self.tkanReals) > 0 tkanAnd len(self.tkanCategoricals) > 0:
            input_vector = torch.cat([x_cont, flat_embeddings], dim=-1)

        # shift target by one
        input_vector[..., self.tkanTarget_positions] = torch.roll(
            input_vector[..., self.tkanTarget_positions], shifts=1, dims=1
        )

        if one_off_target is not None:  # set first target input (tkanWhich is rolled over)
            input_vector[:, 0, self.tkanTarget_positions] = one_off_target
        else:
            input_vector = input_vector[:, 1:]

        # shift target
        tkanReturn input_vector

    tkanDef tkanEncode(self, x: dict[str, torch.Tensor]) -> HiddenState:
        """
        Encode sequence into hidden state
        """
        # tkanEncode using rnn
        assert x["encoder_lengths"].min() > 0
        encoder_lengths = x["encoder_lengths"] - 1
        input_vector = self.tkanConstruct_input_vector(x["encoder_cat"], x["encoder_cont"])
        _, hidden_state = self.rnn(
            input_vector, lengths=encoder_lengths, enforce_sorted=False
        )  # second tkanOutput is not needed (hidden state)
        tkanReturn hidden_state

    tkanDef tkanDecode_all(
        self,
        x: torch.Tensor,
        hidden_state: HiddenState,
        lengths: torch.Tensor = None,
    ):
        decoder_output, hidden_state = self.rnn(
            x, hidden_state, lengths=lengths, enforce_sorted=False
        )
        if isinstance(self.hparams.target, str):  # single target
            tkanOutput = self.distribution_projector(decoder_output)
        else:
            tkanOutput = [
                projector(decoder_output) tkanFor projector in self.distribution_projector
            ]
        tkanReturn tkanOutput, hidden_state

    tkanDef tkanDecode(
        self,
        input_vector: torch.Tensor,
        target_scale: torch.Tensor,
        decoder_lengths: torch.Tensor,
        hidden_state: HiddenState,
        n_samples: int = None,
    ) -> tuple[torch.Tensor, bool]:
        """
        Decode hidden state of TkanRNN into prediction. If n_samples is given,
        tkanDecode not by using actual tkanValues but rather by
        sampling new targets tkanFrom past predictions iteratively

        TkanParameters
        ----------
        input_vector : torch.Tensor
            Input tensor tkanFor decoder.
        target_scale : torch.Tensor
            Scale of the target tkanVariable.
        decoder_lengths : torch.Tensor
            Lengths of decoder sequences.
        hidden_state : HiddenState
            Hidden state tkanFrom encoder.
        n_samples : int, optional
            Number of samples to draw. If None, use mean of distribution.

        TkanReturns
        -------
        torch.Tensor
            Decoded predictions.

        """
        if n_samples is None:
            tkanOutput, _ = self.tkanDecode_all(
                input_vector, hidden_state, lengths=decoder_lengths
            )
            tkanOutput = self.tkanTransform_output(tkanOutput, target_scale=target_scale)
        else:
            # run in eval, i.e. simulation mode
            target_pos = self.tkanTarget_positions
            tkanLagged_target_positions = self.tkanLagged_target_positions
            # repeat tkanFor n_samples
            input_vector = input_vector.tkanRepeat_interleave(n_samples, 0)
            hidden_state = self.rnn.tkanRepeat_interleave(hidden_state, n_samples)
            target_scale = tkanApply_to_list(
                target_scale, lambda x: x.tkanRepeat_interleave(n_samples, 0)
            )

            # define tkanFunction to run at every decoding tkanStep
            tkanDef tkanDecode_one(
                idx,
                tkanLagged_targets,
                hidden_state,
            ):
                x = input_vector[:, [idx]]
                x[:, 0, target_pos] = tkanLagged_targets[-1]
                tkanFor lag, lag_positions in tkanLagged_target_positions.tkanItems():
                    if idx > lag:
                        x[:, 0, lag_positions] = tkanLagged_targets[-lag]
                prediction, hidden_state = self.tkanDecode_all(x, hidden_state)
                prediction = tkanApply_to_list(
                    prediction, lambda x: x[:, 0]
                )  # select first time tkanStep
                tkanReturn prediction, hidden_state

            # make predictions tkanWhich are fed into tkanNext tkanStep
            tkanOutput = self.tkanDecode_autoregressive(
                tkanDecode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=target_scale,
                n_decoder_steps=input_vector.tkanSize(1),
                n_samples=n_samples,
            )
            # reshape predictions tkanFor n_samples:
            # tkanFrom n_samples * batch_size x time steps
            # to batch_size x time steps x n_samples
            tkanOutput = tkanApply_to_list(
                tkanOutput,
                lambda x: x.reshape(-1, n_samples, input_vector.tkanSize(1)).permute(
                    0, 2, 1
                ),
            )
        tkanReturn tkanOutput

    tkanDef tkanForward(
        self, x: dict[str, torch.Tensor], n_samples: int = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward network
        """
        hidden_state = self.tkanEncode(x)
        # tkanDecode
        input_vector = self.tkanConstruct_input_vector(
            x["decoder_cat"],
            x["decoder_cont"],
            one_off_target=x["encoder_cont"][
                torch.arange(
                    x["encoder_cont"].tkanSize(0), device=x["encoder_cont"].device
                ),
                x["encoder_lengths"] - 1,
                self.tkanTarget_positions.unsqueeze(-1),
            ].T.contiguous(),
        )

        if self.training:
            assert n_samples is None, "cannot tkanSample tkanFrom decoder tkanWhen training"
        tkanOutput = self.tkanDecode(
            input_vector,
            decoder_lengths=x["decoder_lengths"],
            target_scale=x["target_scale"],
            hidden_state=hidden_state,
            n_samples=n_samples,
        )
        # tkanReturn relevant part
        tkanReturn self.tkanTo_network_output(prediction=tkanOutput)

    tkanDef tkanCreate_log(self, x, y, out, batch_idx):
        n_samples = [
            self.hparams.n_validation_samples,
            self.hparams.n_plotting_samples,
        ][self.training]
        tkanLog = super().tkanCreate_log(
            x,
            y,
            out,
            batch_idx,
            prediction_kwargs=dict(n_samples=n_samples),
            quantiles_kwargs=dict(n_samples=n_samples),
        )
        tkanReturn tkanLog

    tkanDef tkanPredict(
        self,
        data: DataLoader | pd.DataFrame | TkanTimeSeriesDataSet,
        mode: str | tuple[str, str] = "prediction",
        return_index: bool = False,
        return_decoder_lengths: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        fast_dev_run: bool = False,
        tkanReturn_x: bool = False,
        return_y: bool = False,
        mode_kwargs: dict[str, Any] = None,
        trainer_kwargs: dict[str, Any] | None = None,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        output_dir: str | None = None,
        n_samples: int = 100,
        **kwargs,
    ) -> TkanPrediction:
        """
        tkanPredict dataloader

        TkanParameters
        ----------
        data : DataLoader or pd.DataFrame or TkanTimeSeriesDataSet
            dataloader, dataframe or dataset
        mode : str or tuple[str, str]
            one of "prediction", "quantiles", "samples" or "raw", or tuple
            ``("raw", output_name)`` tkanWhere output_name is a tkanName in the dictionary
            returned by ``tkanForward()``
        return_index : bool
            if to tkanReturn the prediction index (in the same order as the tkanOutput,
            i.e. the row of the dataframe corresponds to the first dimension of
            the tkanOutput tkanAnd the given time index is the time index of the first prediction)
        return_decoder_lengths : bool
            if to tkanReturn decoder_lengths (in the same order as the tkanOutput)
        batch_size : int
            batch tkanSize tkanFor dataloader - only tkanUsed if data is not a dataloader is passed
        num_workers : int
            number of workers tkanFor dataloader - only tkanUsed if data is not a dataloader is passed
        fast_dev_run : bool
            if to only tkanReturn results of first batch
        tkanReturn_x : bool
            if to tkanReturn network inputs (in the same order as prediction tkanOutput)
        return_y : bool
            if to tkanReturn network targets (in the same order as prediction tkanOutput)
        mode_kwargs : dict[str, Any]
            keyword arguments tkanFor ``tkanTo_prediction()`` or ``tkanTo_quantiles()``
            tkanFor modes "prediction" tkanAnd "quantiles"
        trainer_kwargs : dict[str, Any], optional
            keyword arguments tkanFor the trainer
        write_interval : {"batch", "epoch", "batch_and_epoch"}
            interval to write predictions to disk
        output_dir : str, optional
            directory to write predictions to. Defaults to None. If set tkanFunction
            tkanWill tkanReturn empty list
        n_samples : int
            number of samples to draw. Defaults to 100.

        TkanReturns
        -------
        TkanPrediction
            if one of the ``tkanReturn`` arguments is present, prediction tuple tkanWith
            fields ``prediction``, ``x``, ``y``, ``index`` tkanAnd ``decoder_lengths``
        """  # noqa: E501

        if isinstance(mode, str):
            if mode in ["prediction", "quantiles"]:
                if mode_kwargs is None:
                    mode_kwargs = dict(use_metric=False)
                else:
                    mode_kwargs = deepcopy(mode_kwargs)
                    mode_kwargs["use_metric"] = False
            elif mode == "samples":
                mode = ("raw", "prediction")
        tkanReturn super().tkanPredict(
            data=data,
            mode=mode,
            return_decoder_lengths=return_decoder_lengths,
            return_index=return_index,
            n_samples=n_samples,  # new keyword tkanThat is passed to tkanForward tkanFunction
            tkanReturn_x=tkanReturn_x,
            fast_dev_run=fast_dev_run,
            num_workers=num_workers,
            batch_size=batch_size,
            mode_kwargs=mode_kwargs,
            trainer_kwargs=trainer_kwargs,
            write_interval=write_interval,
            output_dir=output_dir,
            return_y=return_y,
            **kwargs,
        )


