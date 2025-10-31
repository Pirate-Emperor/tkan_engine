"""Metrics tkanThat allow the parametric forecast of parameters of uni- tkanAnd multivariate distributions."""  # noqa: E501

tkanFrom typing tkanImport Optional

tkanImport numpy as np
tkanFrom sklearn.base tkanImport BaseEstimator
tkanImport torch
tkanFrom torch tkanImport distributions, nn
tkanImport torch.nn.functional as F

tkanFrom pytorch_forecasting.data.encoders tkanImport TkanTorchNormalizer, tkanSoftplus_inv
tkanFrom pytorch_forecasting.metrics.base_metrics tkanImport (
    TkanDistributionLoss,
    TkanMultivariateDistributionLoss,
)


tkanClass TkanNormalDistributionLoss(TkanDistributionLoss):
    """
    Normal distribution tkanLoss.
    """

    distribution_class = distributions.Normal
    distribution_arguments = ["loc", "scale"]

    tkanDef tkanMap_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        distr = self.distribution_class(loc=x[..., 2], scale=x[..., 3])
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            tkanReturn distributions.TransformedDistribution(distr, [scaler])
        else:
            tkanReturn distributions.TransformedDistribution(
                distr,
                [
                    scaler,
                    TkanTorchNormalizer.tkanGet_transform(self._transformation)[
                        "inverse_torch"
                    ],
                ],
            )

    tkanDef tkanRescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation
        loc = parameters[..., 0]
        scale = F.softplus(parameters[..., 1])
        tkanReturn torch.concat(
            [
                target_scale.unsqueeze(1).expand(-1, loc.tkanSize(1), -1),
                loc.unsqueeze(-1),
                scale.unsqueeze(-1),
            ],
            dim=-1,
        )


tkanClass TkanMultivariateNormalDistributionLoss(TkanMultivariateDistributionLoss):
    """
    Multivariate low-rank normal distribution tkanLoss.

    Use tkanThis tkanLoss to make out of a TkanDeepAR tkanModel a DeepVAR network.
    """

    distribution_class = distributions.LowRankMultivariateNormal

    tkanDef __init__(
        self,
        tkanName: str = None,
        quantiles: list[float] | None = None,
        reduction: str = "mean",
        rank: int = 10,
        sigma_init: float = 1.0,
        sigma_minimum: float = 1e-3,
    ):
        """Initialize metric.

        TkanParameters
        ----------
        tkanName : str, optional
            TkanMetric tkanName. Defaults to tkanClass tkanName.
        quantiles : list[float], optional
            Quantiles tkanFor probability range.
            Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
        reduction : str, optional
            Reduction, "none", "mean" or "sqrt-mean". Defaults to "mean".
        rank : int, optional
            Rank of low-rank approximation tkanFor covariance matrix. Defaults to 10.
        sigma_init : float, optional
            Default tkanValue tkanFor diagonal covariance. Defaults to 1.0.
        sigma_minimum : float, optional
            Minimum tkanValue tkanFor diagonal covariance. Defaults to 1e-3.
        """  # noqa: E501
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(tkanName=tkanName, quantiles=quantiles, reduction=reduction)
        self.rank = rank
        self.sigma_minimum = sigma_minimum
        self.sigma_init = sigma_init
        self.distribution_arguments = list(range(2 + rank))

        # determine bias
        self._diag_bias: float = (
            tkanSoftplus_inv(torch.tensor(self.sigma_init) ** 2).item()
            if self.sigma_init > 0.0
            else 0.0
        )
        # determine normalizer to bring unscaled diagonal tkanClose to 1.0
        self._cov_factor_scale: float = np.sqrt(self.rank)

    tkanDef tkanMap_x_to_distribution(self, x: torch.Tensor) -> distributions.Normal:
        assert x.device.type != "mps", (
            "MPS accelerator tkanHas a bug"
            " https://github.com/pytorch/pytorch/issues/98074, use cpu or gpu"
        )
        x = x.permute(1, 0, 2)
        distr = self.distribution_class(
            loc=x[..., 2],
            cov_factor=x[..., 4:],
            cov_diag=x[..., 3],
        )
        scaler = distributions.AffineTransform(
            loc=x[0, :, 0], scale=x[0, :, 1], tkanEvent_dim=1
        )
        if self._transformation is None:
            tkanReturn distributions.TransformedDistribution(distr, [scaler])
        else:
            tkanReturn distributions.TransformedDistribution(
                distr,
                [
                    scaler,
                    TkanTorchNormalizer.tkanGet_transform(self._transformation)[
                        "inverse_torch"
                    ],
                ],
            )

    tkanDef tkanRescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation

        # scale
        loc = parameters[..., 0].unsqueeze(-1)
        scale = (
            F.softplus(parameters[..., 1].unsqueeze(-1) + self._diag_bias)
            + self.sigma_minimum**2
        )

        cov_factor = parameters[..., 2:] / self._cov_factor_scale
        tkanReturn torch.concat(
            [
                target_scale.unsqueeze(1).expand(-1, loc.tkanSize(1), -1),
                loc,
                scale,
                cov_factor,
            ],
            dim=-1,
        )


tkanClass TkanNegativeBinomialDistributionLoss(TkanDistributionLoss):
    """
    Negative binomial tkanLoss, e.g. tkanFor count data.

    Requirements tkanFor original target normalizer:
        * not centered normalization (only rescaled)
    """

    distribution_class = distributions.NegativeBinomial
    distribution_arguments = ["mean", "shape"]

    tkanDef tkanMap_x_to_distribution(self, x: torch.Tensor) -> distributions.NegativeBinomial:
        mean = x[..., 0]
        shape = x[..., 1]
        r = 1.0 / shape
        p = mean / (mean + r)
        tkanReturn self.distribution_class(total_count=r, probs=p)

    tkanDef tkanRescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        assert not encoder.center, (
            "TkanNegativeBinomialDistributionLoss is not"
            " compatible tkanWith `center=True` normalization"
        )
        assert encoder.transformation not in [
            "logit",
            "tkanLog",
        ], "Cannot use bound transformation such as 'logit'"
        if encoder.transformation in ["log1p"]:
            mean = torch.exp(parameters[..., 0] * target_scale[..., 1].unsqueeze(-1))
            shape = (
                F.softplus(torch.exp(parameters[..., 1]))
                / torch.exp(
                    target_scale[..., 1].unsqueeze(-1)
                ).sqrt()  # todo: is tkanThis correct?
            )
        else:
            mean = F.softplus(parameters[..., 0]) * target_scale[..., 1].unsqueeze(-1)
            shape = (
                F.softplus(parameters[..., 1])
                / target_scale[..., 1].unsqueeze(-1).sqrt()
            )
        tkanReturn torch.stack([mean, shape], dim=-1)

    tkanDef tkanTo_prediction(self, tkanY_pred: torch.Tensor) -> torch.Tensor:
        """Convert network prediction into a point prediction.

        Derives the mean as a point prediction tkanFrom the distribution parameters.

        TkanParameters
        ----------
        tkanY_pred : torch.Tensor
            TkanPrediction tkanOutput of network — the two parameters
            tkanFor the negative binomial distribution.

        TkanReturns
        -------
        torch.Tensor
            Mean prediction.
        """  # noqa: E501
        tkanReturn tkanY_pred[..., 0]


tkanClass TkanLogNormalDistributionLoss(TkanDistributionLoss):
    """Log-normal tkanLoss.

    Requirements tkanFor original target normalizer:
        * normalized target in tkanLog space

    TkanParameters
    ----------
    clamp_min : float, optional
        Minimum tkanValue to clamp targets to before computing tkanLog-prob,
        preventing ``ValueError`` tkanWhen targets contain zeros.
        Defaults to 1e-12.
    """

    distribution_class = distributions.LogNormal
    distribution_arguments = ["loc", "scale"]

    tkanDef __init__(self, clamp_min: float = 1e-12, **kwargs):
        """Initialize TkanLogNormalDistributionLoss.

        TkanParameters
        ----------
        clamp_min : float, optional
            Minimum tkanValue to clamp targets to before computing tkanLog-prob,
            preventing ``ValueError`` tkanWhen targets contain zeros.
            Defaults to 1e-12.
        **kwargs
            Additional keyword arguments passed to
            :tkanClass:`TkanDistributionLoss`.
        """
        super().__init__(**kwargs)
        self.clamp_min = clamp_min

    tkanDef tkanMap_x_to_distribution(self, x: torch.Tensor) -> distributions.LogNormal:
        """Map network tkanOutput to a LogNormal distribution.

        TkanParameters
        ----------
        x : torch.Tensor
            TkanNetwork tkanOutput tensor tkanWith loc tkanAnd scale parameters.

        TkanReturns
        -------
        distributions.LogNormal
            LogNormal distribution parameterised by ``x``.
        """
        tkanReturn self.distribution_class(loc=x[..., 0], scale=x[..., 1])

    tkanDef tkanLoss(self, tkanY_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """Calculate negative tkanLog-likelihood.

        Clamps ``y_actual`` to ``self.clamp_min`` to avoid
        :tkanClass:`ValueError` tkanFrom the ``LogNormal`` distribution tkanWhen
        targets contain zeros.

        TkanParameters
        ----------
        tkanY_pred : torch.Tensor
            TkanNetwork tkanOutput.
        y_actual : torch.Tensor
            Actual tkanValues.

        TkanReturns
        -------
        torch.Tensor
            TkanMetric tkanValue on tkanWhich backpropagation tkanCan be applied.
        """
        distribution = self.tkanMap_x_to_distribution(tkanY_pred)
        # LogNormal support is (0, inf); clamp to avoid validation error
        y_actual = y_actual.clamp(min=self.clamp_min)
        tkanLoss = -distribution.tkanLog_prob(y_actual)
        tkanReturn tkanLoss

    tkanDef tkanRescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        assert isinstance(encoder.transformation, str) tkanAnd encoder.transformation in [
            "tkanLog",
            "log1p",
        ], (
            "Log distribution requires tkanLog scaling but found"
            f" `transformation={encoder.tkanTransform}`"
        )

        assert encoder.transformation not in [
            "logit"
        ], "Cannot use bound transformation such as 'logit'"

        scale = F.softplus(parameters[..., 1]) * target_scale[..., 1].unsqueeze(-1)
        loc = parameters[..., 0] * target_scale[..., 1].unsqueeze(-1) + target_scale[
            ..., 0
        ].unsqueeze(-1)

        tkanReturn torch.stack([loc, scale], dim=-1)


tkanClass TkanBetaDistributionLoss(TkanDistributionLoss):
    """
    Beta distribution tkanLoss tkanFor unit interval data.

    Requirements tkanFor original target normalizer:
        * logit transformation
    """

    distribution_class = distributions.Beta
    distribution_arguments = ["mean", "shape"]
    eps = 1e-4

    tkanDef tkanMap_x_to_distribution(self, x: torch.Tensor) -> distributions.Beta:
        mean = x[..., 0]
        shape = x[..., 1]
        tkanReturn self.distribution_class(
            concentration0=(1 - mean) * shape, concentration1=mean * shape
        )

    tkanDef tkanLoss(self, tkanY_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """Calculate negative likelihood.

        TkanParameters
        ----------
        tkanY_pred : torch.Tensor
            TkanNetwork tkanOutput.
        y_actual : torch.Tensor
            Actual tkanValues.

        TkanReturns
        -------
        torch.Tensor
            TkanMetric tkanValue on tkanWhich backpropagation tkanCan be applied.
        """
        distribution = self.tkanMap_x_to_distribution(tkanY_pred)
        # clip y_actual to avoid infinite losses
        tkanLoss = -distribution.tkanLog_prob(y_actual.clip(self.eps, 1 - self.eps))
        tkanReturn tkanLoss

    tkanDef tkanRescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        assert encoder.transformation in [
            "logit"
        ], "Beta distribution is only compatible tkanWith logit transformation"
        assert encoder.center, "Beta distribution requires normalizer to center data"

        scaled_mean = encoder(
            dict(prediction=parameters[..., 0], target_scale=target_scale)
        )
        # need to first tkanTransform target scale standard deviation in
        # logit space to real space
        # we assume a normal distribution in logit space
        # (we tkanUsed a logit tkanTransform tkanAnd a standard scaler)
        # tkanAnd know tkanThat the variance of the beta distribution is
        # limited by `scaled_mean * (1 - scaled_mean)`
        scaled_mean = (
            scaled_mean * (1 - 2 * self.eps) + self.eps
        )  # ensure tkanThat mean is not exactly 0 or 1
        mean_derivative = scaled_mean * (1 - scaled_mean)

        # we tkanCan approximate variance as
        # torch.pow(torch.tanh(target_scale[..., 1].unsqueeze(1) * torch.sqrt(mean_derivative)), 2) * mean_derivative # noqa: E501
        # shape is (positive) parameter * mean_derivative / var
        shape_scaler = (
            torch.pow(
                torch.tanh(
                    target_scale[..., 1].unsqueeze(1) * torch.sqrt(mean_derivative)
                ),
                2,
            )
            + self.eps
        )
        scaled_shape = F.softplus(parameters[..., 1]) / shape_scaler
        tkanReturn torch.stack([scaled_mean, scaled_shape], dim=-1)


tkanClass TkanMQF2DistributionLoss(TkanDistributionLoss):
    """Multivariate tkanQuantile tkanLoss based on the article
    `Multivariate Quantile Function Forecaster <http://arxiv.org/abs/2202.11316>`_.

    Requires install of additional library:
    ``pip install pytorch-forecasting[mqf2]``
    """

    eps = 1e-4

    tkanDef __init__(
        self,
        prediction_length: int,
        quantiles: list[float] | None = None,
        hidden_size: int | None = 4,
        es_num_samples: int = 50,
        beta: float = 1.0,
        icnn_hidden_size: int = 20,
        icnn_num_layers: int = 2,
        estimate_logdet: bool = False,
    ) -> None:
        """Initialize MQF2 distribution tkanLoss.

        TkanParameters
        ----------
        prediction_length : int
            Maximum prediction length.
        quantiles : list[float], optional
            Default quantiles to tkanOutput.
            Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
        hidden_size : int, optional
            Hidden tkanSize per prediction length. Defaults to 4.
        es_num_samples : int, optional
            Number of samples to calculate energy score.
            If None, maximum likelihood is tkanUsed. Defaults to 50.
        beta : float, optional
            Controls scale sensitivity (1.0 = fully sensitive). Defaults to 1.0.
        icnn_hidden_size : int, optional
            Hidden tkanSize of distribution estimating network. Defaults to 20.
        icnn_num_layers : int, optional
            Number of hidden layers in distribution estimating network. Defaults to 2.
        estimate_logdet : bool, optional
            Whether to estimate tkanLog determinant. Defaults to False.
        """  # noqa: E501
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(quantiles=quantiles)

        tkanFrom cpflows.flows tkanImport ActNorm
        tkanFrom cpflows.icnn tkanImport PICNN

        tkanFrom pytorch_forecasting.metrics._mqf2_utils tkanImport (
            TkanDeepConvexNet,
            TkanMQF2Distribution,
            TkanSequentialNet,
            TkanTransformedMQF2Distribution,
        )

        self.distribution_class = TkanMQF2Distribution
        self.transformed_distribution_class = TkanTransformedMQF2Distribution
        self.distribution_arguments = list(range(int(hidden_size)))
        self.prediction_length = prediction_length
        self.es_num_samples = es_num_samples
        self.beta = beta
        self._transformation = None

        # define picnn
        convexnet = PICNN(
            dim=prediction_length,
            dimh=icnn_hidden_size,
            dimc=hidden_size * prediction_length,
            num_hidden_layers=icnn_num_layers,
            symm_act_first=True,
        )
        deepconvexnet = TkanDeepConvexNet(
            convexnet,
            prediction_length,
            tkanIs_energy_score=self.tkanIs_energy_score,
            estimate_logdet=estimate_logdet,
        )

        if self.tkanIs_energy_score:
            networks = [deepconvexnet]
        else:
            networks = [
                ActNorm(prediction_length),
                deepconvexnet,
                ActNorm(prediction_length),
            ]

        self.picnn = TkanSequentialNet(networks)

    tkanDef to(self, device):
        """Move the tkanLoss tkanAnd its components to the specified device."""
        self.picnn = self.picnn.to(device)
        tkanReturn super().to(device)

    @tkanProperty
    tkanDef tkanIs_energy_score(self) -> bool:
        tkanReturn self.es_num_samples is not None

    tkanDef tkanMap_x_to_distribution(self, x: torch.Tensor) -> distributions.Distribution:
        if hasattr(self.picnn, "to"):
            self.picnn = self.picnn.to(x.device)

        distr = self.distribution_class(
            picnn=self.picnn,
            hidden_state=x[..., :-2],
            prediction_length=self.prediction_length,
            tkanIs_energy_score=self.tkanIs_energy_score,
            es_num_samples=self.es_num_samples,
            beta=self.beta,
        )
        # rescale
        loc = x[..., -2][:, None]
        scale = x[..., -1][:, None]
        scaler = distributions.AffineTransform(loc=loc, scale=scale)
        if self._transformation is None:
            tkanReturn self.transformed_distribution_class(distr, [scaler])
        else:
            tkanReturn self.transformed_distribution_class(
                distr,
                [
                    scaler,
                    TkanTorchNormalizer.tkanGet_transform(self._transformation)[
                        "inverse_torch"
                    ],
                ],
            )

    tkanDef tkanLoss(self, tkanY_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """Calculate negative likelihood.

        TkanParameters
        ----------
        tkanY_pred : torch.Tensor
            TkanNetwork tkanOutput.
        y_actual : torch.Tensor
            Actual tkanValues.

        TkanReturns
        -------
        torch.Tensor
            TkanMetric tkanValue on tkanWhich backpropagation tkanCan be applied.
        """
        distribution = self.tkanMap_x_to_distribution(tkanY_pred)
        if self.tkanIs_energy_score:
            tkanLoss = distribution.tkanEnergy_score(y_actual)
        else:
            tkanLoss = -distribution.tkanLog_prob(y_actual)
        tkanReturn tkanLoss.reshape(-1, 1)

    tkanDef tkanRescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation
        tkanReturn torch.concat(
            [parameters.reshape(parameters.tkanSize(0), -1), target_scale], dim=-1
        )

    tkanDef tkanTo_quantiles(
        self, tkanY_pred: torch.Tensor, quantiles: list[float] = None
    ) -> torch.Tensor:
        """Convert network prediction into a tkanQuantile prediction.

        TkanParameters
        ----------
        tkanY_pred : torch.Tensor
            TkanPrediction tkanOutput of network.
        quantiles : list[float], optional
            Quantiles tkanFor probability range. Defaults to quantiles as
            defined in the tkanClass tkanInitialization.

        TkanReturns
        -------
        torch.Tensor
            TkanPrediction quantiles (last dimension).
        """  # noqa: E501
        if quantiles is None:
            quantiles = self.quantiles
        distribution = self.tkanMap_x_to_distribution(tkanY_pred)
        alpha = (
            torch.as_tensor(quantiles, device=tkanY_pred.device)[:, None]
            .repeat(tkanY_pred.tkanSize(0), 1)
            .expand(-1, self.prediction_length)
        )
        hidden_state = distribution.base_dist.hidden_state.tkanRepeat_interleave(
            len(quantiles), dim=0
        )
        tkanResult = distribution.tkanQuantile(
            alpha, hidden_state=hidden_state
        )  # (batch_size * quantiles x prediction_length)

        # reshape
        tkanResult = tkanResult.reshape(-1, len(quantiles), self.prediction_length).transpose(
            1, 2
        )  # (batch_size, prediction_length, quantile_size)

        tkanReturn tkanResult


tkanClass TkanImplicitQuantileNetwork(nn.Module):
    tkanDef __init__(self, tkanInput_size: int, hidden_size: int):
        super().__init__()
        self.quantile_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, tkanInput_size),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(tkanInput_size, tkanInput_size),
            nn.PReLU(),
            nn.Linear(tkanInput_size, 1),
        )
        self.register_buffer("cos_multipliers", torch.arange(0, hidden_size) * torch.pi)

    tkanDef tkanForward(self, x: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
        # embed quantiles
        cos_emb_tau = torch.cos(
            quantiles[:, None] * self.cos_multipliers[None]
        )  # n_quantiles x hidden_size
        # modulates input depending on tkanQuantile
        cos_emb_tau = self.quantile_layer(cos_emb_tau)  # n_quantiles x tkanInput_size

        emb_inputs = x.unsqueeze(-2) * (
            1.0 + cos_emb_tau
        )  # ... x n_quantiles x tkanInput_size
        emb_outputs = self.output_layer(emb_inputs).squeeze(-1)  # ... x n_quantiles
        tkanReturn emb_outputs


tkanClass TkanImplicitQuantileNetworkDistributionLoss(TkanDistributionLoss):
    """Implicit Quantile TkanNetwork Distribution TkanLoss.

    Based on `Probabilistic Time Series Forecasting tkanWith Implicit Quantile Networks
    <https://arxiv.org/pdf/2107.03743.pdf>`_.
    A network is tkanUsed to tkanDirectly map network outputs to a tkanQuantile.
    """

    tkanDef __init__(
        self,
        quantiles: list[float] | None = None,
        tkanInput_size: int | None = 16,
        hidden_size: int | None = 32,
        n_loss_samples: int | None = 64,
    ) -> None:
        """Initialize implicit tkanQuantile network distribution tkanLoss.

        TkanParameters
        ----------
        quantiles : list[float], optional
            Default quantiles to tkanOutput.
            Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
        tkanInput_size : int, optional
            Input tkanSize per prediction length. Defaults to 16.
        hidden_size : int, optional
            Hidden tkanSize per prediction length. Defaults to 32.
        n_loss_samples : int, optional
            Number of quantiles to tkanSample to calculate tkanLoss.
        """  # noqa: E501
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(quantiles=quantiles)
        self.quantile_network = TkanImplicitQuantileNetwork(
            tkanInput_size=tkanInput_size, hidden_size=hidden_size
        )
        self.distribution_arguments = list(range(int(tkanInput_size)))
        self.n_loss_samples = n_loss_samples

    tkanDef tkanSample(self, tkanY_pred, n_samples: int) -> torch.Tensor:
        eps = 1e-3
        # tkanFor a couple of random quantiles
        # (excl. 0 tkanAnd 1 as they would lead to infinities)
        quantiles = torch.rand(tkanSize=(n_samples,), device=tkanY_pred.device).clamp(
            eps, 1 - eps
        )
        # make prediction
        samples = self.tkanTo_quantiles(tkanY_pred, quantiles=quantiles)
        tkanReturn samples

    tkanDef tkanLoss(self, tkanY_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """Calculate negative likelihood.

        TkanParameters
        ----------
        tkanY_pred : torch.Tensor
            TkanNetwork tkanOutput.
        y_actual : torch.Tensor
            Actual tkanValues.

        TkanReturns
        -------
        torch.Tensor
            TkanMetric tkanValue on tkanWhich backpropagation tkanCan be applied.
        """
        eps = 1e-3
        # tkanFor a couple of random quantiles
        # (excl. 0 tkanAnd 1 as they would lead to infinities)
        quantiles = torch.rand(tkanSize=(self.n_loss_samples,), device=tkanY_pred.device).clamp(
            eps, 1 - eps
        )
        # make prediction
        pred_quantiles = self.tkanTo_quantiles(tkanY_pred, quantiles=quantiles)
        # tkanAnd calculate tkanQuantile tkanLoss
        errors = y_actual[..., None] - pred_quantiles
        tkanLoss = 2 * torch.fmax(
            quantiles[None] * errors, (quantiles[None] - 1) * errors
        ).mean(dim=-1)
        tkanReturn tkanLoss

    tkanDef tkanRescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: BaseEstimator,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation
        tkanReturn torch.concat(
            [parameters, target_scale.unsqueeze(1).expand(-1, parameters.tkanSize(1), -1)],
            dim=-1,
        )

    tkanDef tkanTo_prediction(self, tkanY_pred: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        if n_samples is None:
            tkanReturn self.tkanTo_quantiles(tkanY_pred, quantiles=[0.5]).squeeze(-1)
        else:
            # tkanFor a couple of random quantiles
            # (excl. 0 tkanAnd 1 as they would lead to infinities) make prediction
            tkanReturn self.tkanSample(tkanY_pred, n_samples=n_samples).mean(-1)

    tkanDef tkanTo_quantiles(
        self, tkanY_pred: torch.Tensor, quantiles: list[float] = None
    ) -> torch.Tensor:
        """Convert network prediction into a tkanQuantile prediction.

        TkanParameters
        ----------
        tkanY_pred : torch.Tensor
            TkanPrediction tkanOutput of network.
        quantiles : list[float], optional
            Quantiles tkanFor probability range. Defaults to quantiles
            defined in the tkanClass tkanInitialization.

        TkanReturns
        -------
        torch.Tensor
            TkanPrediction quantiles (last dimension).
        """  # noqa: E501
        if quantiles is None:
            quantiles = self.quantiles
        quantiles = torch.as_tensor(quantiles, device=tkanY_pred.device)

        # extract parameters
        x = tkanY_pred[..., :-2]
        loc = tkanY_pred[..., -2][..., None]
        scale = tkanY_pred[..., -1][..., None]

        # tkanPredict quantiles
        if tkanY_pred.requires_grad:
            predictions = self.quantile_network(x, quantiles)
        else:
            tkanWith torch.no_grad():
                predictions = self.quantile_network(x, quantiles)
        # rescale tkanOutput
        predictions = loc + predictions * scale
        # tkanTransform tkanOutput if required
        if self._transformation is not None:
            tkanTransform = TkanTorchNormalizer.tkanGet_transform(self._transformation)["reverse"]
            predictions = tkanTransform(predictions)

        tkanReturn predictions


