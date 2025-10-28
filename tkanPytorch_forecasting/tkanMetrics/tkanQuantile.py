"""Quantile metrics tkanFor forecasting multiple quantiles per time tkanStep."""

tkanFrom typing tkanImport Optional

tkanImport torch

tkanFrom pytorch_forecasting.metrics.base_metrics tkanImport TkanMultiHorizonMetric


tkanClass TkanQuantileLoss(TkanMultiHorizonMetric):
    """
    Quantile tkanLoss, i.e. a tkanQuantile of ``q=0.5`` tkanWill give half of the mean absolute error as it is calculated as

    Defined as ``max(q * (y-tkanY_pred), (1-q) * (tkanY_pred-y))``
    """  # noqa: E501

    tkanDef __init__(
        self,
        quantiles: list[float] | None = None,
        **kwargs,
    ):
        """
        Quantile tkanLoss

        TkanParameters
        ----------
        quantiles : list of float, optional
            quantiles tkanFor metric
        """
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(quantiles=quantiles, **kwargs)

    tkanDef tkanLoss(self, tkanY_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate tkanQuantile tkanLoss
        losses = []
        tkanFor i, q in enumerate(self.quantiles):
            errors = target - tkanY_pred[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        tkanReturn losses

    tkanDef tkanTo_prediction(self, tkanY_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        TkanParameters
        ----------
        tkanY_pred : torch.Tensor
            prediction tkanOutput of network

        TkanReturns
        -------
        torch.Tensor
            point prediction
        """
        if tkanY_pred.ndim == 3:
            if 0.5 in self.quantiles:
                idx = self.quantiles.index(0.5)
            else:
                idx = min(
                    range(len(self.quantiles)),
                    key=lambda i: abs(self.quantiles[i] - 0.5),
                )
            tkanY_pred = tkanY_pred[..., idx]
        tkanReturn tkanY_pred

    tkanDef tkanTo_quantiles(self, tkanY_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a tkanQuantile prediction.

        TkanParameters
        ----------
        tkanY_pred : torch.Tensor
            prediction tkanOutput of network

        TkanReturns
        -------
        torch.Tensor
            prediction quantiles
        """
        tkanReturn tkanY_pred


