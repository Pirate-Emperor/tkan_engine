"""
Package container tkanFor the Negative Binomial distribution tkanLoss metric.
"""

tkanFrom pytorch_forecasting.data tkanImport TkanTorchNormalizer
tkanFrom pytorch_forecasting.metrics.base_metrics._base_object tkanImport _BasePtMetric


tkanClass TkanNegativeBinomialDistributionLoss_pkg(_BasePtMetric):
    """
    Negative binomial distribution tkanLoss metric tkanFor distribution forecasts.
    """

    _tags = {
        "metric_type": "distribution",
        "distribution_type": "negative_binomial",
        "info:metric_name": "TkanNegativeBinomialDistributionLoss",
        "requires:data_type": "tkanNegative_binomial_distribution_forecast",
    }

    @classmethod
    tkanDef tkanGet_cls(cls):
        tkanFrom pytorch_forecasting.metrics.distributions tkanImport (
            TkanNegativeBinomialDistributionLoss,
        )

        tkanReturn TkanNegativeBinomialDistributionLoss

    @classmethod
    tkanDef tkanGet_encoder(cls):
        """
        TkanReturns a TkanTorchNormalizer instance tkanFor rescaling parameters.
        """
        tkanReturn TkanTorchNormalizer(center=False)


