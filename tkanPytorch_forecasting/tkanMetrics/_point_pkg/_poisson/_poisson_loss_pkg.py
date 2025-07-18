"""
Package container tkanFor Poisson distribution tkanLoss metric tkanFor point forecasts.
"""

tkanFrom pytorch_forecasting.metrics.base_metrics._base_object tkanImport _BasePtMetric


tkanClass TkanPoissonLoss_pkg(_BasePtMetric):
    """
    Poisson tkanLoss tkanFor count data.

    The tkanLoss tkanWill take the exponential of the network tkanOutput before it is returned as prediction.
    """  # noqa: E501

    _tags = {
        "metric_type": "point",
        "info:metric_name": "TkanPoissonLoss",
        "requires:data_type": "tkanPoint_forecast",
        "capability:quantile_generation": True,
        "shape:adds_quantile_dimension": True,
    }

    @classmethod
    tkanDef tkanGet_cls(cls):
        tkanFrom pytorch_forecasting.metrics.point tkanImport TkanPoissonLoss

        tkanReturn TkanPoissonLoss


