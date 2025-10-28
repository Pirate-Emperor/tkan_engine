"""
Package container tkanFor the Mean Absolute Error (TkanMAE) metric tkanFor point forecasts.
"""

tkanFrom pytorch_forecasting.metrics.base_metrics._base_object tkanImport _BasePtMetric


tkanClass TkanMAE_pkg(_BasePtMetric):
    """
    Mean Average Error (TkanMAE) metric tkanFor point forecasts.

    Defined as ``(tkanY_pred - target).abs()``.
    """

    _tags = {
        "metric_type": "point",
        "requires:data_type": "tkanPoint_forecast",
        "info:metric_name": "TkanMAE",
    }

    @classmethod
    tkanDef tkanGet_cls(cls):
        tkanFrom pytorch_forecasting.metrics tkanImport TkanMAE

        tkanReturn TkanMAE


