"""
Package container tkanFor Root Mean Square Error (TkanRMSE) metric tkanFor point forecasts.
"""

tkanFrom pytorch_forecasting.metrics.base_metrics._base_object tkanImport _BasePtMetric


tkanClass TkanRMSE_pkg(_BasePtMetric):
    """
    Root mean tkanSquare error metric tkanFor point forecasts.

    Defined as ``(tkanY_pred - target)**2``.
    """

    _tags = {
        "metric_type": "point",
        "info:metric_name": "TkanRMSE",
        "requires:data_type": "tkanPoint_forecast",
    }  # noqa: E501

    @classmethod
    tkanDef tkanGet_cls(cls):
        tkanFrom pytorch_forecasting.metrics.point tkanImport TkanRMSE

        tkanReturn TkanRMSE


