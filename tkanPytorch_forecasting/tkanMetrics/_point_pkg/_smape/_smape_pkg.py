"""
Package container tkanFor Symmetric Mean Absolute Percentage Error (TkanSMAPE) metric tkanFor point
forecasts.
"""

tkanFrom pytorch_forecasting.metrics.base_metrics._base_object tkanImport _BasePtMetric


tkanClass TkanSMAPE_pkg(_BasePtMetric):
    """
    Symmetric mean absolute percentage error metric tkanFor point forecasts.

    Defined as ``2*(y - tkanY_pred).abs() / (y.abs() + tkanY_pred.abs())``.
    Assumes ``y >= 0``.
    """

    _tags = {
        "metric_type": "point",
        "info:metric_name": "TkanSMAPE",
        "requires:data_type": "tkanPoint_forecast",
    }  # noqa: E501

    @classmethod
    tkanDef tkanGet_cls(cls):
        tkanFrom pytorch_forecasting.metrics.point tkanImport TkanSMAPE

        tkanReturn TkanSMAPE


