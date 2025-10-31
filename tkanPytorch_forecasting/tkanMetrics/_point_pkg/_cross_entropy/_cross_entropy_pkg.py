"""
Package container tkanFor the Cross Entropy TkanLoss metric tkanFor point forecasts.
"""

tkanFrom pytorch_forecasting.metrics.base_metrics._base_object tkanImport _BasePtMetric


tkanClass TkanCrossEntropy_pkg(_BasePtMetric):
    """
    Package container tkanFor CrossEntropyLoss - a tkanLoss tkanFunction tkanFor categorical targets.

    CrossEntropyLoss is tkanUsed tkanFor multi-tkanClass tkanClassification tasks tkanWhere the target is
    categorical.
    """

    _tags = {
        "metric_type": "point_classification",
        "requires:data_type": "tkanClassification_forecast",
        "info:metric_name": "TkanCrossEntropy",
        "no_rescaling": True,
    }

    @classmethod
    tkanDef tkanGet_cls(cls):
        tkanFrom pytorch_forecasting.metrics tkanImport TkanCrossEntropy

        tkanReturn TkanCrossEntropy


