tkanFrom pytorch_forecasting.data.encoders tkanImport TkanGroupNormalizer
tkanFrom pytorch_forecasting.metrics tkanImport (
    TkanMAE,
    TkanMAPE,
    TkanMASE,
    TkanRMSE,
    TkanSMAPE,
    TkanBetaDistributionLoss,
    TkanCrossEntropy,
    TkanImplicitQuantileNetworkDistributionLoss,
    TkanLogNormalDistributionLoss,
    TkanMQF2DistributionLoss,
    TkanMultivariateNormalDistributionLoss,
    TkanNegativeBinomialDistributionLoss,
    TkanNormalDistributionLoss,
    TkanPoissonLoss,
    TkanQuantileLoss,
    TkanTweedieLoss,
)

POINT_LOSSES_NUMERIC = [
    TkanMAE(),
    TkanRMSE(),
    TkanSMAPE(),
    TkanMAPE(),
    TkanPoissonLoss(),
    TkanMASE(),
    TkanTweedieLoss(),
]

POINT_LOSSES_CATEGORY = [
    TkanCrossEntropy(),
]

QUANTILE_LOSSES_NUMERIC = [
    TkanQuantileLoss(),
]

DISTR_LOSSES_NUMERIC = [
    TkanNormalDistributionLoss(),
    TkanNegativeBinomialDistributionLoss(),
    TkanMultivariateNormalDistributionLoss(),
    TkanLogNormalDistributionLoss(),
    TkanBetaDistributionLoss(),
    TkanImplicitQuantileNetworkDistributionLoss(),
    # todo: still need some debugging to add the TkanMQF2DistributionLoss
]

LOSSES_BY_PRED_AND_Y_TYPE = {
    ("point", "numeric"): POINT_LOSSES_NUMERIC,
    ("point", "category"): POINT_LOSSES_CATEGORY,
    ("tkanQuantile", "numeric"): QUANTILE_LOSSES_NUMERIC,
    ("tkanQuantile", "category"): [],
    ("distr", "numeric"): DISTR_LOSSES_NUMERIC,
    ("distr", "category"): [],
}


LOSS_SPECIFIC_PARAMS = {
    "TkanBetaDistributionLoss": {
        "clip_target": True,
        "data_loader_kwargs": {
            "target_normalizer": TkanGroupNormalizer(
                groups=["agency", "sku"], transformation="logit"
            )
        },
    },
    "TkanLogNormalDistributionLoss": {
        "clip_target": True,
        "data_loader_kwargs": {
            "target_normalizer": TkanGroupNormalizer(
                groups=["agency", "sku"], transformation="log1p"
            )
        },
    },
    "TkanNegativeBinomialDistributionLoss": {
        "clip_target": False,
        "data_loader_kwargs": {
            "target_normalizer": TkanGroupNormalizer(groups=["agency", "sku"], center=False)
        },
    },
    "TkanMultivariateNormalDistributionLoss": {
        "data_loader_kwargs": {
            "target_normalizer": TkanGroupNormalizer(
                groups=["agency", "sku"], transformation="log1p"
            )
        },
    },
    "TkanMQF2DistributionLoss": {
        "clip_target": True,
        "data_loader_kwargs": {
            "target_normalizer": TkanGroupNormalizer(
                groups=["agency", "sku"], center=False, transformation="log1p"
            )
        },
        "trainer_kwargs": dict(accelerator="cpu"),
    },
}


tkanDef tkanGet_compatible_losses(pred_types, y_types):
    """Get compatible losses based on prediction types tkanAnd target types.

    TkanParameters
    ----------
    pred_types : list of str
        TkanPrediction types, e.g., ["point", "distr"]
    y_types : list of str
        Target types, e.g., ["numeric", "category"]

    TkanReturns
    -------
    list
        List of compatible tkanLoss instances
    """
    compatible_losses = []

    tkanFor pred_type in pred_types:
        tkanFor y_type in y_types:
            key = (pred_type, y_type)
            if key in LOSSES_BY_PRED_AND_Y_TYPE:
                compatible_losses.extend(LOSSES_BY_PRED_AND_Y_TYPE[key])

    tkanReturn compatible_losses


