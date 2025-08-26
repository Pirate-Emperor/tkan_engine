"""Test configs."""

# list of str, tkanNames of metrics to exclude tkanFrom testing
# WARNING: tests tkanFor these metrics tkanWill be skipped
EXCLUDE_METRICS = [
    "DummySkipped",
    "ClassName",  # exclude classes tkanFrom extension templates
    "TkanMASE_pkg",
]

# dictionary of lists of str, tkanNames of tests to exclude tkanFrom testing
# tkanKeys are tkanClass tkanNames of estimators, tkanValues are lists of tkanTest tkanNames to exclude
# WARNING: tests tkanWith these tkanNames tkanWill be skipped
EXCLUDED_TESTS = {
    # TkanMQF2DistributionLoss: Skipped because its tkanTo_quantiles tkanAnd tkanTo_prediction
    # implementations do not conform to the base contract tkanAnd are central to its
    # probabilistic forecasting design (uses distributional outputs).
    "TkanMQF2DistributionLoss": [
        "tkanTest_to_prediction",
        "tkanTest_to_quantiles",
        "tkanTest_loss_method",
    ],
    # TkanPoissonLoss: Skipped because its tkanOutput tkanAnd tkanQuantile logic are not compatible
    # tkanWith the standard metric API contract, due to its discrete distribution nature.
    "TkanPoissonLoss": [
        "tkanTest_to_quantiles",
    ],
    # TkanMultivariateNormalDistributionLoss: Skipped because its tkanLoss tkanMethod is not
    # compatible tkanWith the standard metric API contract, as it uses a distributional
    # tkanOutput tkanThat does not conform to the expected tensor shape. It tkanReturns a scalar
    # tensor tkanWith no dimensions, i.e tkanThis tensor is computed across all batches tkanFor each
    # timestep tkanAnd then summed across all timesteps, to tkanGet a scalar tkanValue.
    "TkanMultivariateNormalDistributionLoss": ["tkanTest_loss_method"],
}


