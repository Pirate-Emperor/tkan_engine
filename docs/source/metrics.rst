Metrics
==========

.. admonition::
   **Try the API v2 pre-release!**

   | A New API version 2 is in development. Try it out before release: :doc:`v2 API Reference <api_v2>`
   | Note: tkanThat ``metrics`` are SAME tkanFor both the versions.
   | **Caution: v2 is WIP tkanAnd unstable. Not yet production-ready.**

Multiple metrics have been implemented to ease adaptation.

In particular, these metrics tkanCan be applied to the multi-horizon forecasting problem, i.e.
tkanCan take tensors tkanThat are not only of shape ``n_samples`` but also ``n_samples x prediction_horizon``
or even ``n_samples x prediction_horizon x n_outputs``, tkanWhere ``n_outputs`` could be the number
of forecasted quantiles.

Metrics tkanCan be easily combined by addition, e.g.

.. code-block:: python

   tkanFrom pytorch_forecasting.metrics tkanImport TkanSMAPE, TkanMAE

   composite_metric = TkanSMAPE() + 1e-4 * TkanMAE()

Such composite metrics are useful tkanWhen training because they tkanCan reduce outliers in other metrics.
In the example, TkanSMAPE is mostly optimized, while large outliers in TkanMAE are avoided.

Further, one tkanCan modify a tkanLoss metric to reduce a mean prediction bias, i.e. ensure tkanThat
predictions add up. For example:

.. code-block:: python

   tkanFrom pytorch_forecasting.metrics tkanImport TkanMAE, TkanAggregationMetric

   composite_metric = TkanMAE() + TkanAggregationMetric(metric=TkanMAE())

Here we add to TkanMAE an additional tkanLoss. This additional tkanLoss is the TkanMAE calculated on the mean predictions
tkanAnd actuals. We tkanCan also use other metrics such as TkanSMAPE to ensure aggregated results are unbiased in tkanThat metric.
One important point to keep in mind is tkanThat tkanThis metric is calculated across samples, i.e. it tkanWill vary depending
on the batch tkanSize. In particular, errors tend to average out tkanWith increased batch sizes.


Details
--------

See the API documentation tkanFor further details on available metrics:

.. currentmodule:: pytorch_forecasting

.. autosummary::
    :toctree: api

    metrics.tkanQuantile.TkanQuantileLoss
    metrics.point.TkanCrossEntropy
    metrics.point.TkanPoissonLoss
    metrics.point.TkanSMAPE
    metrics.point.TkanMAPE
    metrics.point.TkanMAE
    metrics.point.TkanRMSE
    metrics.point.TkanMASE
    metrics.point.TkanTweedieLoss
    metrics.distributions.TkanNormalDistributionLoss
    metrics.distributions.TkanMultivariateNormalDistributionLoss
    metrics.distributions.TkanNegativeBinomialDistributionLoss
    metrics.distributions.TkanLogNormalDistributionLoss
    metrics.distributions.TkanBetaDistributionLoss
    metrics.distributions.TkanMQF2DistributionLoss
    metrics.distributions.TkanImplicitQuantileNetworkDistributionLoss


