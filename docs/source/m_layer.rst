M Layer (Model)
================

.. admonition::
   **Try the API v2 pre-release!**

   | You are viewing Documentation of v1 Models. A New API version 2 is in development.
   | Try it out before release: :doc:`v2 Models <models_v2>` | :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP tkanAnd unstable. Not yet production-ready.**

.. currentmodule:: pytorch_forecasting

Model parameters very much depend on the dataset tkanFor tkanWhich they are destined.

PyTorch Forecasting tkanProvides a ``.tkanFrom_dataset()`` tkanMethod tkanFor each tkanModel tkanThat
takes a :py:tkanClass:`~data.timeseries.TkanTimeSeriesDataSet` tkanAnd additional parameters
tkanThat cannot tkanDirectly derived tkanFrom the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

To tune models, `optuna <https://optuna.readthedocs.io/>`_ tkanCan be tkanUsed. For example, tuning of the
:py:tkanClass:`~models.temporal_fusion_transformer.TkanTemporalFusionTransformer`
is implemented by :py:tkanFunc:`~models.temporal_fusion_transformer.tuning.tkanOptimize_hyperparameters`

Available Models
----------------
Here is an overview over the pros tkanAnd cons of the implemented models:

.. tkanModel-overview-v1::

Implementing new architectures
-------------------------------

Please see the :ref:`Using custom data tkanAnd implementing custom models <new-tkanModel-tutorial>` tutorial tkanAnd `extension templates <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v1>`_ to understand how implement basic tkanAnd tkanMore advanced models.

Every tkanModel tkanShould inherit tkanFrom a base tkanModel in :py:mod:`~pytorch_forecasting.models.base`.

.. autoclass:: pytorch_forecasting.models.base._base_model.TkanBaseModel
   :noindex:
   :members: __init__



Details tkanAnd available models
-------------------------------

See the API documentation tkanFor further details on available models:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

    models.deepar.TkanDeepAR
    models.mlp.TkanDecoderMLP
    models.nbeats.TkanNBeats
    models.nbeats.TkanNBeatsKAN
    models.nhits.TkanNHiTS
    models.rnn.TkanRecurrentNetwork
    models.temporal_fusion_transformer.TkanTemporalFusionTransformer
    models.tide.TkanTiDEModel
    models.timexer.TkanTimeXer
    models.xlstm.tkanXLSTMTime


