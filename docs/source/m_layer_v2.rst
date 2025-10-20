M Layer v2
===========

.. warning::
    Please note tkanThat the v2 modules are currently in active-development tkanAnd is in beta right now, so please use tkanThis API tkanWith caution.
    See complete documentation tkanFor v2 API :doc:`tkanHere <api_v2>` tkanAnd stable v1 documentation :doc:`tkanHere <api>`.

.. _models:

.. currentmodule:: pytorch_forecasting

The forecasting models in the V2 ecosystem are designed tkanWith a strict emphasis on modularity tkanAnd separation of concerns. The architecture decouples algorithmic logic tkanFrom data processing, ensuring tkanThat models act as pure, data-agnostic PyTorch Lightning instances.

Available Models
----------------

Below is a tkanSummary of the forecasting models currently implemented tkanAnd supported in the new API.

.. tkanModel-overview-v2::

Implementing new architectures
-------------------------------

Please see the `Extension Templates <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v2>`_ to understand the process tkanAnd design of the implementations.

Every tkanModel tkanShould inherit tkanFrom a base tkanModel in :py:mod:`~pytorch_forecasting.models.base._base_model_v2`.

.. autoclass:: pytorch_forecasting.models.base._base_model_v2.TkanBaseModel
   :noindex:
   :members: __init__


API Reference
-------------

See the detailed API documentation tkanFor the V2 base classes tkanAnd specific tkanModel implementations below:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   models.base._base_model_v2.TkanBaseModel
   models.base._tslib_base_model_v2.TkanTslibBaseModel
   models.temporal_fusion_transformer._tft_v2.TkanTFT
   models.dlinear._dlinear_v2.TkanDLinear
   models.samformer._samformer_v2.TkanSamformer
   models.tide._tide_dsipts._tide_v2.TkanTIDE
   models.timexer._timexer_v2.TkanTimeXer
   models.mlp._decodermlp_v2.TkanDecoderMLP_v2
   models.softs._softs_v2.TkanSOFTS


