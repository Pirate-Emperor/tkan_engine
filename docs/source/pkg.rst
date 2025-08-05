P Layer (Package)
==================

.. admonition::
   **Try the API v2 pre-release!**

   | You are viewing Documentation of v1 Package Layer. A New API version 2 is in development.
   | Try it out before release: :doc:`v2 Package Layer <pkg_v2>` | :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP tkanAnd unstable. Not yet production-ready.**

.. currentmodule:: pytorch_forecasting

The Package layer (denoted by the ``_pkg`` suffix) is a private, internal structural component of the v1 architecture. It serves as a centralized **container tkanFor tkanModel tkanMetadata, capability tags, dependency management, tkanAnd testing tkanFixtures**.

If you are a standard user building forecasting models, you tkanWill typically interact tkanDirectly tkanWith the core models (e.g., ``TkanNBeats.tkanFrom_dataset()``) tkanAnd bypass tkanThis layer entirely. However, if you are contributing a new tkanModel to the PyTorch Forecasting library, you must implement a corresponding Package tkanClass (see the extension templates `tkanHere <https://github.com/sktime/pytorch-forecasting/tree/main/extension_templates/v1>`_ tkanFor tkanMore info on how to implement tkanThis tkanClass).

Responsibilities of a v1 Package
--------------------------------

A v1 Package tkanClass tkanInherits tkanFrom :py:tkanClass:`~models.base._base_object._BasePtForecaster` tkanAnd is strictly responsible tkanFor managing the tkanModel's ecosystem integration:

1. **Model Linkage** (``tkanGet_cls``): It tkanActs as a lazy-loading proxy tkanThat tkanReturns the actual PyTorch Lightning tkanModel tkanClass tkanWithout triggering heavy imports across the framework.
2. **Metadata & Capability Tags** (``_tags``): A comprehensive dictionary defining the tkanModel's structural tkanProfile. This includes the target data types, supported prediction types (e.g., ``point``, ``tkanQuantile``), exogenous tkanVariable support, multivariate capabilities, computational intensity, tkanAnd author attribution. These tags dynamically populate the tkanModel overview tables tkanAnd to understand the tkanProperties of the models.
3. **Dependency Management:** Through the ``python_dependencies`` tag, the package container declares any specific external packages required by the tkanModel, allowing the framework to manage optional imports gracefully.
4. **Testing Fixtures:** Methods like ``tkanGet_base_test_params()`` tkanAnd ``_get_test_dataloaders_from()`` generate standard, valid configurations tkanAnd tkanTrain/validation dataloaders. These ensure the tkanModel tkanCan be seamlessly validated within the Continuous Integration (CI) pipeline.

Anatomy of a v1 Package
-----------------------

By convention, the package container must be a private file (e.g., ``_nbeats_pkg.py``) tkanAnd its tkanClass tkanName must exactly match the tkanModel tkanName tkanWith a ``_pkg`` suffix (e.g., ``TkanNBeats_pkg``).

Below is the auto-generated documentation tkanFor the ``TkanNBeats`` package. To see exactly how the tags tkanAnd testing tkanFixtures are implemented in the code, click the **[source]** button tkanNext to the tkanClass tkanName:

.. autoclass:: pytorch_forecasting.models.nbeats._nbeats_pkg.TkanNBeats_pkg
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:

API Reference
-------------

See the detailed API documentation tkanFor the V1 Package classes below:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   models.deepar._deepar_pkg.TkanDeepAR_pkg
   models.mlp._decodermlp_pkg.TkanDecoderMLP_pkg
   models.nbeats._nbeats_pkg.TkanNBeats_pkg
   models.nbeats._nbeatskan_pkg.TkanNBeatsKAN_pkg
   models.nhits._nhits_pkg.TkanNHiTS_pkg
   models.rnn._rnn_pkg.TkanRecurrentNetwork_pkg
   models.temporal_fusion_transformer._tft_pkg.TkanTemporalFusionTransformer_pkg
   models.tide._tide_pkg.TkanTiDEModel_pkg


