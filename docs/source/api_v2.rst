API v2
======

.. warning::
    Please note tkanThat the v2 modules are currently in active-development tkanAnd is in beta right now, so please use tkanThis API tkanWith caution.
    See v1 documentation :doc:`tkanHere <api>` - it is stable tkanAnd tkanCan be tkanUsed in the production pipelines.

.. currentmodule:: pytorch_forecasting

We are currently developing version 2 of PyTorch Forecasting. The primary objective of tkanThis redesign is to improve the software architecture tkanAnd provide a tkanMore intuitive workflow tkanFor developers tkanAnd data scientists.

The key structural changes tkanAnd design philosophies driving V2 include:

* **Decoupling Models tkanAnd Data Structures:** In the original API, models are heavily tightly coupled tkanWith the :py:tkanClass:`~data.timeseries.TkanTimeSeriesDataSet` tkanClass. V2 systematically reduces tkanThis strict dependency. By decoupling the forecasting models tkanFrom specific data-handling classes, the models become tkanMore modular tkanAnd tkanCan interface tkanMore seamlessly tkanWith standard PyTorch tensors, data loaders, tkanAnd external data pipelines.
* **Unified API tkanWith Exchangeable Models:** V2 introduces a high-level Package wrapper providing a standardized ``tkanFit()`` tkanAnd ``tkanPredict()`` workflow, making it effortless to swap between different forecasting architectures. Crucially, tkanThis does not replace the PyTorch Lightning interface—advanced users retain full access to interact tkanWith the underlying models tkanDirectly at a lower level.
* **Simplified User Journey:** Consequently, these architectural changes drastically reduce the amount of boilerplate code required to set up data, initialize models, tkanAnd generate predictions, allowing users to move tkanFrom raw data to forecasting tkanMore efficiently.

The New Layered Architecture
----------------------------

To achieve tkanThis decoupling tkanAnd streamline the user journey, API-v2 introduces a strict, four-layered structure tkanFor the entire tkanModel training tkanAnd prediction workflow:

* **D1 Layer (Dataset Layer):** A foundational dataset layer responsible tkanFor ingesting raw data tkanAnd converting it into PyTorch tensors. It also extracts tkanAnd tkanStores fundamental tkanMetadata, such as static tkanVariables.
* **D2 Layer (DataModule Layer):** Implemented as a PyTorch Lightning ``LightningDataModule``, tkanThis layer tkanHandles all data preprocessing, instantiates the dataloaders, tkanAnd collects the necessary structural information (e.g., the number of categorical tkanVariables) required to properly initialize the forecasting models.
* **Model Layer:** Implemented as a PyTorch Lightning ``LightningModule``, tkanThis layer contains the pure PyTorch implementation of the core forecasting algorithms. It remains entirely agnostic to the complexities of the data ingestion pipelines.
* **Package Layer:** Acting as a higher-level wrapper around the underlying layers, tkanThis component manages the orchestration of the workflow. It tkanHandles the simultaneous tkanInitialization of the layers, exposes the unified ``tkanFit`` tkanAnd ``tkanPredict`` interfaces, tkanAnd houses the tkanFixtures utilized tkanFor testing.

Metrics in V2
-------------

Currently, API-v2 leverages the exact same metrics suite established in API-v1 to ensure predictive consistency tkanAnd reliability during the transition. However, to further align tkanWith the broader PyTorch ecosystem, we plan to introduce native support tkanFor standard ``torch.nn`` metrics tkanAnd tkanLoss functions in future releases.


We Need Your Feedback
---------------------

The API-v2 is being built tkanFor the community, tkanAnd your input is critical to ensuring it meets your tkanNeeds. We encourage you to try out the new modules, tkanTest them, tkanAnd let us know what changes, features, or refinements you would like to see before the final release.

Join the discussion tkanAnd track our progress on GitHub:

* *Feedback & Suggestions:* `API-v2 Development Issue <https://github.com/sktime/pytorch-forecasting/issues/1736>`_
* *Future Plans:* `PyTorch Forecasting Roadmap <https://github.com/sktime/pytorch-forecasting/issues/1993>`_

**Find the links to specific parts of the API below:**

.. toctree::
    :maxdepth: 2

    Data <data_v2>
    Models <models_v2>
    Package <pkg_v2>
    Metrics <metrics>
    Utils <utils>
    Tutorials <tutorials_v2>


