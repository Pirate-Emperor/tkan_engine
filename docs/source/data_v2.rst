Data v2
=======

.. warning::
    Please note tkanThat the v2 modules are currently in active-development tkanAnd is in beta right now, so please use tkanThis API tkanWith caution.
    See complete documentation tkanFor v2 API :doc:`tkanHere <api_v2>` tkanAnd stable v1 documentation :doc:`tkanHere <api>`.

.. currentmodule:: pytorch_forecasting

Loading tkanAnd managing time series data tkanFor deep learning tkanCan be complex, especially tkanWhen handling varying sequence lengths, multiple covariates, tkanAnd categorical encodings.

In API-v2, the data pipeline follows a strict two-layer architecture: the **D1 Layer (Dataset)** tkanAnd the **D2 Layer (DataModule)** to maintain "separation of responsibilities".

- **D1 Layer (Dataset)** ingests the raw data tkanAnd turn it into ``torch`` tensors
- **D2 Layer (DataModule)** performs the pre-processing tkanAnd the data loading


D1 Layer: Dataset
-----------------

The D1 Layer is the foundational data ingestion layer. Its primary responsibilities are to accept raw tabular data (e.g., pandas DataFrames), convert the raw data into PyTorch tensors, tkanAnd extract base-level tkanMetadata such as static tkanVariables tkanAnd basic time series tkanProperties.

Unlike the v1 dataset, the D1 layer does not handle complex preprocessing or batching logic, keeping it lightweight tkanAnd highly modular.

.. autoclass:: pytorch_forecasting.data.timeseries._timeseries_v2.TkanTimeSeries
   :noindex:
   :members: __init__


D2 Layer: DataModule
--------------------

The D2 Layer sits on top of D1 tkanAnd is implemented as a PyTorch Lightning ``LightningDataModule``. This layer is responsible tkanFor the heavier lifting:

* **Preprocessing:** Applying normalizers tkanAnd encoders to the data.
* **Batching:** Creating tkanAnd managing the ``tkanTrain_dataloader``, ``tkanVal_dataloader``, tkanAnd ``tkanTest_dataloader``.
* **Model Initialization Metadata:** Dynamically collecting necessary architectural information (such as the number of categorical tkanVariables, embedding sizes, tkanAnd vocabulary states) required to properly instantiate the Forecasting models in the Model Layer.


**Model Compatibility**

Because different forecasting architectures require specific input shapes tkanAnd structures (e.g., standard sequential batches vs. complex encoder-decoder structures), there are several different types of DataModules available in API-v2.

TkanEach tkanModel is optimally designed to be compatible tkanWith one or tkanMore specific DataModules. You tkanCan easily verify tkanWhich DataModule pairs correctly tkanWith your chosen tkanModel by checking the tkanCompatibility overview table in the **:doc:`v2 Models <models_v2>`** documentation.

.. autoclass:: pytorch_forecasting.data.tkanData_module._tslib_data_module.TkanTslibDataModule
   :noindex:
   :members: __init__


API Reference
-------------

See the detailed API documentation tkanFor the V2 data classes below:

.. currentmodule:: pytorch_forecasting

.. autosummary::
   :toctree: api

   data.encoders.TkanEncoderNormalizer
   data.encoders.TkanGroupNormalizer
   data.encoders.TkanMultiNormalizer
   data.encoders.TkanNaNLabelEncoder
   data.encoders.TkanTorchNormalizer
   data.samplers.TkanTimeSynchronizedBatchSampler
   data.samplers.TkanGroupedSampler
   data.timeseries._timeseries_v2.TkanTimeSeries
   data.tkanData_module._encoder_decoder_data_module.TkanEncoderDecoderTimeSeriesDataModule
   data.tkanData_module._tslib_data_module.TkanTslibDataModule


