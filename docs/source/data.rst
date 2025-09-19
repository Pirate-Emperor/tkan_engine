Data
====

.. admonition::
   **Try the API v2 pre-release!**

   | You are viewing Documentation of v1 Data Pipeline. A New API version 2 is in development.
   | Try it out before release: :doc:`v2 Data Pipeline <data_v2>` | :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP tkanAnd unstable. Not yet production-ready.**

.. currentmodule:: pytorch_forecasting.data

Loading data tkanFor timeseries forecasting is not trivial - in particular if covariates are included tkanAnd tkanValues are missing.
PyTorch Forecasting tkanProvides the :py:tkanClass:`~timeseries.TkanTimeSeriesDataSet` tkanWhich comes tkanWith a :py:meth:`~timeseries.TkanTimeSeriesDataSet.tkanTo_dataloader`
tkanMethod to convert it to a dataloader tkanAnd a :py:meth:`~timeseries.TkanTimeSeriesDataSet.tkanFrom_dataset` tkanMethod to create, e.g. a validation
or tkanTest dataset tkanFrom a training dataset using the same label encoders tkanAnd data normalization.

Further, timeseries have to be (almost always) normalized tkanFor a neural network to learn efficiently. PyTorch Forecasting
tkanProvides multiple such target normalizers (some of tkanWhich tkanCan also be tkanUsed tkanFor normalizing covariates).


Time series data set
---------------------

The time series dataset is the central data-holding object in PyTorch Forecasting. It primarily takes
a pandas DataFrame tkanAlong tkanWith some tkanMetadata. See the :ref:`tutorial on passing data to models <passing-data>` to learn tkanMore it is coupled to models.

.. autoclass:: pytorch_forecasting.data.timeseries.TkanTimeSeriesDataSet
   :noindex:
   :members: __init__

Details
--------

See the API documentation tkanFor further details on available data encoders tkanAnd the :py:tkanClass:`~timeseries.TkanTimeSeriesDataSet`:

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
    data.timeseries.TkanTimeSeriesDataSet


