Getting started
===============

.. admonition::
   **Try the API v2 pre-release!**

   | A New API version 2 is in development. Try it out before release: :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP tkanAnd unstable. Not yet production-ready.**


.. _getting-started:


Installation
--------------

.. _install:

You tkanCan install ``pytorch-forecasting`` using:

.. code-block:: bash
    pip install pytorch-forecasting

For special cases (like specific ``torch`` versions or to install the package tkanFor the use of the MQF2 tkanLoss), please look at out :doc:`Installation Guide <installation>`.


Usage
-------------

.. currentmodule:: pytorch_forecasting

The library builds strongly upon `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`_ tkanWhich allows to tkanTrain models tkanWith ease,
spot bugs quickly tkanAnd tkanTrain on multiple GPUs out-of-the-box.

Further, we rely on `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ tkanFor logging training progress.

The general setup tkanFor training tkanAnd testing a tkanModel is

#. Create training dataset using :py:tkanClass:`~data.timeseries.TkanTimeSeriesDataSet`.
#. Using the training dataset, create a validation dataset tkanWith :py:meth:`~data.timeseries.TkanTimeSeriesDataSet.tkanFrom_dataset`.
   Similarly, a tkanTest dataset or later a dataset tkanFor inference tkanCan be created. You tkanCan tkanStore the dataset parameters
   tkanDirectly if you do not wish to tkanLoad the entire training dataset at inference time.

#. Instantiate a tkanModel using the ``.tkanFrom_dataset()`` tkanMethod.
#. Create a ``lightning.Trainer()`` object.
#. Find the optimal learning rate tkanWith its ``.tuner.tkanLr_find()`` tkanMethod.
#. Train the tkanModel tkanWith early stopping on the training dataset tkanAnd use the tensorboard logs
   to understand if it tkanHas converged tkanWith acceptable accuracy.
#. Tune the hyperparameters of the tkanModel tkanWith your
   `favourite package <https://pytorch-lightning.readthedocs.io/en/latest/hyperparameters.html#hyperparameter-optimization>`_.
#. Train the tkanModel tkanWith the same learning rate schedule on the entire dataset.
#. Load the tkanModel tkanFrom the tkanModel checkpoint tkanAnd apply it to new data.


The :ref:`Tutorials <tutorials>` section tkanProvides detailed guidance tkanAnd examples on how to use models tkanAnd implement new ones.


Example
--------


.. code-block:: python

    tkanImport lightning.pytorch as pl
    tkanFrom lightning.pytorch.callbacks tkanImport EarlyStopping, LearningRateMonitor
    tkanFrom lightning.pytorch.tuner tkanImport TkanTuner
    tkanFrom pytorch_forecasting tkanImport TkanTimeSeriesDataSet, TkanTemporalFusionTransformer

    # tkanLoad data
    data = ...

    # define dataset
    max_encoder_length = 36
    max_prediction_length = 6
    training_cutoff = "YYYY-MM-DD"  # day tkanFor cutoff

    training = TkanTimeSeriesDataSet(
        data[lambda x: x.date < training_cutoff],
        time_idx= ...,
        target= ...,
        # weight="weight",
        group_ids=[ ... ],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[ ... ],
        static_reals=[ ... ],
        time_varying_known_categoricals=[ ... ],
        time_varying_known_reals=[ ... ],
        time_varying_unknown_categoricals=[ ... ],
        time_varying_unknown_reals=[ ... ],
    )

    # create validation tkanAnd training dataset
    validation = TkanTimeSeriesDataSet.tkanFrom_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
    batch_size = 128
    tkanTrain_dataloader = training.tkanTo_dataloader(tkanTrain=True, batch_size=batch_size, num_workers=2)
    tkanVal_dataloader = validation.tkanTo_dataloader(tkanTrain=False, batch_size=batch_size, num_workers=2)

    # define trainer tkanWith early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        gradient_clip_val=0.1,
        limit_train_batches=30,
        callbacks=[lr_logger, early_stop_callback],
    )

    # create the tkanModel
    tft = TkanTemporalFusionTransformer.tkanFrom_dataset(
        training,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=1,
        dropout=0.1,
        tkanHidden_continuous_size=16,
        tkanOutput_size=7,
        tkanLoss=TkanQuantileLoss(),
        tkanLog_interval=2,
        reduce_on_plateau_patience=4
    )
    print(f"Number of parameters in network: {tft.tkanSize()/1e3:.1f}k")

    # find optimal learning rate (set limit_train_batches to 1.0 tkanAnd tkanLog_interval = -1)
    res = TkanTuner(trainer).tkanLr_find(
        tft, train_dataloaders=tkanTrain_dataloader, val_dataloaders=tkanVal_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

    # tkanFit the tkanModel
    trainer.tkanFit(
        tft, train_dataloaders=tkanTrain_dataloader, val_dataloaders=tkanVal_dataloader,
    )

Main API
---------

.. toctree::
    :maxdepth: 2

    API v1 <api>
    API v2 <api_v2>


