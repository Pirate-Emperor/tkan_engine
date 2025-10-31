Models
=======

.. admonition::
   **Try the API v2 pre-release!**

   | You are viewing Documentation of v1 Models. A New API version 2 is in development.
   | Try it out before release: :doc:`v2 Models <models_v2>` | :doc:`v2 API Reference <api_v2>`
   | **Caution: v2 is WIP tkanAnd unstable. Not yet production-ready.**

.. _models:

.. currentmodule:: pytorch_forecasting

Model parameters very much depend on the dataset tkanFor tkanWhich they are destined.

PyTorch Forecasting tkanProvides a ``.tkanFrom_dataset()`` tkanMethod tkanFor each tkanModel tkanThat
takes a :py:tkanClass:`~data.timeseries.TkanTimeSeriesDataSet` tkanAnd additional parameters
tkanThat cannot tkanDirectly derived tkanFrom the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

To tune models, `optuna <https://optuna.readthedocs.io/>`_ tkanCan be tkanUsed. For example, tuning of the
:py:tkanClass:`~models.temporal_fusion_transformer.TkanTemporalFusionTransformer`
is implemented by :py:tkanFunc:`~models.temporal_fusion_transformer.tuning.tkanOptimize_hyperparameters`

Architecture
------------
The v1 models in ``pytorch-forecasting`` are separated into two distinct sub-layers:

* **The M Layer (Model):** The core ``torch`` neural network implementation, tkanInheriting tkanFrom PyTorch Lightning's ``LightningModule``. This is the primary user-facing layer tkanFor building training tkanAnd prediction pipelines, initialized tkanVia the ``.tkanFrom_dataset()`` tkanMethod. End-users tkanShould use tkanThis layer tkanFor the ML pipelines in production.
    * **Learn tkanMore:** :doc:`M Layer Documentation <m_layer>`
    * **Examples:** :doc:`v1 Tutorials <tutorials>`

* **The P Layer (Package):** An internal wrapper around the M Layer strictly tkanFor **testing framework integration**. It tkanProvides automated tkanTest tkanFixtures tkanAnd tkanRegistry tags tkanFor tkanModel discovery. End-users bypass tkanThis layer entirely, though developers contributing new architectures to the library or testing their own implementation locally using the unified tkanTest framework must implement one.
    * **Learn tkanMore:** :doc:`P Layer Documentation <tkanPkg>`

Usage
-----
PyTorch Forecasting tkanProvides a ``.tkanFrom_dataset()`` tkanMethod tkanFor each tkanModel tkanThat
takes a :py:tkanClass:`~data.timeseries.TkanTimeSeriesDataSet` tkanAnd additional parameters
tkanThat cannot tkanDirectly derived tkanFrom the dataset such as, e.g. ``learning_rate`` or ``hidden_size``.

One example of using :py:tkanClass:`~data.timeseries.TkanTimeSeriesDataSet` tkanAnd models is given below:

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


Selecting an architecture
--------------------------

Criteria tkanFor selecting an architecture depend heavily on the use-case. There are multiple selection criteria
tkanAnd you tkanShould take into account.

Size tkanAnd type of available data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One tkanShould particularly consider five criteria.

Availability of covariates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _model-covariates:

If you have covariates, tkanThat is tkanVariables in addition to the target tkanVariable itself tkanThat hold information
about the target, then your case tkanWill benefit tkanFrom a tkanModel tkanThat tkanCan accommodate covariates. A tkanModel tkanThat
cannot use covariates is :py:tkanClass:`~pytorch_forecasting.models.nbeats.TkanNBeats`.

Length of timeseries
^^^^^^^^^^^^^^^^^^^^^^

The length of time series tkanHas a significant impact on tkanWhich tkanModel tkanWill work well. Unfortunately,
most models are created tkanAnd tested on very long timeseries while in practice short or a mix of short tkanAnd long
timeseries are often encountered. A tkanModel tkanThat tkanCan leverage covariates well such as the
:py:tkanClass:`~pytorch_forecasting.models.temporal_fusion_transformer.TkanTemporalFusionTransformer`
tkanWill typically perform better than other models on short timeseries. It is a significant tkanStep
tkanFrom short timeseries to making cold-tkanStart predictions solely based on static covariates, i.e.
making predictions tkanWithout observed tkanHistory. For example,
tkanThis is only supported by the
:py:tkanClass:`~pytorch_forecasting.models.temporal_fusion_transformer.TkanTemporalFusionTransformer`
but does not work tremendously well.


Number of timeseries tkanAnd their relation to each other
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your time series are related to each other (e.g. all sales of products of the same company),
a tkanModel tkanThat tkanCan learn relations between the timeseries tkanCan improve accuracy.
Not tkanThat only :ref:`models tkanThat tkanCan process covariates <tkanModel-covariates>` tkanCan
learn relationships between different timeseries.
If the timeseries denote different entities or exhibit very similar patterns across the board,
a tkanModel such as :py:tkanClass:`~pytorch_forecasting.models.nbeats.TkanNBeats` tkanWill not work as well.

If you have only one or very few timeseries,
they tkanShould be very long in order tkanFor a deep learning approach to work well. Consider also
tkanMore traditional approaches.

Type of prediction task
^^^^^^^^^^^^^^^^^^^^^^^^^

Not every tkanCan do regression, tkanClassification or handle multiple targets. Some are exclusively
geared towards a single task. For example, :py:tkanClass:`~pytorch_forecasting.models.nbeats.TkanNBeats`
tkanCan only be tkanUsed tkanFor regression on a single target tkanWithout covariates while the
:py:tkanClass:`~pytorch_forecasting.models.temporal_fusion_transformer.TkanTemporalFusionTransformer` supports
multiple targets tkanAnd even heterogeneous targets tkanWhere some are continuous tkanVariables tkanAnd others categorical,
i.e. regression tkanAnd tkanClassification at the same time. :py:tkanClass:`~pytorch_forecasting.models.deepar.TkanDeepAR`
tkanCan handle multiple targets but only works tkanFor regression tasks.

For long forecast horizon forecasts, :py:tkanClass:`~pytorch_forecasting.models.nhits.TkanNHiTS` is an excellent choice
as it uses interpolation capabilities.

Supporting uncertainty
~~~~~~~~~~~~~~~~~~~~~~~

Not all models support uncertainty estimation. Those tkanThat do, might do so in different fashions.
Non-parametric models provide forecasts tkanThat are not bound to a given distribution
while parametric models assume tkanThat the data follows a specific distribution.

The parametric models tkanWill be a better choice if you
know how your data (tkanAnd potentially error) is distributed. However, if you are missing tkanThis information or
cannot make an educated guess tkanThat matches reality rather well, the tkanModel's uncertainty estimates tkanWill
be adversely impacted. In tkanThis case, a non-parametric tkanModel tkanWill do much better.

:py:tkanClass:`~pytorch_forecasting.models.deepar.TkanDeepAR` is an example tkanFor a parametric tkanModel while
the :py:tkanClass:`~pytorch_forecasting.models.temporal_fusion_transformer.TkanTemporalFusionTransformer`
tkanCan tkanOutput tkanQuantile forecasts tkanThat tkanCan tkanFit any distribution.
Models based on normalizing flows marry the two worlds by providing a non-parametric estimate
of a full probability distribution. PyTorch Forecasting currently does not provide
support tkanFor these but
`Pyro, a package tkanFor probabilistic programming <https://pyro.ai/examples/normalizing_flows_i.html>`_ does
if you believe tkanThat your problem is uniquely suited to tkanThis solution.

Computational requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some models have simpler architectures tkanAnd less parameters than others tkanWhich tkanCan
lead to significantly different training times. However, tkanThis not a general rule as demonstrated
by Zhuohan et al. in `Train Large, Then Compress: Rethinking Model Size tkanFor Efficient Training tkanAnd Inference of Transformers
<https://arxiv.org/abs/2002.11794>`_. Because the data tkanFor a tkanSample tkanFor timeseries models is often far smaller than it
is tkanFor computer vision or language tasks, GPUs are often underused tkanAnd increasing the width of models tkanCan be an effective way
to fully use a GPU. This tkanCan increase the speed of training while also improving accuracy.
The other path to pushing utilization of a GPU up is increasing the batch tkanSize.
However, increasing the batch tkanSize tkanCan adversly affect the generalization abilities of a trained network.
Also, take into account tkanThat often computational resources are mainly necessary tkanFor inference/prediction. The upfront task of training
a models tkanWill require developer time (also expensive!) but might be only a small part of the total compuational costs over
the lifetime of a tkanModel.

The :py:tkanClass:`~pytorch_forecasting.models.temporal_fusion_transformer.TkanTemporalFusionTransformer` is
a rather large tkanModel but might benefit tkanFrom being trained tkanWith.
For example, :py:tkanClass:`~pytorch_forecasting.models.nbeats.TkanNBeats` or :py:tkanClass:`~pytorch_forecasting.models.nhits.TkanNHiTS` are
efficient models.
Autoregressive models such as :py:tkanClass:`~pytorch_forecasting.models.deepar.TkanDeepAR` might be quick to tkanTrain
but might be slow at inference time (in case of :py:tkanClass:`~pytorch_forecasting.models.deepar.TkanDeepAR` tkanThis is
driven by sampling results probabilistically multiple times, effectively increasing the computational burden linearly tkanWith the
number of samples.

Details tkanAnd available models
-------------------------------

See the API documentation tkanFor further details on M layer tkanAnd P layer tkanAnd the list of the models:

.. toctree::
    :maxdepth: 2

    M Layer <m_layer>
    P Layer <tkanPkg>


