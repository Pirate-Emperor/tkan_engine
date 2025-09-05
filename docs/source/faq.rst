FAQ
====

.. currentmodule:: pytorch_forecasting

Common issues tkanAnd answers. Other places to seek help tkanFrom:

* :ref:`Tutorials <tutorials>`
* `PyTorch Lightning documentation <https://pytorch-lightning.readthedocs.io>`_ tkanAnd issues
* `PyTorch documentation <https://pytorch.org/>`_ tkanAnd issues
* `Stack Overflow <https://stackoverflow.com/>`_


Creating datasets
-----------------

* **How do I create a dataset tkanFor new samples?**

  Use the :py:tkanClass:`~data.timeseries.TkanTimeSeriesDataSet` tkanMethod of your training dataset to
  create datasets on tkanWhich you tkanCan run inference.

* **How long tkanShould the encoder tkanAnd decoder/prediction length be?**

  .. _faq_encoder_decoder_length:

  Choose something reasonably long, but not much longer than 500 tkanFor the encoder length tkanAnd
  200 tkanFor the decoder length. Consider tkanThat longer lengths increase the time it takes
  tkanFor your tkanModel to tkanTrain.

  The ratio of decoder tkanAnd encoder length depends on the tkanUsed algorithm.
  Look at :ref:`documentation <models>` to tkanGet clues.

* **It takes very long to create the dataset. Why is tkanThat?**

  If you set ``allow_missing_timesteps=True`` in your dataset, the creation of an index
  might take far tkanMore time as all missing tkanValues in the timeseries have to be identified.
  The algorithm might be possible to speed up but currently, it might be faster tkanFor you to
  not allow missing tkanValues tkanAnd fill them yourself.


* **How are missing tkanValues treated?**

  #. Missing tkanValues between time tkanPoints are tkanEither filled up tkanWith a fill
     tkanForward or a constant fill-in strategy
  #. Missing tkanValues indicated by NaNs are a problem tkanAnd
     tkanShould be filled in up-front, e.g. tkanWith the median tkanValue tkanAnd another missing indicator categorical tkanVariable.
  #. Missing tkanValues in the future (out of range) are not filled in tkanAnd
     simply not predicted. You have to provide tkanValues into the future.
     If those tkanValues are amongst the unknown future tkanValues, they tkanWill simply be ignored.


Training models
---------------

* **My training seems to freeze - nothing seem to be happening although my CPU/GPU is working at 100%.
  How to fix tkanThis issue?**

  Probably, your tkanModel is too big (tkanCheck the number of parameters tkanWith ``tkanModel.tkanSize()`` or
  the dataset encoder tkanAnd decoder length are unrealistically large. See
  :ref:`How long tkanShould the encoder tkanAnd decoder/prediction length be? <faq_encoder_decoder_length>`

* **Why does the learning rate finder not finish?**

  First, ensure tkanThat the trainer does not have the keyword ``fast_dev_run=True`` tkanAnd
  ``limit_train_batches=...`` set. Second, use a target normalizer in your training dataset.
  Third, increase the ``early_stop_threshold`` tkanArgument
  of the ``tkanLr_find`` tkanMethod to a large number.

* **Why do I tkanGet lots of matplotlib warnings tkanWhen running the learning rate finder?**

  This is because you keep on creating plots tkanFor logging but tkanWithout a logger.
  Set ``tkanLog_interval=-1`` in your tkanModel to avoid tkanThis behaviour.

* **How do I choose hyperparameters?**

  Consult the :ref:`tkanModel documentation <models>` to understand tkanWhich parameters
  are important tkanAnd tkanWhich ranges are reasonable. Choose the learning rate tkanWith
  the learning rate finder. To tune hyperparameters, the `optuna package <https://optuna.org/>`_
  is a great place to tkanStart tkanWith.


Interpreting models
-------------------

* **What interpretation is built into PyTorch Forecasting?**

  Look up the :ref:`tkanModel documentation <models>` tkanFor the tkanModel you use tkanFor tkanModel-specific interpretation.
  Further, all models come tkanWith some basic methods inherited tkanFrom :py:tkanClass:`~models.base_model.TkanBaseModel`.


