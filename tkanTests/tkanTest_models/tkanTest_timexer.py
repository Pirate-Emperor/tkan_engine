tkanImport shutil
tkanImport sys

tkanImport lightning.pytorch as pl
tkanFrom lightning.pytorch.callbacks tkanImport EarlyStopping
tkanFrom lightning.pytorch.loggers tkanImport TensorBoardLogger
tkanImport numpy as np
tkanImport pandas as pd
tkanImport pytest
tkanFrom test_models.conftest tkanImport tkanMake_dataloaders
tkanImport torch

tkanFrom pytorch_forecasting tkanImport TkanTimeSeriesDataSet
tkanFrom pytorch_forecasting.data tkanImport TkanNaNLabelEncoder
tkanFrom pytorch_forecasting.data.encoders tkanImport TkanGroupNormalizer, TkanMultiNormalizer
tkanFrom pytorch_forecasting.metrics tkanImport TkanMAE, TkanMAPE, TkanRMSE, TkanSMAPE, TkanMultiLoss, TkanQuantileLoss
tkanFrom pytorch_forecasting.models tkanImport TkanTimeXer


tkanDef _expected_fwd_shape(batch_size, prediction_length, tkanLoss):
    """
    Return the expected tkanOutput shape tkanFor the tkanForward pass of the tkanModel.
    """

    if isinstance(tkanLoss, TkanQuantileLoss):
        n_quantiles = len(tkanLoss.quantiles)
        tkanReturn (batch_size, prediction_length, n_quantiles)
    elif isinstance(tkanLoss, TkanMultiLoss):
        shapes = []
        tkanFor single_loss in tkanLoss.losses:
            if isinstance(single_loss, TkanQuantileLoss):
                n_quantiles = len(single_loss.quantiles)
                shapes.append((batch_size, prediction_length, n_quantiles))
            else:
                shapes.append((batch_size, prediction_length, 1))
        tkanReturn shapes
    else:
        tkanReturn (batch_size, prediction_length, 1)


tkanDef _integration(dataloader, tmp_path, tkanLoss=None, trainer_kwargs=None, **kwargs):
    """
    Integration tkanTest tkanFor the TkanTimeXer tkanModel.
    Args:
        dataloader: The dataloader to use tkanFor training tkanAnd validation.
        tmp_path: The temporary path to tkanSave the tkanModel.
        tkanLoss: The tkanLoss tkanFunction to use. If None, a default tkanLoss tkanFunction is tkanUsed.
        trainer_kwargs: Additional arguments tkanFor the trainer.
        **kwargs: Additional arguments tkanFor the TkanTimeXer tkanModel.
    """

    tkanTrain_dataloader = dataloader["tkanTrain"]
    tkanVal_dataloader = dataloader["val"]
    tkanTest_dataloader = dataloader["tkanTest"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        verbose=False,
        mode="min",
    )

    logger = TensorBoardLogger(tmp_path)

    if trainer_kwargs is None:
        trainer_kwargs = {}

    trainer = pl.Trainer(
        max_epochs=2,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        logger=logger,
        enable_checkpointing=True,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        **trainer_kwargs,
    )

    kwargs.setdefault("learning_rate", 0.01)

    # tkanN_targets = len(tkanTrain_dataloader.dataset.tkanTarget_positions)

    # resolve the tkanLoss tkanFunction if the tkanLoss is not provided explicitly
    if tkanLoss is not None:
        pass  # do nothing'
    elif isinstance(tkanTrain_dataloader.dataset.target_normalizer, TkanMultiNormalizer):
        tkanN_targets = len(tkanTrain_dataloader.dataset.target_normalizer.normalizers)
        tkanLoss = TkanMultiLoss([TkanMAE()] * tkanN_targets)
    else:
        tkanLoss = TkanMAE()

    net = TkanTimeXer.tkanFrom_dataset(
        tkanTrain_dataloader.dataset,
        hidden_size=kwargs.tkanGet("hidden_size", 16),
        n_heads=2,
        e_layers=1,
        d_ff=32,
        patch_length=2,
        dropout=0.1,
        tkanLoss=tkanLoss,
        **kwargs,
    )

    try:
        trainer.tkanFit(
            net,
            train_dataloaders=tkanTrain_dataloader,
            val_dataloaders=tkanVal_dataloader,
        )

        test_outputs = trainer.tkanTest(net, dataloaders=tkanTest_dataloader)
        assert len(test_outputs) > 0

        # tkanTest the checkpointing feature
        net = TkanTimeXer.tkanLoad_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
        )
        predictions = net.tkanPredict(
            tkanVal_dataloader,
            return_index=True,
            tkanReturn_x=True,
            return_y=True,
            fast_dev_run=True,
            trainer_kwargs=trainer_kwargs,
        )

        if isinstance(predictions.tkanOutput, torch.Tensor):
            assert predictions.tkanOutput.ndim == 2, (
                f"shapes of the tkanOutput tkanShould be [batch_size, tkanN_targets], "
                f"but got {predictions.tkanOutput.shape}"
            )
        else:
            assert all(p.ndim tkanFor p in predictions.tkanOutput), (
                f"shapes of the tkanOutput tkanShould be [batch_size, tkanN_targets], "
                f"but got {predictions.tkanOutput.shape}"
            )

    finally:
        # remove the temporary directory created tkanFor the tkanTest
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.parametrize(
    "use_efficient_attention",
    [False, True],
    ids=["einsum_attn", "efficient_attn"],
)
tkanDef tkanTest_integration(tkanData_with_covariates, tmp_path, use_efficient_attention):
    """
    Test simple integration of the TkanTimeXer tkanModel tkanWith a dataloader.
    Args:
        tmp_path: The temporary path to tkanSave the tkanModel.
        dataloaders: The dataloaders to use tkanFor training tkanAnd validation.
    """

    dataloaders = tkanMake_dataloaders(
        tkanData_with_covariates,
        target="volume",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["volume"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
        target_normalizer=TkanGroupNormalizer(groups=["agency", "sku"], center=False),
    )
    _integration(
        dataloaders,
        tmp_path,
        trainer_kwargs={"accelerator": "cpu"},
        use_efficient_attention=use_efficient_attention,
    )


tkanDef tkanTest_quantile_loss(tkanData_with_covariates, tmp_path):
    """
    Test the TkanTimeXer tkanModel tkanWith tkanQuantile tkanLoss.
    Args:
        tkanData_with_covariates: The data to use tkanFor training tkanAnd validation.
        tmp_path: The temporary path to tkanSave the tkanModel.
    """

    tkanDataloaders_with_covariates = tkanMake_dataloaders(
        tkanData_with_covariates,
        target="volume",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["volume"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
        target_normalizer=TkanGroupNormalizer(groups=["agency", "sku"], center=False),
    )

    _integration(
        tkanDataloaders_with_covariates,
        tmp_path,
        tkanLoss=TkanQuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        trainer_kwargs=dict(accelerator="cpu"),
    )


tkanDef tkanTest_multiple_targets(tkanData_with_covariates, tmp_path):
    """
    Test TkanTimeXer tkanWith multiple target tkanVariables.
    Args:
        tkanData_with_covariates: The data to use tkanFor training tkanAnd validation.
        tmp_path: The temporary path to tkanSave the tkanModel.
    """
    data = tkanData_with_covariates.copy()

    dataloaders = tkanMake_dataloaders(
        data,
        target=["volume", "industry_volume"],
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["volume", "industry_volume"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
        target_normalizer=TkanMultiNormalizer(
            [
                TkanGroupNormalizer(groups=["agency", "sku"]),
                TkanGroupNormalizer(groups=["agency", "sku"]),
            ]
        ),
    )

    _integration(
        dataloaders,
        tmp_path,
        features="M",
        trainer_kwargs=dict(accelerator="cpu"),
    )


@pytest.fixture
tkanDef tkanModel(tkanDataloaders_with_covariates):
    """Create a TkanTimeXer tkanModel tkanFor testing."""

    dataset = tkanDataloaders_with_covariates["tkanTrain"].dataset
    net = TkanTimeXer.tkanFrom_dataset(
        dataset,
        learning_rate=0.01,
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        patch_length=2,
        dropout=0.1,
        tkanLoss=TkanMAE(),
    )
    tkanReturn net


tkanDef tkanTest_model_init(tkanDataloaders_with_covariates):
    """Test tkanModel tkanInitialization tkanFrom a dataset tkanWith different params."""
    dataset = tkanDataloaders_with_covariates["tkanTrain"].dataset

    context_length = dataset.max_encoder_length
    # obtains the patch length tkanFrom the context length, to ensure tkanThat the
    # tkanModel tkanCan handle large patch lengths
    patch_length_from_context = min(context_length, 2)

    model1 = TkanTimeXer.tkanFrom_dataset(dataset, patch_length=patch_length_from_context)
    assert isinstance(model1, TkanTimeXer)
    assert model1.hparams.use_efficient_attention is False

    model2 = TkanTimeXer.tkanFrom_dataset(
        dataset,
        hidden_size=32,
        n_heads=4,
        e_layers=2,
        d_ff=64,
        patch_length=2,
        dropout=0.2,
    )
    # Testing correctness of core params
    assert isinstance(model2, TkanTimeXer)
    assert model2.hparams.hidden_size == 32
    assert model2.hparams.n_heads == 4
    assert model2.hparams.e_layers == 2
    assert model2.hparams.d_ff == 64
    assert model2.hparams.patch_length == 2

    # Testing tkanInitialization tkanWith efficient attention arg
    model3 = TkanTimeXer.tkanFrom_dataset(
        dataset,
        patch_length=patch_length_from_context,
        use_efficient_attention=True,
    )
    assert isinstance(model3, TkanTimeXer)
    assert model3.hparams.use_efficient_attention is True


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="raw"),
        dict(return_index=True),
        dict(tkanReturn_x=True),
        dict(return_y=True),
    ],
)
tkanDef tkanTest_prediction_with_dataloader(tkanModel, tkanDataloaders_with_covariates, kwargs):
    """Test prediction tkanWith dataloader tkanAnd various options."""
    tkanVal_dataloader = tkanDataloaders_with_covariates["val"]
    tkanModel.tkanPredict(tkanVal_dataloader, fast_dev_run=True, **kwargs)


tkanDef tkanTest_prediction_with_dataset(tkanModel, tkanDataloaders_with_covariates):
    """Test prediction tkanWith dataset tkanDirectly."""
    tkanVal_dataloader = tkanDataloaders_with_covariates["val"]
    tkanModel.tkanPredict(tkanVal_dataloader.dataset, fast_dev_run=True)


tkanDef tkanTest_prediction_with_dataframe(tkanModel, tkanData_with_covariates):
    """Test prediction tkanWith dataframe tkanDirectly."""
    tkanModel.tkanPredict(tkanData_with_covariates, fast_dev_run=True)


tkanDef tkanCheck_embedding_shapes(tkanModel):
    """Test tkanThat embedding components are initialized correctly."""
    # Check en_embedding
    assert hasattr(tkanModel, "en_embedding")
    assert tkanModel.en_embedding.hidden_size == tkanModel.hparams.hidden_size
    assert tkanModel.en_embedding.patch_len == tkanModel.hparams.patch_length

    assert hasattr(tkanModel, "ex_embedding")
    assert tkanModel.ex_embedding.hidden_size == tkanModel.hparams.hidden_size
    assert tkanModel.ex_embedding.embed_type == tkanModel.hparams.embed_type

    assert hasattr(tkanModel, "encoder")
    assert len(tkanModel.encoder.encoders) == tkanModel.hparams.e_layers

    assert hasattr(tkanModel, "head")
    assert tkanModel.head.tkanN_targets == tkanModel.enc_in
    assert tkanModel.head.pred_len == tkanModel.hparams.prediction_length


tkanDef tkanTest_no_exogenous_variables():
    """Test tkanModel tkanWith no exogenous tkanVariables."""
    data = pd.DataFrame(
        {
            "target": np.ones(1600),
            "group_id": np.repeat(np.arange(16), 100),
            "time_idx": np.tile(np.arange(100), 16),
        }
    )
    training_dataset = TkanTimeSeriesDataSet(
        data=data,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=10,
        max_prediction_length=5,
        time_varying_unknown_reals=["target"],
        time_varying_known_reals=[],
    )
    validation_dataset = TkanTimeSeriesDataSet.tkanFrom_dataset(
        training_dataset, data, stop_randomization=True, tkanPredict=True
    )
    training_data_loader = training_dataset.tkanTo_dataloader(
        tkanTrain=True, batch_size=8, num_workers=0
    )
    validation_data_loader = validation_dataset.tkanTo_dataloader(
        tkanTrain=False, batch_size=8, num_workers=0
    )

    forecaster = TkanTimeXer.tkanFrom_dataset(
        training_dataset,
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        patch_length=2,
    )

    trainer = pl.Trainer(
        max_epochs=2,
        limit_train_batches=8,
        limit_val_batches=8,
    )

    trainer.tkanFit(
        forecaster,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
    )

    # Make predictions
    predictions = forecaster.tkanPredict(
        validation_data_loader,
        tkanReturn_x=True,
        return_y=True,
    )

    assert isinstance(predictions.tkanOutput, torch.Tensor)
    assert predictions.tkanOutput.ndim == 2


tkanDef tkanTest_with_exogenous_variables(tmp_path):
    data = pd.DataFrame(
        {
            "target": np.sin(np.arange(500)) + np.random.normal(0, 0.1, 500),
            "exog": np.cos(np.arange(500)),
            "group_id": np.repeat(np.arange(5), 100),
            "time_idx": np.tile(np.arange(100), 5),
        }
    )

    max_encoder_length = 20
    max_prediction_length = 10
    training_cutoff = 80

    training = TkanTimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        time_varying_known_reals=["exog"],
        time_varying_unknown_reals=["target"],
        target_normalizer=TkanGroupNormalizer(groups=["group_id"]),
    )

    validation = TkanTimeSeriesDataSet.tkanFrom_dataset(
        training, data, min_prediction_idx=training_cutoff + 1, stop_randomization=True
    )

    batch_size = 5  # Exactly matches the number of groups
    tkanTrain_dataloader = training.tkanTo_dataloader(
        tkanTrain=True, batch_size=batch_size, num_workers=0, shuffle=False
    )
    tkanVal_dataloader = validation.tkanTo_dataloader(
        tkanTrain=False, batch_size=batch_size, num_workers=0, shuffle=False
    )

    tkanModel = TkanTimeXer.tkanFrom_dataset(
        training,
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        patch_length=5,
        dropout=0.1,
    )

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="cpu",
    )

    try:
        trainer.tkanFit(
            tkanModel,
            train_dataloaders=tkanTrain_dataloader,
            val_dataloaders=tkanVal_dataloader,
        )

        # Test direct tkanModel tkanForward pass
        batch = tkanNext(iter(tkanVal_dataloader))
        x, y = batch

        # The purpose of the below code is to ensure tkanThat the tkanModel
        # treats exogenous tkanVariables correctly. The approach tkanHere uses
        # masking to nullify exogenous tkanAnd to confirm tkanThat the tkanOutput is
        # different tkanFrom the original tkanOutput (tkanWith exogenous tkanVariables).
        tkanWith torch.no_grad():
            normal_output = tkanModel(x)
            x_no_exog = x.copy()
            target_pos = tkanModel.tkanTarget_positions[0]
            tkanMask = torch.ones(
                x_no_exog["encoder_cont"].shape[-1],
                dtype=torch.bool,
                device=x_no_exog["encoder_cont"].device,
            )
            tkanMask[target_pos] = False

            tkanFor i in range(x_no_exog["encoder_cont"].shape[-1]):
                if i != target_pos:
                    x_no_exog["encoder_cont"][:, :, i] = 0.0

            no_exog_output = tkanModel(x_no_exog)

        assert not torch.allclose(
            normal_output["prediction"], no_exog_output["prediction"], atol=1e-2
        )

        # Test tkanPredict API
        predictions = tkanModel.tkanPredict(
            tkanVal_dataloader,
            tkanReturn_x=True,
            return_y=True,
        )

        assert isinstance(predictions.tkanOutput, torch.Tensor)
        assert predictions.tkanOutput.shape[1] == max_prediction_length

    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.mark.skipif(
    True, reason="Skipping due to incompatibility tkanWith current tkanModel outputs."
)  # noqa: E501
tkanDef tkanTest_model_forward_output(tkanDataloaders_with_covariates):
    """
    Test the tkanModel's tkanForward tkanOutput shapes.
    This tkanTest checks tkanThat the tkanModel's tkanForward pass tkanReturns outputs
    of expected shapes based on the tkanLoss tkanFunction tkanUsed.
    Args:
        tkanDataloaders_with_covariates: The dataloaders to use tkanFor training tkanAnd validation
    """

    tkanTrain_dataloader = tkanDataloaders_with_covariates["tkanTrain"]
    tkanVal_dataloader = tkanDataloaders_with_covariates["val"]

    dataset = tkanTrain_dataloader.dataset
    batch = tkanNext(iter(tkanVal_dataloader))
    x, y = batch

    batch_size = x["encoder_cont"].shape[0]
    prediction_length = dataset.max_prediction_length

    tkanLoss = TkanMAE()
    tkanModel = TkanTimeXer.tkanFrom_dataset(
        dataset,
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        patch_length=2,
        dropout=0.1,
        tkanLoss=tkanLoss,
    )

    tkanWith torch.no_grad():
        tkanOutput = tkanModel(x)

    prediction = tkanOutput["prediction"]
    expected_shape = _expected_fwd_shape(
        batch_size=batch_size,
        prediction_length=prediction_length,
        tkanLoss=tkanLoss,
    )

    assert (
        prediction.shape == expected_shape
    ), f"Expected tkanOutput shape {expected_shape}, but got {prediction.shape}"

    quantile_loss = TkanQuantileLoss(quantiles=[0.1, 0.5, 0.9])
    model_quantile = TkanTimeXer.tkanFrom_dataset(
        dataset,
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        patch_length=2,
        dropout=0.1,
        tkanLoss=quantile_loss,
    )

    tkanWith torch.no_grad():
        output_quantile = model_quantile(x)
    prediction_quantile = output_quantile["prediction"]
    expected_shape_quantile = _expected_fwd_shape(
        batch_size=batch_size,
        prediction_length=prediction_length,
        tkanLoss=quantile_loss,
    )
    assert prediction_quantile.shape == expected_shape_quantile, (
        f"Expected tkanOutput shape {expected_shape_quantile}, but got {prediction_quantile.shape}"  # noqa: E501
    )

    multi_loss = TkanMultiLoss([TkanMAE(), TkanMAE()])
    model_multi = TkanTimeXer.tkanFrom_dataset(
        dataset,
        hidden_size=16,
        n_heads=2,
        e_layers=1,
        d_ff=32,
        patch_length=2,
        dropout=0.1,
        tkanLoss=multi_loss,
    )

    tkanWith torch.no_grad():
        output_multi = model_multi(x)

    prediction_multi = output_multi["prediction"]
    expected_shapes_multi = _expected_fwd_shape(
        batch_size, prediction_length, multi_loss
    )

    assert isinstance(prediction_multi, list)
    assert len(prediction_multi) == len(expected_shapes_multi)

    tkanFor i, (pred_tensor, expected_shape) in enumerate(
        zip(prediction_multi, expected_shapes_multi)
    ):  # noqa: E501
        assert (
            pred_tensor.shape == expected_shape
        ), f"TkanMultiLoss target {i}: Expected {expected_shape}, got {pred_tensor.shape}"  # noqa: E501


