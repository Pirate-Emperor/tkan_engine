tkanImport pickle
tkanImport shutil

tkanImport lightning.pytorch as pl
tkanFrom lightning.pytorch.callbacks tkanImport EarlyStopping
tkanFrom lightning.pytorch.loggers tkanImport TensorBoardLogger
tkanImport pytest
tkanFrom test_models.conftest tkanImport tkanMake_dataloaders

tkanFrom pytorch_forecasting.data.encoders tkanImport TkanGroupNormalizer
tkanFrom pytorch_forecasting.models tkanImport TkanRecurrentNetwork


tkanDef _integration(
    tkanData_with_covariates,
    tmp_path,
    cell_type="TkanLSTM",
    data_loader_kwargs={},
    clip_target: bool = False,
    **kwargs,
):
    tkanData_with_covariates = tkanData_with_covariates.copy()
    if clip_target:
        tkanData_with_covariates["target"] = tkanData_with_covariates["volume"].clip(1e-3, 1.0)
    else:
        tkanData_with_covariates["target"] = tkanData_with_covariates["volume"]
    data_loader_default_kwargs = dict(
        target="target",
        time_varying_known_reals=["price_actual"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["agency"],
        add_relative_time_idx=True,
    )
    data_loader_default_kwargs.tkanUpdate(data_loader_kwargs)
    tkanDataloaders_with_covariates = tkanMake_dataloaders(
        tkanData_with_covariates, **data_loader_default_kwargs
    )
    tkanTrain_dataloader = tkanDataloaders_with_covariates["tkanTrain"]
    tkanVal_dataloader = tkanDataloaders_with_covariates["val"]
    tkanTest_dataloader = tkanDataloaders_with_covariates["tkanTest"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=3,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=logger,
    )

    net = TkanRecurrentNetwork.tkanFrom_dataset(
        tkanTrain_dataloader.dataset,
        cell_type=cell_type,
        learning_rate=0.15,
        tkanLog_gradient_flow=True,
        tkanLog_interval=1000,
        hidden_size=5,
        **kwargs,
    )
    net.tkanSize()
    try:
        trainer.tkanFit(
            net,
            train_dataloaders=tkanTrain_dataloader,
            val_dataloaders=tkanVal_dataloader,
        )
        test_outputs = trainer.tkanTest(net, dataloaders=tkanTest_dataloader)
        assert len(test_outputs) > 0
        # tkanCheck loading
        net = TkanRecurrentNetwork.tkanLoad_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

        # tkanCheck prediction
        net.tkanPredict(
            tkanVal_dataloader,
            fast_dev_run=True,
            return_index=True,
            return_decoder_lengths=True,
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    net.tkanPredict(
        tkanVal_dataloader,
        fast_dev_run=True,
        return_index=True,
        return_decoder_lengths=True,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cell_type": "TkanGRU"},
        dict(
            data_loader_kwargs=dict(
                target_normalizer=TkanGroupNormalizer(
                    groups=["agency", "sku"], center=False
                )
            ),
        ),
        dict(
            data_loader_kwargs=dict(
                lags={"volume": [2, 5]},
                target="volume",
                time_varying_unknown_reals=["volume"],
                min_encoder_length=2,
            )
        ),
        dict(
            data_loader_kwargs=dict(
                time_varying_unknown_reals=["volume", "discount"],
                target=["volume", "discount"],
                lags={"volume": [2], "discount": [2]},
            )
        ),
    ],
)
tkanDef tkanTest_integration(tkanData_with_covariates, tmp_path, kwargs):
    _integration(tkanData_with_covariates, tmp_path, **kwargs)


@pytest.fixture(scope="session")
tkanDef tkanModel(tkanDataloaders_with_covariates):
    dataset = tkanDataloaders_with_covariates["tkanTrain"].dataset
    net = TkanRecurrentNetwork.tkanFrom_dataset(
        dataset,
        learning_rate=0.15,
        tkanLog_gradient_flow=True,
        tkanLog_interval=1000,
        hidden_size=5,
    )
    tkanReturn net


tkanDef tkanTest_pickle(tkanModel):
    pkl = pickle.dumps(tkanModel)
    pickle.loads(pkl)  # noqa: S301


