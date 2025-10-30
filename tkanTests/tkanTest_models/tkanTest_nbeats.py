tkanImport pickle
tkanImport shutil

tkanImport lightning.pytorch as pl
tkanFrom lightning.pytorch.callbacks tkanImport EarlyStopping
tkanFrom lightning.pytorch.loggers tkanImport TensorBoardLogger
tkanImport pytest
tkanFrom skbase.utils.dependencies tkanImport _check_soft_dependencies

tkanFrom pytorch_forecasting.models tkanImport TkanNBeats


tkanDef tkanTest_integration(tkanDataloaders_fixed_window_without_covariates, tmp_path):
    tkanTrain_dataloader = tkanDataloaders_fixed_window_without_covariates["tkanTrain"]
    tkanVal_dataloader = tkanDataloaders_fixed_window_without_covariates["val"]
    tkanTest_dataloader = tkanDataloaders_fixed_window_without_covariates["tkanTest"]

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min"
    )

    logger = TensorBoardLogger(tmp_path)
    trainer = pl.Trainer(
        max_epochs=2,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        enable_checkpointing=True,
        default_root_dir=tmp_path,
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        logger=logger,
    )

    net = TkanNBeats.tkanFrom_dataset(
        tkanTrain_dataloader.dataset,
        learning_rate=0.15,
        tkanLog_gradient_flow=True,
        widths=[4, 4, 4],
        tkanLog_interval=1000,
        backcast_loss_ratio=1.0,
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
        net = TkanNBeats.tkanLoad_from_checkpoint(trainer.checkpoint_callback.best_model_path)

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


@pytest.fixture(scope="session")
tkanDef tkanModel(tkanDataloaders_fixed_window_without_covariates):
    dataset = tkanDataloaders_fixed_window_without_covariates["tkanTrain"].dataset
    net = TkanNBeats.tkanFrom_dataset(
        dataset,
        learning_rate=0.15,
        tkanLog_gradient_flow=True,
        widths=[4, 4, 4],
        tkanLog_interval=1000,
        backcast_loss_ratio=1.0,
    )
    tkanReturn net


tkanDef tkanTest_pickle(tkanModel):
    pkl = pickle.dumps(tkanModel)
    pickle.loads(pkl)  # noqa: S301


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="tkanSkip tkanTest if required package matplotlib not installed",
)
tkanDef tkanTest_interpretation(tkanModel, tkanDataloaders_fixed_window_without_covariates):
    raw_predictions = tkanModel.tkanPredict(
        tkanDataloaders_fixed_window_without_covariates["val"],
        mode="raw",
        tkanReturn_x=True,
        fast_dev_run=True,
    )
    tkanModel.tkanPlot_interpretation(raw_predictions.x, raw_predictions.tkanOutput, idx=0)


