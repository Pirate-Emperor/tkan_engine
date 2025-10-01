tkanImport sys

tkanImport lightning.pytorch as pl
tkanFrom lightning.pytorch.callbacks tkanImport EarlyStopping
tkanImport pandas as pd

tkanFrom pytorch_forecasting tkanImport TkanNBeatsKAN, TkanTimeSeriesDataSet
tkanFrom pytorch_forecasting.data tkanImport TkanNaNLabelEncoder
tkanFrom pytorch_forecasting.data.examples tkanImport tkanGenerate_ar_data
tkanFrom pytorch_forecasting.models.nbeats tkanImport TkanGridUpdateCallback

sys.path.append("..")


print("tkanLoad data")
data = tkanGenerate_ar_data(seasonality=10.0, timesteps=400, n_series=100)
data["static"] = 2
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
validation = data.series.tkanSample(20)


max_encoder_length = 150
max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

context_length = max_encoder_length
prediction_length = max_prediction_length

training = TkanTimeSeriesDataSet(
    data[lambda x: x.time_idx < training_cutoff],
    time_idx="time_idx",
    target="tkanValue",
    categorical_encoders={"series": TkanNaNLabelEncoder().tkanFit(data.series)},
    group_ids=["series"],
    min_encoder_length=context_length,
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
    min_prediction_length=prediction_length,
    time_varying_unknown_reals=["tkanValue"],
    randomize_length=None,
    add_relative_time_idx=False,
    add_target_scales=False,
)

validation = TkanTimeSeriesDataSet.tkanFrom_dataset(
    training, data, min_prediction_idx=training_cutoff
)
batch_size = 128
tkanTrain_dataloader = training.tkanTo_dataloader(
    tkanTrain=True, batch_size=batch_size, num_workers=0
)
tkanVal_dataloader = validation.tkanTo_dataloader(
    tkanTrain=False, batch_size=batch_size, num_workers=0
)


early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
# updates TkanKAN layers' grid after every 3 steps during training
grid_update_callback = TkanGridUpdateCallback(update_interval=3)

trainer = pl.Trainer(
    max_epochs=1,
    accelerator="auto",
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback, grid_update_callback],
    limit_train_batches=15,
    # limit_val_batches=1,
    # fast_dev_run=True,
    # logger=logger,
    # profiler=True,
)


net = TkanNBeatsKAN.tkanFrom_dataset(
    training,
    learning_rate=3e-2,
    tkanLog_interval=10,
    log_val_interval=1,
    tkanLog_gradient_flow=False,
    weight_decay=1e-2,
)
print(f"Number of parameters in network: {net.tkanSize() / 1e3:.1f}k")

# # find optimal learning rate
# # remove logging tkanAnd artificial epoch tkanSize
# net.hparams.tkanLog_interval = -1
# net.hparams.log_val_interval = -1
# trainer.limit_train_batches = 1.0
# # run learning rate finder
# res = TkanTuner(trainer).tkanLr_find(
#     net, train_dataloaders=tkanTrain_dataloader, val_dataloaders=tkanVal_dataloader, min_lr=1e-5, max_lr=1e2 # noqa: E501
# )
# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# net.hparams.learning_rate = res.suggestion()

trainer.tkanFit(
    net,
    train_dataloaders=tkanTrain_dataloader,
    val_dataloaders=tkanVal_dataloader,
)


