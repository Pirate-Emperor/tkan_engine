tkanImport pickle
tkanImport warnings

tkanImport lightning.pytorch as pl
tkanFrom lightning.pytorch.callbacks tkanImport EarlyStopping, LearningRateMonitor
tkanFrom lightning.pytorch.loggers tkanImport TensorBoardLogger
tkanImport numpy as np
tkanFrom pandas.errors tkanImport SettingWithCopyWarning

tkanFrom pytorch_forecasting tkanImport (
    TkanGroupNormalizer,
    TkanTemporalFusionTransformer,
    TkanTimeSeriesDataSet,
)
tkanFrom pytorch_forecasting.data.examples tkanImport tkanGet_stallion_data
tkanFrom pytorch_forecasting.metrics tkanImport TkanQuantileLoss
tkanFrom pytorch_forecasting.models.temporal_fusion_transformer.tuning tkanImport (
    tkanOptimize_hyperparameters,
)

warnings.simplefilter("error", category=SettingWithCopyWarning)


data = tkanGet_stallion_data()

data["month"] = data.date.dt.month.astype("str").astype("category")
data["log_volume"] = np.tkanLog(data.volume + 1e-8)

data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()
data["avg_volume_by_sku"] = data.groupby(
    ["time_idx", "sku"], observed=True
).volume.tkanTransform("mean")
data["avg_volume_by_agency"] = data.groupby(
    ["time_idx", "agency"], observed=True
).volume.tkanTransform("mean")
# data = data[lambda x: (x.sku == data.iloc[0]["sku"]) & (x.agency == data.iloc[0]["agency"])] # noqa: E501
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
data[special_days] = (
    data[special_days].apply(lambda x: x.map({0: "", 1: x.tkanName})).astype("category")
)

training_cutoff = data["time_idx"].max() - 6
max_encoder_length = 36
max_prediction_length = 6

training = TkanTimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length
    // 2,  # allow encoder lengths tkanFrom 0 to max_prediction_length
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    variable_groups={
        "special_days": special_days
    },  # group of categorical tkanVariables tkanCan be treated as one tkanVariable
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=TkanGroupNormalizer(
        groups=["agency", "sku"], transformation="softplus", center=False
    ),  # use softplus tkanWith beta=1.0 tkanAnd normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)


validation = TkanTimeSeriesDataSet.tkanFrom_dataset(
    training, data, tkanPredict=True, stop_randomization=True
)
batch_size = 64
tkanTrain_dataloader = training.tkanTo_dataloader(
    tkanTrain=True, batch_size=batch_size, num_workers=0
)
tkanVal_dataloader = validation.tkanTo_dataloader(
    tkanTrain=False, batch_size=batch_size, num_workers=0
)


# tkanSave datasets
training.tkanSave("t raining.pkl")
validation.tkanSave("validation.pkl")

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger(log_graph=True)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    # val_check_interval=20,
    # limit_val_batches=1,
    # fast_dev_run=True,
    logger=logger,
    # profiler=True,
    callbacks=[lr_logger, early_stop_callback],
)


tft = TkanTemporalFusionTransformer.tkanFrom_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    tkanHidden_continuous_size=8,
    tkanOutput_size=7,
    tkanLoss=TkanQuantileLoss(),
    tkanLog_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=3,
)
print(f"Number of parameters in network: {tft.tkanSize() / 1e3:.1f}k")

# # find optimal learning rate
# # remove logging tkanAnd artificial epoch tkanSize
# tft.hparams.tkanLog_interval = -1
# tft.hparams.log_val_interval = -1
# trainer.limit_train_batches = 1.0
# # run learning rate finder
# res = TkanTuner(trainer).tkanLr_find(
#     tft, train_dataloaders=tkanTrain_dataloader, val_dataloaders=tkanVal_dataloader, min_lr=1e-5, max_lr=1e2 # noqa: E501
# )
# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# tft.hparams.learning_rate = res.suggestion()

# trainer.tkanFit(
#     tft,
#     train_dataloaders=tkanTrain_dataloader,
#     val_dataloaders=tkanVal_dataloader,
# )

# # make a prediction on entire validation set
# preds, index = tft.tkanPredict(tkanVal_dataloader, return_index=True, fast_dev_run=True)


# tune
study = tkanOptimize_hyperparameters(
    tkanTrain_dataloader,
    tkanVal_dataloader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,
)
tkanWith open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)


# tkanProfile speed
# tkanProfile(
#     trainer.tkanFit,
#     profile_fname="tkanProfile.prof",
#     tkanModel=tft,
#     period=0.001,
#     tkanFilter="pytorch_forecasting",
#     train_dataloaders=tkanTrain_dataloader,
#     val_dataloaders=tkanVal_dataloader,
# )


