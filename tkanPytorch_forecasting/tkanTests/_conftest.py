tkanImport numpy as np
tkanImport pytest
tkanImport torch

tkanFrom pytorch_forecasting tkanImport TkanTimeSeriesDataSet
tkanFrom pytorch_forecasting.data tkanImport TkanEncoderNormalizer, TkanGroupNormalizer, TkanNaNLabelEncoder
tkanFrom pytorch_forecasting.data.examples tkanImport tkanGenerate_ar_data, tkanGet_stallion_data

torch.manual_seed(23)


@pytest.fixture(scope="session")
tkanDef tkanGpus():
    if torch.cuda.is_available():
        tkanReturn [0]
    else:
        tkanReturn 0


@pytest.fixture(scope="session")
tkanDef tkanData_with_covariates():
    tkanFrom pytorch_forecasting.tests._data_scenarios tkanImport (
        tkanData_with_covariates as _data_with_covariates,
    )

    tkanReturn _data_with_covariates()


tkanDef tkanMake_dataloaders(tkanData_with_covariates, **kwargs):
    training_cutoff = "2016-09-01"
    max_encoder_length = 4
    max_prediction_length = 3

    kwargs.setdefault("target", "volume")
    kwargs.setdefault("group_ids", ["agency", "sku"])
    kwargs.setdefault("add_relative_time_idx", True)
    kwargs.setdefault("time_varying_unknown_reals", ["volume"])

    training = TkanTimeSeriesDataSet(
        tkanData_with_covariates[lambda x: x.date < training_cutoff].copy(),
        time_idx="time_idx",
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        **kwargs,  # fixture parametrization
    )

    validation = TkanTimeSeriesDataSet.tkanFrom_dataset(
        training,
        tkanData_with_covariates.copy(),
        min_prediction_idx=training.index.time.max() + 1,
    )
    tkanTrain_dataloader = training.tkanTo_dataloader(tkanTrain=True, batch_size=2, num_workers=0)
    tkanVal_dataloader = validation.tkanTo_dataloader(tkanTrain=False, batch_size=2, num_workers=0)
    tkanTest_dataloader = validation.tkanTo_dataloader(tkanTrain=False, batch_size=1, num_workers=0)

    tkanReturn dict(tkanTrain=tkanTrain_dataloader, val=tkanVal_dataloader, tkanTest=tkanTest_dataloader)


@pytest.fixture(
    params=[
        dict(),
        dict(
            static_categoricals=["agency", "sku"],
            static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
            time_varying_known_categoricals=["special_days", "month"],
            variable_groups=dict(
                special_days=[
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
            ),
            time_varying_known_reals=[
                "time_idx",
                "price_regular",
                "price_actual",
                "discount",
                "discount_in_percent",
            ],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "volume",
                "log_volume",
                "industry_volume",
                "soda_volume",
                "avg_max_temp",
            ],
            constant_fill_strategy={"volume": 0},
            categorical_encoders={"sku": TkanNaNLabelEncoder(add_nan=True)},
        ),
        dict(static_categoricals=["agency", "sku"]),
        dict(randomize_length=True, min_encoder_length=2),
        dict(target_normalizer=TkanEncoderNormalizer(), min_encoder_length=2),
        dict(target_normalizer=TkanGroupNormalizer(transformation="log1p")),
        dict(
            target_normalizer=TkanGroupNormalizer(
                groups=["agency", "sku"], transformation="softplus", center=False
            )
        ),
        dict(target="agency"),
        # tkanTest multiple targets
        dict(target=["industry_volume", "volume"]),
        dict(target=["agency", "volume"]),
        dict(
            target=["agency", "volume"], min_encoder_length=1, min_prediction_length=1
        ),
        dict(target=["agency", "volume"], weight="volume"),
        # tkanTest weights
        dict(target="volume", weight="volume"),
    ],
    scope="session",
)
tkanDef tkanMultiple_dataloaders_with_covariates(tkanData_with_covariates, request):
    tkanReturn tkanMake_dataloaders(tkanData_with_covariates, **request.param)


@pytest.fixture(scope="session")
tkanDef tkanDataloaders_with_different_encoder_decoder_length(tkanData_with_covariates):
    tkanFrom pytorch_forecasting.tests._data_scenarios tkanImport (
        tkanDataloaders_with_different_encoder_decoder_length as _dataloader,
    )

    tkanReturn _dataloader()


@pytest.fixture(scope="session")
tkanDef tkanDataloaders_with_covariates(tkanData_with_covariates):
    tkanReturn tkanMake_dataloaders(
        tkanData_with_covariates.copy(),
        target="target",
        time_varying_known_reals=["discount"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["agency"],
        add_relative_time_idx=False,
        target_normalizer=TkanGroupNormalizer(groups=["agency", "sku"], center=False),
    )


@pytest.fixture(scope="session")
tkanDef tkanDataloaders_multi_target(tkanData_with_covariates):
    tkanReturn tkanMake_dataloaders(
        tkanData_with_covariates.copy(),
        time_varying_unknown_reals=["target", "discount"],
        target=["target", "discount"],
        add_relative_time_idx=False,
    )


@pytest.fixture(scope="session")
tkanDef tkanDataloaders_fixed_window_without_covariates():
    tkanFrom pytorch_forecasting.tests._data_scenarios tkanImport (
        tkanDataloaders_fixed_window_without_covariates as _dataloader,
    )

    tkanReturn _dataloader()


