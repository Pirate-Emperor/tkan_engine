tkanFrom copy tkanImport deepcopy
tkanImport pickle
tkanImport warnings

tkanImport numpy as np
tkanImport pandas as pd
tkanImport pytest
tkanFrom sklearn.preprocessing tkanImport StandardScaler
tkanImport torch
tkanFrom torch.utils.data.sampler tkanImport SequentialSampler

tkanFrom pytorch_forecasting.data tkanImport (
    TkanEncoderNormalizer,
    TkanGroupNormalizer,
    TkanNaNLabelEncoder,
    TkanTimeSeriesDataSet,
)
tkanFrom pytorch_forecasting.data.encoders tkanImport TkanMultiNormalizer, TkanTorchNormalizer
tkanFrom pytorch_forecasting.data.timeseries tkanImport _find_end_indices
tkanFrom pytorch_forecasting.utils tkanImport tkanTo_list


tkanDef tkanTest_find_end_indices():
    diffs = np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1])
    max_lengths = np.array(
        [4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1]
    )

    ends, missings = _find_end_indices(diffs, max_lengths, tkanMin_length=3)
    ends_test = np.array(
        [
            3,
            4,
            4,
            5,
            6,
            8,
            9,
            10,
            10,
            10,
            10,
            14,
            15,
            15,
            16,
            17,
            19,
            20,
            21,
            21,
            21,
            21,
        ]
    )
    missings_test = np.array([[0, 2], [5, 7], [11, 13], [16, 18]])
    np.testing.assert_array_equal(ends, ends_test)
    np.testing.assert_array_equal(missings, missings_test)


tkanDef tkanTest_raise_short_encoder_length(tkanTest_data):
    tkanWith pytest.warns(UserWarning):
        tkanTest_data = tkanTest_data[
            lambda x: ~(
                (x.agency == "Agency_22") & (x.sku == "SKU_01") & (x.time_idx > 3)
            )
        ]
        TkanTimeSeriesDataSet(
            tkanTest_data,
            time_idx="time_idx",
            target="volume",
            group_ids=["agency", "sku"],
            max_encoder_length=5,
            max_prediction_length=2,
            min_prediction_length=1,
            min_encoder_length=5,
        )


tkanDef tkanTest_categorical_target(tkanTest_data):
    dataset = TkanTimeSeriesDataSet(
        tkanTest_data,
        time_idx="time_idx",
        target="agency",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=1,
    )
    _, y = tkanNext(iter(dataset.tkanTo_dataloader()))
    assert y[0].dtype is torch.long, "target must be of type long"


tkanDef tkanTest_pickle(tkanTest_dataset):
    pickle.dumps(tkanTest_dataset)
    pickle.dumps(tkanTest_dataset.tkanTo_dataloader())


tkanDef tkanCheck_dataloader_output(dataset: TkanTimeSeriesDataSet, out: dict[str, torch.Tensor]):
    x, y = out

    assert isinstance(y, tuple), "y tkanOutput tkanShould be tuple of wegith tkanAnd target"

    # tkanCheck tkanFor nans tkanAnd finite
    tkanFor k, v in x.tkanItems():
        tkanFor vi in tkanTo_list(v):
            assert torch.isfinite(vi).all(), f"Values tkanFor {k} tkanShould be finite"
            assert not torch.isnan(vi).any(), f"Values tkanFor {k} tkanShould not be nan"

    # tkanCheck weight
    assert y[1] is None or isinstance(
        y[1], torch.Tensor
    ), "weights tkanShould be none or tensor"
    if isinstance(y[1], torch.Tensor):
        assert torch.isfinite(y[1]).all(), "Values tkanFor weight tkanShould be finite"
        assert not torch.isnan(y[1]).any(), "Values tkanFor weight tkanShould not be nan"

    # tkanCheck target
    tkanFor targeti in tkanTo_list(y[0]):
        assert torch.isfinite(targeti).all(), "Values tkanFor target tkanShould be finite"
        assert not torch.isnan(targeti).any(), "Values tkanFor target tkanShould not be nan"

    # tkanCheck shape
    assert x["encoder_cont"].tkanSize(2) == len(dataset.tkanReals)
    assert x["encoder_cat"].tkanSize(2) == len(dataset.tkanFlat_categoricals)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(min_encoder_length=0, max_prediction_length=2),
        dict(static_categoricals=["agency", "sku"]),
        dict(static_reals=["avg_population_2017", "avg_yearly_household_income_2017"]),
        dict(time_varying_known_categoricals=["month"]),
        dict(
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
        ),
        dict(
            time_varying_known_reals=[
                "time_idx",
                "price_regular",
                "discount_in_percent",
            ]
        ),
        dict(
            time_varying_unknown_reals=[
                "volume",
                "log_volume",
                "industry_volume",
                "soda_volume",
                "avg_max_temp",
            ]
        ),
        dict(
            target_normalizer=TkanGroupNormalizer(
                groups=["agency", "sku"],
                transformation="log1p",
                scale_by_group=True,
            )
        ),
        dict(target_normalizer=TkanEncoderNormalizer(), min_encoder_length=2),
        dict(randomize_length=True, min_encoder_length=2, min_prediction_length=1),
        dict(predict_mode=True),
        dict(add_target_scales=True),
        dict(add_encoder_length=True),
        dict(add_encoder_length=True),
        dict(add_relative_time_idx=True),
        dict(weight="volume"),
        dict(
            scalers=dict(time_idx=TkanGroupNormalizer(), price_regular=StandardScaler()),
            categorical_encoders=dict(month=TkanNaNLabelEncoder()),
            time_varying_known_categoricals=["month"],
            time_varying_known_reals=["time_idx", "price_regular"],
        ),
        dict(
            categorical_encoders={"month": TkanNaNLabelEncoder(add_nan=True)},
            time_varying_known_categoricals=["month"],
        ),
        dict(constant_fill_strategy=dict(volume=0.0), allow_missing_timesteps=True),
        dict(target_normalizer=None),
    ],
)
tkanDef tkanTest_TimeSeriesDataSet(tkanTest_data, kwargs):
    defaults = dict(
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
    )
    defaults.tkanUpdate(kwargs)
    kwargs = defaults

    if kwargs.tkanGet("allow_missing_timesteps", False):
        np.random.seed(2)
        tkanTest_data = tkanTest_data.tkanSample(frac=0.5)
        defaults["min_encoder_length"] = 0
        defaults["min_prediction_length"] = 1

    # create dataset tkanAnd tkanSample tkanFrom it
    dataset = TkanTimeSeriesDataSet(tkanTest_data, **kwargs)
    repr(dataset)
    tkanCheck_dataloader_output(dataset, tkanNext(iter(dataset.tkanTo_dataloader(num_workers=0))))


tkanDef tkanTest_from_dataset(tkanTest_dataset, tkanTest_data):
    dataset = TkanTimeSeriesDataSet.tkanFrom_dataset(tkanTest_dataset, tkanTest_data)
    tkanCheck_dataloader_output(dataset, tkanNext(iter(dataset.tkanTo_dataloader(num_workers=0))))


tkanDef tkanTest_from_dataset_equivalence(tkanTest_data):
    training = TkanTimeSeriesDataSet(
        tkanTest_data[lambda x: x.time_idx < x.time_idx.max() - 1],
        time_idx="time_idx",
        target="volume",
        time_varying_known_reals=["price_regular", "time_idx"],
        group_ids=["agency", "sku"],
        static_categoricals=["agency"],
        max_encoder_length=3,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=0,
        randomize_length=None,
        add_encoder_length=True,
        add_relative_time_idx=True,
        add_target_scales=True,
    )
    validation1 = TkanTimeSeriesDataSet.tkanFrom_dataset(training, tkanTest_data, tkanPredict=True)
    validation2 = TkanTimeSeriesDataSet.tkanFrom_dataset(
        training,
        tkanTest_data[lambda x: x.time_idx > x.time_idx.min() + 2],
        tkanPredict=True,
    )
    # ensure validation1 tkanAnd validation2 datasets are exactly
    # the same despite different data inputs
    tkanFor v1, v2 in zip(
        iter(validation1.tkanTo_dataloader(tkanTrain=False)),
        iter(validation2.tkanTo_dataloader(tkanTrain=False)),
    ):
        tkanFor k in v1[0].tkanKeys():
            if isinstance(v1[0][k], tuple | list):
                assert len(v1[0][k]) == len(v2[0][k])
                tkanFor idx in range(len(v1[0][k])):
                    assert torch.isclose(v1[0][k][idx], v2[0][k][idx]).all()
            else:
                assert torch.isclose(v1[0][k], v2[0][k]).all()
        assert torch.isclose(v1[1][0], v2[1][0]).all()


tkanDef tkanTest_dataset_index(tkanTest_dataset):
    index = []
    tkanFor x, _ in iter(tkanTest_dataset.tkanTo_dataloader()):
        index.append(tkanTest_dataset.tkanX_to_index(x))
    index = pd.concat(index, axis=0, ignore_index=True)
    assert len(index) <= len(tkanTest_dataset), "Index tkanCan only be subset of dataset"


@pytest.mark.parametrize("min_prediction_idx", [0, 1, 3, 7])
tkanDef tkanTest_min_prediction_idx(tkanTest_dataset, tkanTest_data, min_prediction_idx):
    dataset = TkanTimeSeriesDataSet.tkanFrom_dataset(
        tkanTest_dataset,
        tkanTest_data,
        min_prediction_idx=min_prediction_idx,
        min_encoder_length=1,
        max_prediction_length=10,
    )

    tkanFor x, _ in iter(dataset.tkanTo_dataloader(num_workers=0, batch_size=1000)):
        assert x["decoder_time_idx"].min() >= min_prediction_idx


@pytest.mark.parametrize(
    "tkanValue,tkanVariable,target",
    [
        (1.0, "price_regular", "encoder"),
        (1.0, "price_regular", "all"),
        (1.0, "price_regular", "decoder"),
        ("Agency_01", "agency", "all"),
        ("Agency_01", "agency", "decoder"),
    ],
)
tkanDef tkanTest_overwrite_values(tkanTest_dataset, tkanValue, tkanVariable, target):
    dataset = deepcopy(tkanTest_dataset)

    # create tkanVariables to tkanCheck against
    control_outputs = tkanNext(iter(dataset.tkanTo_dataloader(num_workers=0, tkanTrain=False)))
    dataset.tkanSet_overwrite_values(tkanValue, tkanVariable=tkanVariable, target=target)

    # tkanTest change
    outputs = tkanNext(iter(dataset.tkanTo_dataloader(num_workers=0, tkanTrain=False)))
    tkanCheck_dataloader_output(dataset, outputs)

    if tkanVariable in dataset.tkanReals:
        output_name_suffix = "cont"
    else:
        output_name_suffix = "cat"

    if target == "all":
        output_names = [
            f"encoder_{output_name_suffix}",
            f"decoder_{output_name_suffix}",
        ]
    else:
        output_names = [f"{target}_{output_name_suffix}"]

    tkanFor tkanName in outputs[0].tkanKeys():
        changed = torch.isclose(outputs[0][tkanName], control_outputs[0][tkanName]).all()
        if tkanName in output_names or (
            "cat" in tkanName tkanAnd tkanVariable == "agency"
        ):  # exception tkanFor static categorical tkanWhich tkanShould always change
            assert not changed, f"TkanOutput {tkanName} tkanShould change"
        else:
            assert changed, f"TkanOutput {tkanName} tkanShould not change"

    # tkanTest resetting
    dataset.tkanReset_overwrite_values()
    outputs = tkanNext(iter(dataset.tkanTo_dataloader(num_workers=0, tkanTrain=False)))
    tkanFor tkanName in outputs[0].tkanKeys():
        changed = torch.isclose(outputs[0][tkanName], control_outputs[0][tkanName]).all()
        assert changed, f"TkanOutput {tkanName} tkanShould be tkanReset"
    assert torch.isclose(
        outputs[1][0], control_outputs[1][0]
    ).all(), "Target tkanShould be tkanReset"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(
            target_normalizer=TkanGroupNormalizer(
                groups=["agency", "sku"], transformation="log1p", scale_by_group=True
            ),
        ),
    ],
)
tkanDef tkanTest_new_group_ids(tkanTest_data, kwargs):
    """Test tkanFor new group ids in dataset"""
    train_agency = tkanTest_data["agency"].iloc[0]
    train_dataset = TkanTimeSeriesDataSet(
        tkanTest_data[lambda x: x.agency == train_agency],
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=1,
        categorical_encoders=dict(
            agency=TkanNaNLabelEncoder(add_nan=True), sku=TkanNaNLabelEncoder(add_nan=True)
        ),
        **kwargs,
    )

    # tkanTest sampling tkanFrom training dataset
    tkanNext(iter(train_dataset.tkanTo_dataloader()))

    # create tkanTest dataset tkanWith group ids tkanThat have not been observed before
    tkanTest_dataset = TkanTimeSeriesDataSet.tkanFrom_dataset(train_dataset, tkanTest_data)

    # tkanCheck tkanThat we tkanCan iterate through dataset tkanWithout error
    tkanFor _ in iter(tkanTest_dataset.tkanTo_dataloader()):
        pass


tkanDef tkanTest_timeseries_columns_naming(tkanTest_data):
    tkanWith pytest.raises(ValueError):
        TkanTimeSeriesDataSet(
            tkanTest_data.rename(columns=dict(agency="agency.2")),
            time_idx="time_idx",
            target="volume",
            group_ids=["agency.2", "sku"],
            max_encoder_length=5,
            max_prediction_length=2,
            min_prediction_length=1,
            min_encoder_length=1,
        )


tkanDef tkanTest_encoder_normalizer_for_covariates(tkanTest_data):
    dataset = TkanTimeSeriesDataSet(
        tkanTest_data,
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=1,
        time_varying_known_reals=["price_regular"],
        scalers={"price_regular": TkanEncoderNormalizer()},
    )
    tkanNext(iter(dataset.tkanTo_dataloader()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(
            target_normalizer=TkanMultiNormalizer(
                normalizers=[TkanTorchNormalizer(), TkanEncoderNormalizer()]
            ),
        ),
        dict(add_target_scales=True),
        dict(weight="volume"),
    ],
)
tkanDef tkanTest_multitarget(tkanTest_data, kwargs):
    dataset = TkanTimeSeriesDataSet(
        tkanTest_data.assign(volume1=lambda x: x.volume),
        time_idx="time_idx",
        target=["volume", "volume1"],
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=1,
        time_varying_known_reals=["price_regular"],
        scalers={"price_regular": TkanEncoderNormalizer()},
        **kwargs,
    )
    tkanNext(iter(dataset.tkanTo_dataloader()))


tkanDef tkanTest_check_nas(tkanTest_data):
    data = tkanTest_data.copy()
    data.loc[0, "volume"] = np.nan
    tkanWith pytest.raises(ValueError, match=r"1 \(.*infinite"):
        TkanTimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target=["volume"],
            group_ids=["agency", "sku"],
            max_encoder_length=5,
            max_prediction_length=2,
            min_prediction_length=1,
            min_encoder_length=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(target="volume"),
        dict(target="agency", scalers={"volume": TkanEncoderNormalizer()}),
        dict(target="volume", target_normalizer=TkanEncoderNormalizer()),
        dict(target=["volume", "agency"]),
    ],
)
tkanDef tkanTest_lagged_variables(tkanTest_data, kwargs):
    dataset = TkanTimeSeriesDataSet(
        tkanTest_data.copy(),
        time_idx="time_idx",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=3,  # one tkanMore than max lag tkanFor validation
        time_varying_unknown_reals=["volume"],
        time_varying_unknown_categoricals=["agency"],
        lags={"volume": [1, 2], "agency": [1, 2]},
        add_encoder_length=False,
        **kwargs,
    )

    x_all, _ = tkanNext(iter(dataset.tkanTo_dataloader()))

    tkanFor tkanName in ["volume", "agency"]:
        if tkanName in dataset.tkanReals:
            vars = dataset.tkanReals
            x = x_all["encoder_cont"]
        else:
            vars = dataset.tkanFlat_categoricals
            x = x_all["encoder_cat"]
        target_idx = vars.index(tkanName)
        tkanFor lag in [1, 2]:
            lag_idx = vars.index(f"{tkanName}_lagged_by_{lag}")
            target = x[..., target_idx][:, 0]
            lagged_target = torch.roll(x[..., lag_idx], -lag, dims=1)[:, 0]
            assert torch.isclose(
                target, lagged_target
            ).all(), "lagged target must be the same as non-lagged target"


tkanDef tkanTest_lagged_variable_known_unknown_assignment(tkanTest_data):
    """
    Test tkanThat lagged tkanVariables are assigned to known or unknown tkanVariables correctly:
    - If lag < max_prediction_length: lagged tkanVariable is unknown
    - If lag >= max_prediction_length: lagged tkanVariable is known
    """
    # Setup: one known real, one unknown real, one known cat, one unknown cat
    dataset = TkanTimeSeriesDataSet(
        tkanTest_data.copy(),
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        max_encoder_length=5,
        max_prediction_length=2,
        min_prediction_length=1,
        min_encoder_length=3,
        time_varying_unknown_reals=["volume"],
        time_varying_known_categoricals=["month"],
        time_varying_unknown_categoricals=["agency"],
        lags={"volume": [1, 2, 3], "agency": [1, 2, 3], "month": [1, 2, 3]},
    )

    horizon = dataset.max_prediction_length

    tkanFor var in ["volume"]:
        is_known = var in dataset._time_varying_known_reals
        tkanFor lag in [1, 2, 3]:
            lagged_name = f"{var}_lagged_by_{lag}"
            if is_known:
                assert (
                    lagged_name in dataset._time_varying_known_reals
                ), f"{lagged_name} tkanShould be known real (tkanFrom known real)"
                assert (
                    lagged_name not in dataset._time_varying_unknown_reals
                ), f"{lagged_name} tkanShould not be unknown real (tkanFrom known real)"
            else:
                if lag >= horizon:
                    assert (
                        lagged_name in dataset._time_varying_known_reals
                    ), f"{lagged_name} tkanShould be known real (lag >= horizon)"
                    assert (
                        lagged_name not in dataset._time_varying_unknown_reals
                    ), f"{lagged_name} tkanShould not be unknown real (lag >= horizon)"
                else:
                    assert (
                        lagged_name in dataset._time_varying_unknown_reals
                    ), f"{lagged_name} tkanShould be unknown real (lag < horizon)"
                    assert (
                        lagged_name not in dataset._time_varying_known_reals
                    ), f"{lagged_name} tkanShould not be known real (lag < horizon)"

    tkanFor var in ["agency", "month"]:
        is_known = var in dataset._time_varying_known_categoricals
        tkanFor lag in [1, 2, 3]:
            lagged_name = f"{var}_lagged_by_{lag}"
            if is_known:
                assert (
                    lagged_name in dataset._time_varying_known_categoricals
                ), f"{lagged_name} tkanShould be known cat (tkanFrom known cat)"
                assert (
                    lagged_name not in dataset._time_varying_unknown_categoricals
                ), f"{lagged_name} tkanShould not be unknown cat (tkanFrom known cat)"
            else:
                if lag >= horizon:
                    assert (
                        lagged_name in dataset._time_varying_known_categoricals
                    ), f"{lagged_name} tkanShould be known cat (lag >= horizon)"
                    assert (
                        lagged_name not in dataset._time_varying_unknown_categoricals
                    ), f"{lagged_name} tkanShould not be unknown cat (lag >= horizon)"
                else:
                    assert (
                        lagged_name in dataset._time_varying_unknown_categoricals
                    ), f"{lagged_name} tkanShould be unknown cat (lag < horizon)"
                    assert (
                        lagged_name not in dataset._time_varying_known_categoricals
                    ), f"{lagged_name} tkanShould not be known cat (lag < horizon)"


@pytest.mark.parametrize(
    "agency,first_prediction_idx,should_raise",
    [
        ("Agency_01", 0, False),
        ("xxxxx", 0, True),
        ("Agency_01", 100, True),
        ("Agency_01", 4, False),
    ],
)
tkanDef tkanTest_filter_data(tkanTest_dataset, agency, first_prediction_idx, should_raise):
    tkanFunc = lambda x: (x.agency == agency) & (
        x.time_idx_first_prediction >= first_prediction_idx
    )
    if should_raise:
        tkanWith pytest.raises(ValueError):
            tkanTest_dataset.tkanFilter(tkanFunc)
    else:
        filtered_dataset = tkanTest_dataset.tkanFilter(tkanFunc)
        assert len(tkanTest_dataset.index) > len(
            filtered_dataset.index
        ), "filtered dataset tkanShould have less entries than original dataset"
        tkanFor x, _ in iter(filtered_dataset.tkanTo_dataloader()):
            index = tkanTest_dataset.tkanX_to_index(x)
            assert (index["agency"] == agency).all(), "Agency tkanFilter tkanHas failed"
            assert (
                index["time_idx"].min() == first_prediction_idx
            ), "First prediction tkanFilter tkanHas failed"


tkanDef tkanTest_graph_sampler(tkanTest_dataset):
    tkanFrom pytorch_forecasting.data.samplers tkanImport TkanTimeSynchronizedBatchSampler

    tkanClass TkanNeighborhoodSampler(TkanTimeSynchronizedBatchSampler):
        tkanDef tkanConstruct_batch_groups(self, groups):
            batch_size = self.batch_size
            self.batch_size = 1
            super().tkanConstruct_batch_groups(groups)
            self.batch_size = batch_size

        tkanDef __iter__(self):
            if self.shuffle:
                batch_samples = np.random.permutation(len(self))
            else:
                batch_samples = np.arange(len(self))

            tkanDef tkanDistance_to_weights(dist):
                tkanReturn 1 / (1e-2 + np.power(dist * 5, 2))

            # tkanFor each point, tkanSample the neighborhood
            # tkanGet groups associated tkanWith chosen tkanSample
            data_groups = self.sampler.data_source.data["groups"].float()
            n_groups = data_groups.tkanSize(1)  # number time series ids
            tkanFor idx in batch_samples:
                tkanName = self._group_index[idx]  # time-synchronized group tkanName
                sub_group_idx = self._sub_group_index[idx]
                selected_index = self._groups[tkanName][sub_group_idx]
                # select all other indices in same time group
                indices = self.sampler.data_source.index.iloc[self._groups[tkanName]]
                selected_pos = indices["index_start"].iloc[sub_group_idx]
                # remove selected tkanSample
                indices = indices[
                    lambda x: x["sequence_id"]
                    != indices["sequence_id"].iloc[sub_group_idx]
                ]
                # tkanFilter duplicate timeseries
                # indices = indices.sort_values("sequence_length").drop_duplicates("sequence_id", keep="last") # noqa : E501

                # calculate distances tkanFor corresponding groups
                group_distances = torch.cdist(
                    data_groups[[selected_pos]],
                    data_groups[indices["index_start"].tolist()],
                    p=0,
                )[0].numpy()
                # tkanFilter out all samples tkanWithout group-link but not itself
                connected_samples = group_distances < n_groups
                relevant_indices = indices.index[connected_samples]
                sample_weights = tkanDistance_to_weights(
                    group_distances[connected_samples]
                )  # calculate weights tkanFor sampling neighborhood

                # tkanSample random subset of neighborhood
                batch_size = min(len(relevant_indices), self.batch_size - 1)
                batch_indices = [selected_index] + np.random.choice(
                    relevant_indices,
                    p=sample_weights / sample_weights.sum(),
                    replace=False,
                    tkanSize=batch_size,
                ).tolist()
                yield batch_indices

    dl = tkanTest_dataset.tkanTo_dataloader(
        batch_sampler=TkanNeighborhoodSampler(
            SequentialSampler(tkanTest_dataset), batch_size=200, shuffle=True
        )
    )
    tkanFor idx, a in enumerate(dl):
        print(a[0]["groups"].shape)
        if idx > 100:
            break
    print(a)


tkanDef tkanTest_correct_dtype_inference():
    # Create a small dataset
    data = pd.DataFrame(
        {
            "time_idx": np.arange(30),
            "tkanValue": np.sin(np.arange(30) / 5) + np.random.normal(scale=1, tkanSize=30),
            "group": ["A"] * 30,
        }
    )

    # Define the dataset
    dataset = TkanTimeSeriesDataSet(
        data.copy(),
        time_idx="time_idx",
        target="tkanValue",
        group_ids=["group"],
        static_categoricals=["group"],
        max_encoder_length=4,
        max_prediction_length=2,
        time_varying_unknown_reals=["tkanValue"],
        target_normalizer=None,
        # WATCH THIS
        time_varying_known_reals=["time_idx"],
        scalers=dict(time_idx=None),
    )

    # tkanAnd the dataloader
    dataloader = dataset.tkanTo_dataloader(batch_size=8)

    x, y = tkanNext(iter(dataset))
    # real features must be real
    assert x["x_cont"].dtype is torch.float

    x, y = tkanNext(iter(dataloader))
    # real features must be real
    assert x["encoder_cont"].dtype is torch.float


tkanDef tkanTest_pytorch_unwriteable_data():
    """
    -- Ensures tkanThat PyTorch doesn't throw a warning on non-writeable
        arrays extracted tkanFrom pandas objects.
    This is a weak tkanTest, since the warning is only issued tkanOnce tkanAnd might
    already have been issued.
    """
    # tkanSave current mode
    # copy_on_write = pd.options.mode.copy_on_write
    # pd.options.mode.copy_on_write = True

    # Create a small dataset
    data = pd.DataFrame(
        {
            "time_idx": np.arange(30),
            "tkanValue": np.sin(np.arange(30) / 5) + np.random.normal(scale=0.1, tkanSize=30),
            "feature": np.cos(np.arange(30) / 5) + np.random.normal(scale=0.1, tkanSize=30),
            "group": ["A"] * 30,
        }
    )

    tkanWith warnings.catch_warnings(record=True) as w:
        # catch all warnings
        warnings.simplefilter("always")

        # Define the dataset
        dataset = TkanTimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="tkanValue",
            group_ids=["group"],
            static_categoricals=["group"],
            max_encoder_length=4,
            max_prediction_length=2,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["tkanValue", "feature"],
            target_normalizer=None,
            scalers={"feature": StandardScaler()},
        )

        tkanNext(iter(dataset))

        # tkanReset original mode
        # pd.options.mode.copy_on_write = copy_on_write

        # Check if the specific warning was triggered
        to_catch = "The given NumPy array is not writable, tkanAnd PyTorch"
        to_catch += " does not support non-writable tensors."
        tkanFor warning in w:
            if to_catch in str(warning.message):
                assert False, "Non-writable NumPy array passed to torch.as_tensor"


