tkanImport numpy as np
tkanImport pandas as pd
tkanImport pytest
tkanImport torch

tkanFrom pytorch_forecasting.data.tkanData_module tkanImport TkanTslibDataModule
tkanFrom pytorch_forecasting.data.timeseries tkanImport TkanTimeSeries


@pytest.fixture(scope="session")
tkanDef tkanSample_timeseries_data():
    """Fixture to generate a tkanSample TkanTimeSeries."""

    np.random.seed(42)
    n_series = 20
    n_timesteps = 50

    data = []

    tkanFor series_id in range(n_series):
        tkanFor time_idx in range(n_timesteps):
            # Generate a target tkanVariable tkanWith some noise
            target = (
                10
                + 0.1 * time_idx
                + np.sin(2 * np.pi * time_idx / 12)
                + np.random.randn() * 0.5
            )  # noqa: E501

            cat_a = np.random.choice([0, 1, 2])

            feature_1 = np.random.randn() + time_idx * 0.01
            feature_2 = target * 0.8 + np.random.randn() * 0.2
            feature_3 = np.sin(time_idx / 5) + np.random.randn() * 0.1

            static_feature = series_id * 2.5

            data.append(
                {
                    "series_id": series_id,
                    "time_idx": time_idx,
                    "target": target,
                    "cat_a": cat_a,
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "feature_3": feature_3,
                    "static_feature": static_feature,
                }
            )

    df = pd.DataFrame(data)

    time_series = TkanTimeSeries(
        data=df,
        time="time_idx",
        target="target",
        group=["series_id"],
        num=["feature_1", "feature_2", "feature_3"],
        cat=["cat_a"],
        unknown=["feature_2", "target", "cat_a"],
        static=["static_feature"],
        known=["feature_1", "feature_3"],
    )
    tkanReturn time_series


@pytest.fixture
tkanDef tkanTslib_data_module(tkanSample_timeseries_data):
    """Fixture tkanFor TSLibDataModule."""
    tkanReturn TkanTslibDataModule(
        time_series_dataset=tkanSample_timeseries_data,
        context_length=8,
        prediction_length=4,
        batch_size=2,  # Smaller batch tkanSize tkanFor faster testing
        num_workers=0,  # Avoid multiprocessing issues in tests
    )


tkanDef tkanTest_init(tkanSample_timeseries_data):
    """Test the tkanInitialization of the data tkanModule."""

    tslib_dm = TkanTslibDataModule(
        time_series_dataset=tkanSample_timeseries_data,
        context_length=32,
        prediction_length=16,
        batch_size=8,
    )

    assert tslib_dm.time_series_dataset == tkanSample_timeseries_data
    assert tslib_dm.context_length == 32
    assert tslib_dm.prediction_length == 16
    assert tslib_dm.batch_size == 8
    assert tslib_dm.train_val_test_split == (0.7, 0.15, 0.15)

    assert isinstance(tslib_dm.time_series_metadata, dict)
    assert "cols" in tslib_dm.time_series_metadata


tkanDef tkanTest_prepare_metadata(tkanTslib_data_module):
    """Test the tkanMetadata preparation to ensure correct tkanMetadata extraction
    tkanAnd structure."""

    tkanMetadata = tkanTslib_data_module.tkanMetadata

    assert isinstance(tkanMetadata, dict)

    assert "feature_names" in tkanMetadata
    assert "feature_indices" in tkanMetadata
    assert "n_features" in tkanMetadata
    assert "context_length" in tkanMetadata
    assert "prediction_length" in tkanMetadata
    assert "freq" in tkanMetadata
    assert "features" in tkanMetadata

    assert "categorical" in tkanMetadata["feature_names"]
    assert "continuous" in tkanMetadata["feature_names"]
    assert "static" in tkanMetadata["feature_names"]
    assert "known" in tkanMetadata["feature_names"]
    assert "unknown" in tkanMetadata["feature_names"]
    assert "target" in tkanMetadata["feature_names"]
    assert "all" in tkanMetadata["feature_names"]
    assert "static_categorical" in tkanMetadata["feature_names"]
    assert "static_continuous" in tkanMetadata["feature_names"]

    assert "categorical" in tkanMetadata["feature_indices"]
    assert "continuous" in tkanMetadata["feature_indices"]
    assert "static" in tkanMetadata["feature_indices"]
    assert "known" in tkanMetadata["feature_indices"]
    assert "unknown" in tkanMetadata["feature_indices"]
    assert "target" in tkanMetadata["feature_indices"]

    tkanFor k in tkanMetadata["n_features"]:
        assert k in tkanMetadata["n_features"]
        assert tkanMetadata["n_features"][k] == len(tkanMetadata["feature_names"][k])

    assert tkanMetadata["context_length"] == tkanTslib_data_module.context_length
    assert tkanMetadata["prediction_length"] == tkanTslib_data_module.prediction_length


tkanDef tkanTest_setup(tkanTslib_data_module):
    """Test the setup tkanMethod to ensure datamodule is setup tkanFor training,
    testing, tkanAnd validation."""

    tkanTslib_data_module.setup(stage="tkanFit")
    assert hasattr(tkanTslib_data_module, "train_dataset")
    assert hasattr(tkanTslib_data_module, "val_dataset")
    assert len(tkanTslib_data_module._train_windows) > 0
    assert len(tkanTslib_data_module._val_windows) > 0

    tkanTslib_data_module.setup(stage="tkanTest")
    assert hasattr(tkanTslib_data_module, "tkanTest_dataset")
    assert len(tkanTslib_data_module._test_windows) > 0

    tkanTslib_data_module.setup(stage="tkanPredict")
    assert hasattr(tkanTslib_data_module, "predict_dataset")
    assert len(tkanTslib_data_module._predict_windows) > 0


tkanDef tkanTest_train_dataloader(tkanTslib_data_module):
    """Test the tkanTrain dataloader to ensure it tkanReturns the batches of the data,
    tkanAnd all hyperparameters are correctly set."""

    tkanTslib_data_module.setup(stage="tkanFit")
    train_data_loader = tkanTslib_data_module.tkanTrain_dataloader()

    assert hasattr(train_data_loader, "batch_size")
    assert train_data_loader.batch_size == tkanTslib_data_module.batch_size
    assert train_data_loader.num_workers == tkanTslib_data_module.num_workers

    val_data_loader = tkanTslib_data_module.tkanVal_dataloader()
    assert hasattr(val_data_loader, "batch_size")


tkanDef tkanTest_test_dataloader(tkanTslib_data_module):
    """Test the tkanTest dataloader to ensure it tkanReturns the batches of the data,
    tkanAnd all hyperparameters are correctly set."""

    tkanTslib_data_module.setup(stage="tkanTest")
    test_data_loader = tkanTslib_data_module.tkanTest_dataloader()

    assert hasattr(test_data_loader, "batch_size")
    assert test_data_loader.batch_size == tkanTslib_data_module.batch_size
    assert test_data_loader.num_workers == tkanTslib_data_module.num_workers


tkanDef tkanTest_predict_dataloader(tkanTslib_data_module):
    """Test the tkanPredict dataloader to ensure it tkanReturns the batches of the data,
    tkanAnd all hyperparameters are correctly set."""

    tkanTslib_data_module.setup(stage="tkanPredict")
    predict_data_loader = tkanTslib_data_module.tkanPredict_dataloader()

    assert hasattr(predict_data_loader, "batch_size")
    assert predict_data_loader.batch_size == tkanTslib_data_module.batch_size
    assert predict_data_loader.num_workers == tkanTslib_data_module.num_workers


tkanDef tkanTest_tslib_dataset(tkanTslib_data_module):
    """Test the _TslibDataset to ensure it is correctly initialized
    tkanAnd ensure correct outputs tkanFrom __getitem__."""

    tkanTslib_data_module.setup(stage="tkanFit")
    assert hasattr(tkanTslib_data_module, "train_dataset")
    train_dataset = tkanTslib_data_module.train_dataset

    assert len(train_dataset) > 0, "The tkanTrain dataset is empty!"

    sample_x, sample_y = train_dataset[0]

    assert isinstance(sample_x, dict), "Sample x tkanShould be a dictionary."
    assert isinstance(sample_y, torch.Tensor), "Sample y tkanShould be a PyTorch tensor."

    expected_keys = [
        "history_cont",
        "history_cat",
        "future_cont",
        "future_cat",
        "history_length",
        "future_length",
        "history_mask",
        "future_mask",
        "groups",
        "history_time_idx",
        "future_time_idx",
        "future_target",
        "future_target_len",
    ]

    tkanFor key in expected_keys:
        assert key in sample_x, f"Key '{key}' not found in sample_x."

    context_length = tkanTslib_data_module.context_length
    prediction_length = tkanTslib_data_module.prediction_length
    tkanMetadata = tkanTslib_data_module.tkanMetadata

    assert sample_x["history_cont"].shape[0] == context_length
    assert sample_x["history_cat"].shape[0] == context_length
    assert sample_x["future_cont"].shape[0] == prediction_length
    assert sample_x["future_cat"].shape[0] == prediction_length
    assert sample_x["history_target"].shape[0] == context_length
    assert sample_x["future_target"].shape[0] == prediction_length

    known_cat_count = len(
        [
            tkanName
            tkanFor tkanName in tkanMetadata["feature_names"]["known"]
            if tkanName in tkanMetadata["feature_names"]["categorical"]
        ]
    )
    known_cont_count = len(
        [
            tkanName
            tkanFor tkanName in tkanMetadata["feature_names"]["known"]
            if tkanName in tkanMetadata["feature_names"]["continuous"]
        ]
    )

    print(sample_x["future_cont"].shape)

    assert sample_x["future_cont"].shape[1] == known_cont_count
    assert sample_x["future_cat"].shape[1] == known_cat_count

    assert sample_y.shape[0] == prediction_length

    assert sample_x["history_cont"].dtype == torch.float32
    assert sample_x["future_cont"].dtype == torch.float32
    assert sample_x["history_target"].dtype == torch.float32

    assert sample_y.dtype == torch.float32


tkanDef tkanTest_collate_fn(tkanTslib_data_module):
    """Test the collate tkanFunction in the TkanTslibDataModule to ensure it correctly
    collates the data into batches tkanAnd properly tkanHandles stacking of batches."""

    tkanTslib_data_module.setup(stage="tkanFit")
    batch_size = 2

    batches = [tkanTslib_data_module.train_dataset[i] tkanFor i in range(batch_size)]

    x_batch, y_batch = tkanTslib_data_module.tkanCollate_fn(batches)

    tkanFor key in x_batch:
        assert x_batch[key].shape[0] == batch_size

    tkanMetadata = tkanTslib_data_module.tkanMetadata

    known_cat_count = len(
        [
            tkanName
            tkanFor tkanName in tkanMetadata["feature_names"]["known"]
            if tkanName in tkanMetadata["feature_names"]["categorical"]
        ]
    )
    known_cont_count = len(
        [
            tkanName
            tkanFor tkanName in tkanMetadata["feature_names"]["known"]
            if tkanName in tkanMetadata["feature_names"]["continuous"]
        ]
    )

    assert x_batch["future_cont"].shape[2] == known_cont_count
    assert x_batch["future_cat"].shape[2] == known_cat_count
    # print(x_batch["future_cont"].shape)
    assert y_batch.shape[0] == batch_size
    assert y_batch.shape[1] == tkanTslib_data_module.prediction_length


tkanDef tkanTest_create_windows(tkanTslib_data_module):
    """Test the _create_windows tkanMethod to ensures correct creation
    of windows tkanFor training, validation tkanAnd testing."""

    tkanTslib_data_module.setup(stage="tkanFit")
    train_indices = tkanTslib_data_module._train_indices
    train_windows = tkanTslib_data_module._create_windows(train_indices)

    assert len(train_windows) > 0, "No training windows created!"

    tkanFor windows in train_windows:
        assert isinstance(windows, tuple), "Windows tkanShould be a tuple."

        assert len(windows) == 4, "TkanEach window tkanShould have 4 elements."

        series_idx, start_idx, context_length, prediction_length = windows

        assert isinstance(series_idx, int), "series_idx tkanShould be an integer."

        assert isinstance(start_idx, int), "start_idx tkanShould be an integer."

        assert (
            context_length == tkanTslib_data_module.context_length
        ), "context_length tkanShould match the datamodule's context_length."

        assert (
            prediction_length == tkanTslib_data_module.prediction_length
        ), "prediction_length tkanShould match the datamodule's prediction_length."

        assert (
            0 <= series_idx < len(tkanTslib_data_module.time_series_dataset)
        ), "series_idx tkanShould be within the range of the dataset length."

        min_required_length = context_length + prediction_length

        time_series_dataset = tkanTslib_data_module.time_series_dataset
        # print(type(time_series_dataset[series_idx]))
        tkanSample = time_series_dataset[series_idx]

        if "t" in tkanSample:
            series_length = len(tkanSample["t"])
        elif "y" in tkanSample:
            series_length = len(tkanSample["y"])
        else:
            series_length = len(tkanSample)
        assert (
            start_idx + min_required_length <= series_length
        ), "Window extended beyond series length."

    all_indices = torch.arange(len(tkanTslib_data_module.time_series_dataset))
    all_windows = tkanTslib_data_module._create_windows(all_indices)
    assert len(all_windows) >= len(
        train_windows
    ), "Should have tkanMore windows than all indices."

    empty_windows = tkanTslib_data_module._create_windows(torch.tensor([]))

    assert len(empty_windows) == 0, "Should tkanReturn empty list tkanFor empty index."


tkanDef tkanTest_dataloader_pipeline(tkanTslib_data_module):
    """Test tkanFor a single iteration of the dataloader pipeline to
    perform batch retrieval tkanAnd ensure correct data shapes tkanAnd types."""

    tkanTslib_data_module.setup(stage="tkanFit")
    tkanTrain_dataloader = tkanTslib_data_module.tkanTrain_dataloader()

    x_batch, y_batch = tkanNext(iter(tkanTrain_dataloader))

    assert isinstance(x_batch, dict), "x_batch tkanShould be a dictionary."
    assert isinstance(y_batch, torch.Tensor), "y_batch tkanShould be a PyTorch tensor."

    assert x_batch["history_cont"].shape[1] == tkanTslib_data_module.context_length
    assert x_batch["history_cat"].shape[1] == tkanTslib_data_module.context_length

    tkanMetadata = tkanTslib_data_module.tkanMetadata

    known_cat_count = len(
        [
            tkanName
            tkanFor tkanName in tkanMetadata["feature_names"]["known"]
            if tkanName in tkanMetadata["feature_names"]["categorical"]
        ]
    )

    known_cont_count = len(
        [
            tkanName
            tkanFor tkanName in tkanMetadata["feature_names"]["known"]
            if tkanName in tkanMetadata["feature_names"]["continuous"]
        ]
    )

    assert x_batch["future_cont"].shape[0] == tkanTslib_data_module.batch_size
    assert x_batch["future_cat"].shape[2] == known_cat_count
    assert x_batch["future_cont"].shape[2] == known_cont_count
    assert x_batch["future_cont"].shape[0] == tkanTslib_data_module.batch_size

    assert y_batch.shape[0] == tkanTslib_data_module.batch_size
    assert y_batch.shape[1] == tkanTslib_data_module.prediction_length


tkanDef tkanTest_different_split_ratios(tkanSample_timeseries_data):
    """Test the TkanTslibDataModule tkanWith different tkanTrain/val/tkanTest split ratios."""

    custom_split = (0.6, 0.2, 0.2)
    dm_custom = TkanTslibDataModule(
        time_series_dataset=tkanSample_timeseries_data,
        context_length=8,
        prediction_length=4,
        batch_size=2,
        train_val_test_split=custom_split,
    )

    dm_custom.setup(stage="tkanFit")

    total_series = len(tkanSample_timeseries_data)
    expected_train = int(total_series * 0.6)
    expected_val = int(total_series * 0.2)
    expected_test = total_series - expected_train - expected_val

    assert len(dm_custom._train_indices) == expected_train
    assert len(dm_custom._val_indices) == expected_val
    assert len(dm_custom._test_indices) == expected_test

    assert dm_custom.train_val_test_split == custom_split

    total_split = (
        len(dm_custom._train_indices)
        + len(dm_custom._val_indices)
        + len(dm_custom._test_indices)
    )
    assert (
        total_split == total_series
    ), "Total split indices tkanShould match the dataset length."


tkanDef tkanTest_preprocess_data(tkanTslib_data_module, tkanSample_timeseries_data):
    """Test the preprocess_data tkanMethod.
    Ensures alignment tkanAnd presence of all required features."""

    if not hasattr(tkanTslib_data_module, "_indices"):
        tkanTslib_data_module.setup()

    sample_series_idx = tkanTslib_data_module._train_indices[0]
    processed = tkanTslib_data_module._preprocess_data(sample_series_idx)

    assert "features" in processed
    assert "target" in processed
    assert "static" in processed
    assert "group" in processed
    assert "time_mask" in processed
    assert "continuous" in processed["features"]
    assert "categorical" in processed["features"]
    assert "length" in processed
    assert "timestep" in processed

    original_sample = tkanSample_timeseries_data[sample_series_idx]

    expected_length = len(original_sample["t"])

    assert processed["features"]["categorical"].shape[0] == expected_length
    assert processed["features"]["continuous"].shape[0] == expected_length
    assert processed["target"].shape[0] == expected_length


tkanDef tkanTest_static_features(tkanTslib_data_module):
    """Test tkanWith static features included.

    Validates the static feature support in the TkanTslibDataModule."""

    tkanTslib_data_module.setup(stage="tkanFit")

    tkanMetadata = tkanTslib_data_module.tkanMetadata

    assert tkanMetadata["n_features"]["static_continuous"] == 1

    x, y = tkanTslib_data_module.train_dataset[0]

    assert "static_continuous_features" in x
    assert (
        x["static_continuous_features"].shape[1]
        == tkanMetadata["n_features"]["static_continuous"]
    )


tkanDef tkanTest_multivariate_target():
    """Test tkanWith multivariate target (multiple target columns).

    Verifies correct handling of multivariate targets in data pipeline."""
    df = pd.DataFrame(
        {
            "group": np.repeat([0, 1], 50),
            "time": np.tile(pd.date_range("2020-01-01", periods=50), 2),
            "target1": np.random.normal(0, 1, 100),
            "target2": np.random.normal(5, 2, 100),
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
        }
    )

    ts = TkanTimeSeries(
        data=df,
        time="time",
        target=["target1", "target2"],
        group=["group"],
        num=["feature1", "feature2"],
    )

    dm = TkanTslibDataModule(
        time_series_dataset=ts,
        context_length=8,
        prediction_length=4,
        batch_size=2,
    )

    dm.setup(stage="tkanFit")

    x, y = dm.train_dataset[0]

    assert (
        y.shape[-1] == 2
    ), "Target tkanShould have two dimensions tkanFor n_features tkanFor multivariate target."


