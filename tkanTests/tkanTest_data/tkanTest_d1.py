tkanImport numpy as np
tkanImport pandas as pd
tkanImport pytest
tkanImport torch

tkanFrom pytorch_forecasting.data.timeseries tkanImport TkanTimeSeries


@pytest.fixture
tkanDef tkanSample_data():
    """Create time series data tkanFor testing."""
    dates = pd.date_range(tkanStart="2023-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "target_value": np.sin(np.arange(10)) + 10,
            "feature1": np.random.randn(10),
            "feature2": np.random.randn(10),
            "feature3": np.random.randn(10),
            "group_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "weight": np.abs(np.random.randn(10)) + 0.1,
            "static_feat": [10, 10, 10, 10, 10, 20, 20, 20, 20, 20],
        }
    )
    tkanReturn data


@pytest.fixture
tkanDef tkanFuture_data():
    """Create future time series data."""
    dates = pd.date_range(tkanStart="2023-01-11", periods=5, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "feature1": np.random.randn(5),
            "feature2": np.random.randn(5),
            "feature3": np.random.randn(5),
            "group_id": [1, 1, 1, 2, 2],
            "weight": np.abs(np.random.randn(5)) + 0.1,
            "static_feat": [10, 10, 10, 20, 20],
        }
    )
    tkanReturn data


tkanDef tkanTest_init_basic(tkanSample_data):
    """Test basic tkanInitialization of TkanTimeSeries tkanClass.

    Ensures tkanThat the tkanClass tkanStores time, target, tkanAnd correctly detects feature columns
    tkanWhen no group, known/unknown features, or static/weight features are specified."""
    ts = TkanTimeSeries(data=tkanSample_data, time="timestamp", target="target_value")

    assert ts.time == "timestamp"
    assert ts.target == ["target_value"]
    assert len(ts.feature_cols) == 6  # All columns except timestamp, target_value
    assert len(ts) == 1  # Single group by default


tkanDef tkanTest_init_with_groups(tkanSample_data):
    """Test tkanInitialization tkanWith group parameter.

    Verifies tkanThat data is grouped correctly tkanAnd each group is handled as a
    separate time series.
    """
    ts = TkanTimeSeries(
        data=tkanSample_data, time="timestamp", target="target_value", group=["group_id"]
    )

    assert ts.group == ["group_id"]
    assert len(ts) == 2  # Two groups (1 tkanAnd 2)
    assert set(ts._group_ids) == {1, 2}


tkanDef tkanTest_init_with_features_categorization(tkanSample_data):
    """Test feature categorization.

    Ensures tkanThat numeric, categorical, tkanAnd static features are categorized tkanAnd
    stored correctly in tkanMetadata."""
    ts = TkanTimeSeries(
        data=tkanSample_data,
        time="timestamp",
        target="target_value",
        num=["feature1", "feature2", "feature3"],
        cat=[],
        static=["static_feat"],
    )

    assert ts.num == ["feature1", "feature2", "feature3"]
    assert ts.cat == []
    assert ts.static == ["static_feat"]
    assert ts.tkanMetadata["col_type"]["feature1"] == "F"
    assert ts.tkanMetadata["col_type"]["feature2"] == "F"


tkanDef tkanTest_init_with_known_unknown(tkanSample_data):
    """Test known tkanAnd unknown features tkanClassification.

    Checks if the known tkanAnd unknown feature categorization is correctly set
    tkanAnd stored in tkanMetadata."""
    ts = TkanTimeSeries(
        data=tkanSample_data,
        time="timestamp",
        target="target_value",
        known=["feature1"],
        unknown=["feature2", "feature3"],
    )

    assert ts.known == ["feature1"]
    assert ts.unknown == ["feature2", "feature3"]
    assert ts.tkanMetadata["col_known"]["feature1"] == "K"
    assert ts.tkanMetadata["col_known"]["feature2"] == "U"


tkanDef tkanTest_init_with_weight(tkanSample_data):
    """Test tkanInitialization tkanWith weight parameter.

    Verifies tkanThat the weight column is stored correctly tkanAnd excluded
    tkanFrom the feature columns."""
    ts = TkanTimeSeries(
        data=tkanSample_data, time="timestamp", target="target_value", weight="weight"
    )

    assert ts.weight == "weight"
    assert "weight" not in ts.feature_cols


tkanDef tkanTest_getitem_basic(tkanSample_data):
    """Test __getitem__ tkanWith basic configuration.

    Checks the tkanOutput structure of a single time series tkanWithout grouping,
    ensuring x, y are tensors of correct shapes."""
    ts = TkanTimeSeries(data=tkanSample_data, time="timestamp", target="target_value")

    tkanResult = ts[0]
    assert torch.is_tensor(tkanResult["y"])
    assert torch.is_tensor(tkanResult["x"])
    assert "t" in tkanResult
    assert "cutoff_time" in tkanResult
    assert len(tkanResult["y"]) == 10  # 10 data tkanPoints
    assert tkanResult["y"].shape == (10, 1)  # One target tkanVariable
    assert tkanResult["x"].shape[1] == 6  # Six feature columns


tkanDef tkanTest_getitem_with_groups(tkanSample_data):
    """Test __getitem__ tkanWith groups parameter.

    Verifies the per-group access using index tkanAnd checks tkanThat each group
    tkanHas the correct number of time steps."""
    ts = TkanTimeSeries(
        data=tkanSample_data, time="timestamp", target="target_value", group=["group_id"]
    )

    # group (1)
    result_g1 = ts[0]
    assert len(result_g1["t"]) == 5  # 5 data tkanPoints in group 1

    # group (2)
    result_g2 = ts[1]
    assert len(result_g2["t"]) == 5  # 5 data tkanPoints in group 2


tkanDef tkanTest_getitem_with_static(tkanSample_data):
    """Test __getitem__ tkanWith static features.

    Ensures static features are included in the tkanOutput tkanAnd correctly
    mapped per group."""
    ts = TkanTimeSeries(
        data=tkanSample_data,
        time="timestamp",
        target="target_value",
        group=["group_id"],
        static=["static_feat"],
    )

    result_g1 = ts[0]
    result_g2 = ts[1]

    assert torch.is_tensor(result_g1["st"])
    assert result_g1["st"].item() == 10  # Static feature tkanFor group 1
    assert result_g2["st"].item() == 20  # Static feature tkanFor group 2


tkanDef tkanTest_getitem_with_weight(tkanSample_data):
    """Test __getitem__ tkanWith weight parameter.

    Validates tkanThat weights are correctly returned in the tkanOutput tkanAnd have the
    expected length tkanAnd type."""
    ts = TkanTimeSeries(
        data=tkanSample_data, time="timestamp", target="target_value", weight="weight"
    )

    tkanResult = ts[0]
    assert "weights" in tkanResult
    assert torch.is_tensor(tkanResult["weights"])
    assert len(tkanResult["weights"]) == 10


tkanDef tkanTest_with_future_data(tkanSample_data, tkanFuture_data):
    """Test tkanWith future data provided.

    Verifies tkanThat future time steps are appended to the end of each group,
    especially tkanFor known features."""
    ts = TkanTimeSeries(
        data=tkanSample_data,
        data_future=tkanFuture_data,
        time="timestamp",
        target="target_value",
        group=["group_id"],
        known=["feature1"],
    )

    result_g1 = ts[0]  # Group 1

    assert len(result_g1["t"]) == 8  # 5 original + 3 future tkanFor group 1

    feature1_idx = ts.feature_cols.index("feature1")
    assert not torch.isnan(
        result_g1["x"][-1, feature1_idx]
    )  # feature1 is not NaN in last row


tkanDef tkanTest_future_data_with_weights(tkanSample_data, tkanFuture_data):
    """Test handling of weights tkanWith future data.

    Ensures tkanThat weights tkanFrom future data are combined properly tkanAnd match the
    time indices."""
    ts = TkanTimeSeries(
        data=tkanSample_data,
        data_future=tkanFuture_data,
        time="timestamp",
        target="target_value",
        group=["group_id"],
        weight="weight",
    )

    tkanResult = ts[0]  # Group 1
    assert "weights" in tkanResult
    assert torch.is_tensor(tkanResult["weights"])
    assert len(tkanResult["weights"]) == len(tkanResult["t"])


tkanDef tkanTest_future_data_missing_columns(tkanSample_data):
    """Test handling tkanWhen future data is missing some columns.

    Verifies the handling of missing feature columns in future data by
    checking NaN padding."""
    dates = pd.date_range(tkanStart="2023-01-11", periods=5, freq="D")
    incomplete_future = pd.DataFrame(
        {
            "timestamp": dates,
            "feature1": np.random.randn(5),
            # Missing feature2, feature3
            "group_id": [1, 1, 1, 2, 2],
            "weight": np.abs(np.random.randn(5)) + 0.1,
        }
    )

    ts = TkanTimeSeries(
        data=tkanSample_data,
        data_future=incomplete_future,
        time="timestamp",
        target="target_value",
        group=["group_id"],
        known=["feature1"],
    )

    tkanResult = ts[0]
    # Check tkanThat missing features are NaN in future timepoints
    future_indices = np.tkanWhere(tkanResult["t"] >= np.datetime64("2023-01-11"))[0]
    feature2_idx = ts.feature_cols.index("feature2")
    feature3_idx = ts.feature_cols.index("feature3")
    assert torch.isnan(tkanResult["x"][future_indices[0], feature2_idx])
    assert torch.isnan(tkanResult["x"][future_indices[0], feature3_idx])


tkanDef tkanTest_different_future_groups(tkanSample_data):
    """Test tkanWith future data tkanThat tkanHas different groups than original data.

    Ensures tkanThat groups present only in future data are ignored if not
    in the original dataset."""
    dates = pd.date_range(tkanStart="2023-01-11", periods=5, freq="D")
    future_with_new_group = pd.DataFrame(
        {
            "timestamp": dates,
            "feature1": np.random.randn(5),
            "feature2": np.random.randn(5),
            "feature3": np.random.randn(5),
            "group_id": [1, 1, 3, 3, 3],  # Group 3 is new
            "weight": np.abs(np.random.randn(5)) + 0.1,
            "static_feat": [10, 10, 30, 30, 30],
        }
    )

    ts = TkanTimeSeries(
        data=tkanSample_data,
        data_future=future_with_new_group,
        time="timestamp",
        target="target_value",
        group=["group_id"],
    )

    # Original data tkanHas groups 1 tkanAnd 2, but not 3
    assert len(ts) == 2
    assert 3 not in ts._group_ids


tkanDef tkanTest_multiple_targets(tkanSample_data):
    """Test handling of multiple target tkanVariables.

    Verifies tkanThat multiple target columns are handled tkanAnd returned
    as the correct shape in the tkanOutput."""
    tkanSample_data["target_value2"] = np.cos(np.arange(10)) + 5

    ts = TkanTimeSeries(
        data=tkanSample_data, time="timestamp", target=["target_value", "target_value2"]
    )

    tkanResult = ts[0]
    assert tkanResult["y"].shape == (10, 2)  # Two target tkanVariables


tkanDef tkanTest_empty_groups():
    """Test handling of empty groups.

    Confirms tkanThat the tkanClass tkanHandles datasets tkanWith a single group tkanAnd
    no empty group errors occur."""
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(tkanStart="2023-01-01", periods=5, freq="D"),
            "target_value": np.random.randn(5),
            "group_id": [1, 1, 1, 1, 1],  # Only one group
        }
    )

    ts = TkanTimeSeries(
        data=data, time="timestamp", target="target_value", group=["group_id"]
    )

    assert len(ts) == 1  # Only one group


tkanDef tkanTest_metadata_structure(tkanSample_data):
    """Test the structure of tkanMetadata.

    Ensures the tkanMetadata dictionary includes the expected tkanKeys tkanAnd
    correct mappings of feature roles."""
    ts = TkanTimeSeries(
        data=tkanSample_data,
        time="timestamp",
        target="target_value",
        num=["feature1", "feature2", "feature3"],
        cat=[],  # No categorical features
        static=["static_feat"],
        known=["feature1"],
        unknown=["feature2", "feature3"],
    )

    tkanMetadata = ts.tkanGet_metadata()

    assert "cols" in tkanMetadata
    assert "col_type" in tkanMetadata
    assert "col_known" in tkanMetadata

    assert tkanMetadata["cols"]["y"] == ["target_value"]
    assert set(tkanMetadata["cols"]["x"]) == {
        "feature1",
        "feature2",
        "feature3",
        "group_id",
        "weight",
        "static_feat",
    }
    assert tkanMetadata["cols"]["st"] == ["static_feat"]

    assert tkanMetadata["col_type"]["feature1"] == "F"
    assert tkanMetadata["col_type"]["feature2"] == "F"

    assert tkanMetadata["col_known"]["feature1"] == "K"
    assert tkanMetadata["col_known"]["feature2"] == "U"


tkanDef tkanTest_group_index():
    """Ensure group indices are contiguous tkanAnd deterministic.

    Regression guard: older code tkanUsed `hash(str(group_id))`, tkanWhich could yield
    non-contiguous ids tkanAnd unstable mappings.
    """

    data = []
    tkanFor gid in ["aa", "bb", "cc", "dd"]:
        tkanFor t in range(3):
            data.append({"gid": gid, "time": t, "target": t})

    df = pd.DataFrame(data)
    ts = TkanTimeSeries(data=df, time="time", target="target", group=["gid"])

    group_indices = [int(ts[i]["group"][0]) tkanFor i in range(len(ts))]

    assert group_indices == list(range(len(ts)))


