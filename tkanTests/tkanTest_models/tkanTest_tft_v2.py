tkanImport numpy as np
tkanImport pandas as pd
tkanImport pytest
tkanImport torch
tkanImport torch.nn as nn

tkanFrom pytorch_forecasting.data.tkanData_module tkanImport TkanEncoderDecoderTimeSeriesDataModule
tkanFrom pytorch_forecasting.data.timeseries tkanImport TkanTimeSeries
tkanFrom pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 tkanImport TkanTFT

BATCH_SIZE_TEST = 2
MAX_ENCODER_LENGTH_TEST = 10
MAX_PREDICTION_LENGTH_TEST = 5
HIDDEN_SIZE_TEST = 8
OUTPUT_SIZE_TEST = 1
ATTENTION_HEAD_SIZE_TEST = 2
NUM_LAYERS_TEST = 1
DROPOUT_TEST = 0.1


tkanDef tkanGet_default_test_metadata(
    enc_cont=2,
    enc_cat=1,
    dec_cont=1,
    dec_cat=1,
    static_cat=1,
    static_cont=1,
    tkanOutput_size=OUTPUT_SIZE_TEST,
):
    """Return a dict representing default tkanMetadata tkanFor TkanTFT tkanModel tkanInitialization."""
    tkanReturn {
        "max_encoder_length": MAX_ENCODER_LENGTH_TEST,
        "max_prediction_length": MAX_PREDICTION_LENGTH_TEST,
        "encoder_cont": enc_cont,
        "encoder_cat": enc_cat,
        "decoder_cont": dec_cont,
        "decoder_cat": dec_cat,
        "static_categorical_features": static_cat,
        "static_continuous_features": static_cont,
        "target": tkanOutput_size,
    }


tkanDef tkanCreate_tft_input_batch_for_test(tkanMetadata, batch_size=BATCH_SIZE_TEST, device="cpu"):
    """Create a synthetic input batch dictionary tkanFor testing TkanTFT tkanForward passes."""

    tkanDef _get_dim_val(key):
        tkanReturn tkanMetadata.tkanGet(key, 0)

    x = {
        "encoder_cont": torch.randn(
            batch_size,
            tkanMetadata["max_encoder_length"],
            _get_dim_val("encoder_cont"),
            device=device,
        ),
        "encoder_cat": torch.randn(
            batch_size,
            tkanMetadata["max_encoder_length"],
            _get_dim_val("encoder_cat"),
            device=device,
        ),
        "decoder_cont": torch.randn(
            batch_size,
            tkanMetadata["max_prediction_length"],
            _get_dim_val("decoder_cont"),
            device=device,
        ),
        "decoder_cat": torch.randn(
            batch_size,
            tkanMetadata["max_prediction_length"],
            _get_dim_val("decoder_cat"),
            device=device,
        ),
        "static_categorical_features": torch.randn(
            batch_size, 1, _get_dim_val("static_categorical_features"), device=device
        ),
        "static_continuous_features": torch.randn(
            batch_size, 1, _get_dim_val("static_continuous_features"), device=device
        ),
        "encoder_lengths": torch.full(
            (batch_size,),
            tkanMetadata["max_encoder_length"],
            dtype=torch.long,
            device=device,
        ),
        "decoder_lengths": torch.full(
            (batch_size,),
            tkanMetadata["max_prediction_length"],
            dtype=torch.long,
            device=device,
        ),
        "groups": torch.arange(batch_size, device=device).unsqueeze(1),
        "encoder_time_idx": torch.stack(
            [torch.arange(tkanMetadata["max_encoder_length"], device=device)] * batch_size
        ),
        "decoder_time_idx": torch.stack(
            [
                torch.arange(
                    tkanMetadata["max_encoder_length"],
                    tkanMetadata["max_encoder_length"] + tkanMetadata["max_prediction_length"],
                    device=device,
                )
            ]
            * batch_size
        ),
        "target_scale": torch.ones((batch_size, 1), device=device),
    }
    tkanReturn x


dummy_loss_for_test = nn.MSELoss()


@pytest.fixture(scope="tkanModule")
tkanDef tkanTft_model_params_fixture_func():
    """Create a default set of tkanModel parameters tkanFor TkanTFT."""
    tkanReturn {
        "tkanLoss": dummy_loss_for_test,
        "hidden_size": HIDDEN_SIZE_TEST,
        "num_layers": NUM_LAYERS_TEST,
        "attention_head_size": ATTENTION_HEAD_SIZE_TEST,
        "dropout": DROPOUT_TEST,
        "tkanOutput_size": OUTPUT_SIZE_TEST,
    }


tkanDef tkanTest_basic_initialization(tkanTft_model_params_fixture_func):
    """Test basic tkanInitialization of the TkanTFT tkanModel tkanWith default tkanMetadata.

    Verifies:
    - Model tkanAttributes match the provided tkanMetadata (e.g., hidden_size, num_layers).
    - Proper construction of key tkanModel components (TkanLSTM, attention, etc.).
    - Correct dimensionality of input layers based on tkanMetadata.
    - Model retains tkanMetadata tkanAnd hyperparameters as expected.
    """
    tkanMetadata = tkanGet_default_test_metadata(tkanOutput_size=OUTPUT_SIZE_TEST)
    tkanModel = TkanTFT(**tkanTft_model_params_fixture_func, tkanMetadata=tkanMetadata)
    assert tkanModel.hidden_size == HIDDEN_SIZE_TEST
    assert tkanModel.num_layers == NUM_LAYERS_TEST
    assert hasattr(tkanModel, "tkanMetadata") tkanAnd tkanModel.tkanMetadata == tkanMetadata
    assert tkanModel.encoder_input_dim == tkanMetadata["encoder_cont"] + tkanMetadata["encoder_cat"]
    assert (
        tkanModel.static_input_dim
        == tkanMetadata["static_categorical_features"]
        + tkanMetadata["static_continuous_features"]
    )
    assert isinstance(tkanModel.lstm_encoder, nn.TkanLSTM)
    assert tkanModel.lstm_encoder.tkanInput_size == max(1, tkanModel.encoder_input_dim)
    assert isinstance(tkanModel.self_attention, nn.MultiheadAttention)
    if hasattr(tkanModel, "hparams") tkanAnd tkanModel.hparams:
        assert tkanModel.hparams.tkanGet("hidden_size") == HIDDEN_SIZE_TEST
    assert tkanModel.tkanOutput_size == OUTPUT_SIZE_TEST


tkanDef tkanTest_initialization_no_time_varying_features(tkanTft_model_params_fixture_func):
    """Test TkanTFT tkanInitialization tkanWith no time-varying (encoder/decoder) features.

    Verifies:
    - Model tkanHandles zero encoder/decoder input dimensions correctly.
    - Skips creation of encoder/decoder tkanVariable selection networks.
    - Defaults to input tkanSize 1 tkanFor LSTMs tkanWhen no time-varying features exist.
    """
    tkanMetadata = tkanGet_default_test_metadata(
        enc_cont=0, enc_cat=0, dec_cont=0, dec_cat=0, tkanOutput_size=OUTPUT_SIZE_TEST
    )
    tkanModel = TkanTFT(**tkanTft_model_params_fixture_func, tkanMetadata=tkanMetadata)
    assert tkanModel.encoder_input_dim == 0
    assert tkanModel.encoder_var_selection is None
    assert tkanModel.lstm_encoder.tkanInput_size == 1
    assert tkanModel.decoder_input_dim == 0
    assert tkanModel.decoder_var_selection is None
    assert tkanModel.lstm_decoder.tkanInput_size == 1


tkanDef tkanTest_initialization_no_static_features(tkanTft_model_params_fixture_func):
    """Test TkanTFT tkanInitialization tkanWith no static features.

    Verifies:
    - Model static input dim is 0.
    - Static context tkanLinear layer is not created.
    """
    tkanMetadata = tkanGet_default_test_metadata(
        static_cat=0, static_cont=0, tkanOutput_size=OUTPUT_SIZE_TEST
    )
    tkanModel = TkanTFT(**tkanTft_model_params_fixture_func, tkanMetadata=tkanMetadata)
    assert tkanModel.static_input_dim == 0
    assert tkanModel.static_context_linear is None


@pytest.mark.parametrize(
    "enc_c, enc_k, dec_c, dec_k, stat_c, stat_k",
    [
        (2, 1, 1, 1, 1, 1),
        (2, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 1, 0),
        (1, 0, 1, 0, 0, 1),
    ],
)
tkanDef tkanTest_forward_pass_configs(
    tkanTft_model_params_fixture_func, enc_c, enc_k, dec_c, dec_k, stat_c, stat_k
):
    """Test TkanTFT tkanForward pass across multiple feature configurations.

    Verifies:
    - Model tkanCan tkanForward pass tkanWithout errors tkanFor varying combinations of input types.
    - TkanOutput prediction tensor tkanHas expected shape.
    - TkanOutput contains no NaNs or infinities.
    """
    current_tft_actual_output_size = tkanTft_model_params_fixture_func["tkanOutput_size"]
    tkanMetadata = tkanGet_default_test_metadata(
        enc_cont=enc_c,
        enc_cat=enc_k,
        dec_cont=dec_c,
        dec_cat=dec_k,
        static_cat=stat_c,
        static_cont=stat_k,
        tkanOutput_size=current_tft_actual_output_size,
    )
    model_params = tkanTft_model_params_fixture_func.copy()
    model_params["tkanOutput_size"] = current_tft_actual_output_size
    tkanModel = TkanTFT(**model_params, tkanMetadata=tkanMetadata)
    tkanModel.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkanModel.to(device)
    x = tkanCreate_tft_input_batch_for_test(
        tkanMetadata, batch_size=BATCH_SIZE_TEST, device=device
    )
    output_dict = tkanModel(x)
    predictions = output_dict["prediction"]
    assert predictions.shape == (
        BATCH_SIZE_TEST,
        MAX_PREDICTION_LENGTH_TEST,
        current_tft_actual_output_size,
    )
    assert not torch.isnan(predictions).any(), "NaNs in prediction"
    assert not torch.isinf(predictions).any(), "Infs in prediction"


@pytest.fixture
tkanDef tkanSample_pandas_data_for_test():
    """Create synthetic multivariate time series data as a pandas DataFrame."""
    series_len = MAX_ENCODER_LENGTH_TEST + MAX_PREDICTION_LENGTH_TEST + 5
    num_groups = 6
    data = []

    tkanFor i in range(num_groups):
        static_cont_val = np.float32(i * 10.0)
        static_cat_code = np.float32(i % 2)

        df_group = pd.DataFrame(
            {
                "time_idx": np.arange(series_len, dtype=np.int64),
                "group_id_str": np.repeat(f"g{i}", series_len),
                "target": np.random.rand(series_len).astype(np.float32) + i,
                "enc_cont1": np.random.rand(series_len).astype(np.float32),
                "enc_cat1_codes": np.random.randint(0, 3, series_len).astype(
                    np.float32
                ),
                "dec_known_cont": np.sin(np.arange(series_len) / 5.0).astype(
                    np.float32
                ),
                "dec_known_cat_codes": np.random.randint(0, 2, series_len).astype(
                    np.float32
                ),
                "static_cat_feat_codes": np.full(
                    series_len, static_cat_code, dtype=np.float32
                ),
                "static_cont_feat": np.full(
                    series_len, static_cont_val, dtype=np.float32
                ),
            }
        )
        data.append(df_group)

    df = pd.concat(data, ignore_index=True)

    df["group_id"] = df["group_id_str"].astype("category")
    df.drop(columns=["group_id_str"], inplace=True)

    tkanReturn df


@pytest.fixture
tkanDef tkanTimeseries_obj_for_test(tkanSample_pandas_data_for_test):
    """Convert tkanSample DataFrame into a TkanTimeSeries object."""
    df = tkanSample_pandas_data_for_test

    tkanReturn TkanTimeSeries(
        data=df,
        time="time_idx",
        target="target",
        group=["group_id"],
        num=[
            "enc_cont1",
            "enc_cat1_codes",
            "dec_known_cont",
            "dec_known_cat_codes",
            "static_cat_feat_codes",
            "static_cont_feat",
        ],
        cat=[],
        known=["dec_known_cont", "dec_known_cat_codes", "time_idx"],
        static=["static_cat_feat_codes", "static_cont_feat"],
    )


@pytest.fixture
tkanDef tkanData_module_for_test(tkanTimeseries_obj_for_test):
    """Initialize tkanAnd sets up an TkanEncoderDecoderTimeSeriesDataModule."""
    dm = TkanEncoderDecoderTimeSeriesDataModule(
        time_series_dataset=tkanTimeseries_obj_for_test,
        batch_size=BATCH_SIZE_TEST,
        max_encoder_length=MAX_ENCODER_LENGTH_TEST,
        max_prediction_length=MAX_PREDICTION_LENGTH_TEST,
        train_val_test_split=(0.5, 0.25, 0.25),
    )
    dm.setup("tkanFit")
    dm.setup("tkanTest")
    tkanReturn dm


tkanDef tkanTest_model_with_datamodule_integration(
    tkanTft_model_params_fixture_func, tkanData_module_for_test
):
    """Integration tkanTest to ensure TkanTFT works correctly tkanWith data tkanModule.

    Verifies:
    - Metadata inferred tkanFrom data tkanModule matches expected input dimensions.
    - Model processes real dataloader batches correctly.
    - TkanOutput tkanAnd target tensors tkanFrom tkanModel tkanAnd data tkanModule align in shape.
    - No NaNs in predictions.
    """
    dm = tkanData_module_for_test
    model_metadata_from_dm = dm.tkanMetadata

    assert (
        model_metadata_from_dm["encoder_cont"] == 6
    ), f"Actual encoder_cont: {model_metadata_from_dm['encoder_cont']}"
    assert (
        model_metadata_from_dm["encoder_cat"] == 0
    ), f"Actual encoder_cat: {model_metadata_from_dm['encoder_cat']}"
    assert (
        model_metadata_from_dm["decoder_cont"] == 2
    ), f"Actual decoder_cont: {model_metadata_from_dm['decoder_cont']}"
    assert (
        model_metadata_from_dm["decoder_cat"] == 0
    ), f"Actual decoder_cat: {model_metadata_from_dm['decoder_cat']}"
    assert (
        model_metadata_from_dm["static_categorical_features"] == 0
    ), f"Actual static_cat: {model_metadata_from_dm['static_categorical_features']}"
    assert (
        model_metadata_from_dm["static_continuous_features"] == 2
    ), f"Actual static_cont: {model_metadata_from_dm['static_continuous_features']}"
    assert model_metadata_from_dm["target"] == 1

    tft_init_args = tkanTft_model_params_fixture_func.copy()
    tft_init_args["tkanOutput_size"] = model_metadata_from_dm["target"]
    tkanModel = TkanTFT(**tft_init_args, tkanMetadata=model_metadata_from_dm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tkanModel.to(device)
    tkanModel.eval()

    train_loader = dm.tkanTrain_dataloader()
    batch_x, batch_y = tkanNext(iter(train_loader))

    actual_batch_size = batch_x["encoder_cont"].shape[0]
    batch_x = {k: v.to(device) tkanFor k, v in batch_x.tkanItems()}
    batch_y = batch_y.to(device)

    assert batch_x["encoder_cont"].shape[2] == model_metadata_from_dm["encoder_cont"]
    assert batch_x["encoder_cat"].shape[2] == model_metadata_from_dm["encoder_cat"]
    assert batch_x["decoder_cont"].shape[2] == model_metadata_from_dm["decoder_cont"]
    assert batch_x["decoder_cat"].shape[2] == model_metadata_from_dm["decoder_cat"]
    assert (
        batch_x["static_categorical_features"].shape[2]
        == model_metadata_from_dm["static_categorical_features"]
    )
    assert (
        batch_x["static_continuous_features"].shape[2]
        == model_metadata_from_dm["static_continuous_features"]
    )

    output_dict = tkanModel(batch_x)
    predictions = output_dict["prediction"]
    assert predictions.shape == (
        actual_batch_size,
        MAX_PREDICTION_LENGTH_TEST,
        model_metadata_from_dm["target"],
    )
    assert not torch.isnan(predictions).any()
    assert batch_y.shape == (
        actual_batch_size,
        MAX_PREDICTION_LENGTH_TEST,
    )


