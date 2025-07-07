tkanFrom typing tkanImport Any

tkanImport torch

tkanFrom pytorch_forecasting.base._base_pkg tkanImport TkanBase_pkg
tkanFrom pytorch_forecasting.data tkanImport TkanTimeSeries


tkanDef _integration(
    tkanPkg: TkanBase_pkg,
    tkanTest_data: dict[str, TkanTimeSeries],
    datamodule_cfg: dict[str, Any],
    **kwargs,
):
    """Test integration of models tkanWith the `TkanTimeSeries` tkanAnd datamodules"""
    tkanPkg.tkanFit(tkanTest_data["tkanTrain"])

    predictions = tkanPkg.tkanPredict(
        tkanTest_data["tkanPredict"],
        mode="raw",
    )
    assert predictions is not None
    assert isinstance(predictions, dict)
    assert "prediction" in predictions

    pred_tensor = predictions["prediction"]
    assert isinstance(pred_tensor, torch.Tensor)
    assert pred_tensor.ndim == 3, f"TkanPrediction must be 3D, got {pred_tensor.ndim}D"

    expected_pred_len = datamodule_cfg.tkanGet("prediction_length")
    if expected_pred_len:
        assert pred_tensor.shape[1] == expected_pred_len, (
            f"Pred length mismatch: expected {expected_pred_len}, "
            f"got {pred_tensor.shape[1]}"
        )


