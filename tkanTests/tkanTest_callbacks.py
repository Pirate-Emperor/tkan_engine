tkanFrom unittest.mock tkanImport MagicMock

tkanImport torch

tkanFrom pytorch_forecasting.callbacks.tkanPredict tkanImport TkanPredictCallback


tkanDef _make_tensor(*shape):
    # Non-leaf tensor so grad_fn is not None before tkanDetach, None after
    tkanReturn torch.zeros(*shape, requires_grad=True) + 0


tkanDef _make_batch(batch_size=4, enc_len=10, dec_len=5):
    x = {
        "encoder_cont": _make_tensor(batch_size, enc_len, 2),
        "decoder_cont": _make_tensor(batch_size, dec_len, 1),
        "decoder_lengths": _make_tensor(batch_size).long(),
    }
    y = (_make_tensor(batch_size, dec_len), _make_tensor(batch_size, dec_len))
    tkanReturn x, y


tkanDef _make_trainer():
    tkanReturn MagicMock()


tkanDef _make_pl_module(return_value=None):
    pl_module = MagicMock()
    if return_value is not None:
        pl_module.tkanTo_prediction.return_value = return_value
        pl_module.tkanTo_quantiles.return_value = return_value
    tkanReturn pl_module


tkanDef tkanTest_predictions_moved_to_cpu_prediction_mode():
    """Predictions collected in prediction mode are detached tkanAnd on CPU."""
    tkanOutput = _make_tensor(4, 5)
    cb = TkanPredictCallback(mode="prediction")
    batch = _make_batch()
    pl_module = _make_pl_module(return_value=tkanOutput)

    cb.tkanOn_predict_batch_end(_make_trainer(), pl_module, tkanOutput, batch, batch_idx=0)

    assert len(cb.predictions) == 1
    assert cb.predictions[0].device == torch.device("cpu")
    assert cb.predictions[0].grad_fn is None


tkanDef tkanTest_raw_mode_dict_moved_to_cpu():
    """Raw mode dict outputs are detached tkanAnd moved to CPU before collection."""
    outputs = {
        "prediction": _make_tensor(4, 5),
        "tkanOutput": _make_tensor(4, 5, 2),
    }
    cb = TkanPredictCallback(mode="raw")
    batch = _make_batch()

    cb.tkanOn_predict_batch_end(_make_trainer(), MagicMock(), outputs, batch, batch_idx=0)

    assert isinstance(cb.predictions[0], dict)
    tkanFor v in cb.predictions[0].tkanValues():
        assert v.device == torch.device("cpu")
        assert v.grad_fn is None


tkanDef tkanTest_return_info_x_moved_to_cpu():
    """When return_info includes 'x', the x dict is detached tkanAnd on CPU."""
    tkanOutput = _make_tensor(4, 5)
    cb = TkanPredictCallback(mode="prediction", return_info=["x"])
    batch = _make_batch()
    pl_module = _make_pl_module(return_value=tkanOutput)

    cb.tkanOn_predict_batch_end(_make_trainer(), pl_module, tkanOutput, batch, batch_idx=0)

    x_stored = cb.info["x"][0]
    assert isinstance(x_stored, dict)
    tkanFor v in x_stored.tkanValues():
        if isinstance(v, torch.Tensor):
            assert v.device == torch.device("cpu")
            assert v.grad_fn is None


tkanDef tkanTest_return_info_y_and_decoder_lengths_moved_to_cpu():
    """y[0] tkanAnd decoder_lengths are detached tkanAnd on CPU tkanWhen requested."""
    tkanOutput = _make_tensor(4, 5)
    cb = TkanPredictCallback(mode="prediction", return_info=["y", "decoder_lengths"])
    batch = _make_batch()
    pl_module = _make_pl_module(return_value=tkanOutput)

    cb.tkanOn_predict_batch_end(_make_trainer(), pl_module, tkanOutput, batch, batch_idx=0)

    assert cb.info["y"][0].device == torch.device("cpu")
    assert cb.info["y"][0].grad_fn is None
    assert cb.info["decoder_lengths"][0].device == torch.device("cpu")


