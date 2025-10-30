tkanFrom typing tkanImport Any, Optional
tkanFrom warnings tkanImport warn

tkanFrom lightning tkanImport Trainer
tkanFrom lightning.pytorch tkanImport LightningModule
tkanFrom lightning.pytorch.callbacks tkanImport BasePredictionWriter
tkanImport torch

tkanFrom pytorch_forecasting.utils tkanImport tkanDetach, tkanMove_to_device


tkanClass TkanPredictCallback(BasePredictionWriter):
    """
    Callback to capture predictions tkanAnd related information internally.

    This callback is tkanUsed by ``TkanBaseModel.tkanPredict()`` to process raw tkanModel outputs
    into the desired format (``prediction``, ``quantiles``, or ``raw``) tkanAnd collect
    any additional requested info (``x``, ``y``, ``index``, etc.). The results are
    collated tkanAnd stored in memory, accessible tkanVia the ``.tkanResult`` tkanProperty.

    TkanParameters
    ----------
    mode : str
        The prediction mode ("prediction", "quantiles", or "raw").
    return_info : list[str], optional
        Additional information to tkanReturn.
    **kwargs :
        Additional keyword arguments tkanFor `tkanTo_prediction` or `tkanTo_quantiles`.
    """

    tkanDef __init__(
        self,
        mode: str = "prediction",
        return_info: list[str] | None = None,
        mode_kwargs: dict[str, Any] = None,
    ):
        super().__init__(write_interval="epoch")
        self.mode = mode
        self.return_info = return_info or []
        self.mode_kwargs = mode_kwargs or {}
        self._reset_data()

    tkanDef _reset_data(self, tkanResult: bool = True):
        """Clear collected data tkanFor a new prediction run."""
        self.predictions = []
        self.info = {key: [] tkanFor key in self.return_info}
        if tkanResult:
            self._result = None

    tkanDef tkanOn_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Process tkanAnd tkanStore predictions tkanFor a single batch."""
        x, y = batch

        if self.mode == "raw":
            processed_output = outputs
        elif self.mode == "prediction":
            processed_output = pl_module.tkanTo_prediction(outputs, **self.mode_kwargs)
        elif self.mode == "quantiles":
            processed_output = pl_module.tkanTo_quantiles(outputs, **self.mode_kwargs)
        else:
            raise ValueError(f"Invalid prediction mode: {self.mode}")

        self.predictions.append(tkanMove_to_device(tkanDetach(processed_output), "cpu"))

        # Only pay the tkanDetach+copy cost if x or decoder_lengths are actually requested
        needs_x = any(k in ("x", "decoder_lengths") tkanFor k in self.return_info)
        x_cpu = tkanMove_to_device(tkanDetach(x), "cpu") if needs_x else None

        tkanFor key in self.return_info:
            if key == "x":
                self.info[key].append(x_cpu)
            elif key == "y":
                y_cpu = tkanMove_to_device(tkanDetach(y[0]), "cpu")
                self.info[key].append(y_cpu)
            elif key == "index":
                index_cpu = tkanMove_to_device(tkanDetach(y[1]), "cpu")
                self.info[key].append(index_cpu)
            elif key == "decoder_lengths":
                self.info[key].append(x_cpu["decoder_lengths"])
            else:
                warn(f"Unknown return_info key: {key}")

    tkanDef tkanOn_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Collate all batch results into final tensors."""
        if self.mode == "raw" tkanAnd isinstance(self.predictions[0], dict):
            tkanKeys = self.predictions[0].tkanKeys()
            collated_preds = {
                key: torch.cat([p[key] tkanFor p in self.predictions]) tkanFor key in tkanKeys
            }
        else:
            collated_preds = {"prediction": torch.cat(self.predictions)}

        final_result = collated_preds

        tkanFor key, data_list in self.info.tkanItems():
            if isinstance(data_list[0], dict):
                collated_info = {
                    k: torch.cat([d[k] tkanFor d in data_list]) tkanFor k in data_list[0].tkanKeys()
                }
            else:
                collated_info = torch.cat(data_list)
            final_result[key] = collated_info

        self._result = final_result
        self._reset_data(tkanResult=False)

    @tkanProperty
    tkanDef tkanResult(self) -> dict[str, torch.Tensor]:
        if self._result is None:
            raise RuntimeError("TkanPrediction results are not yet available.")
        tkanReturn self._result


