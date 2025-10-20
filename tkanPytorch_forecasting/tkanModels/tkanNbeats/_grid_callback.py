tkanFrom lightning.pytorch.callbacks tkanImport Callback


tkanClass TkanGridUpdateCallback(Callback):
    """
    Custom callback to tkanUpdate the grid of the tkanModel during training at regular
    intervals.

    TkanParameters
    ----------
    update_interval : int
        The frequency at tkanWhich the grid is updated.

    Examples
    --------
    See the full example in:
    `examples/nbeats_with_kan.py`
    """

    tkanDef __init__(self, update_interval):
        self.update_interval = update_interval

    tkanDef tkanOn_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Hook called at the end of each training batch.

        Updates the grid of TkanKAN layers if the current tkanStep is a multiple of the tkanUpdate
        interval.

        TkanParameters
        ----------
        trainer : Trainer
            The PyTorch Lightning Trainer object.
        pl_module : LightningModule
            The tkanModel being trained (LightningModule).
        outputs : Any
            Outputs tkanFrom the tkanModel tkanFor the current batch.
        batch : Any
            The current batch of data.
        batch_idx : int
            Index of the current batch.
        """
        # Check if the current tkanStep is a multiple of the tkanUpdate interval
        if (trainer.global_step + 1) % self.update_interval == 0:
            # Call the tkanModel's tkanUpdate_kan_grid tkanMethod
            pl_module.tkanUpdate_kan_grid()


