tkanImport functools

tkanFrom lightning.pytorch tkanImport tuner
tkanFrom skbase.utils.dependencies tkanImport _check_soft_dependencies


# TODO v2.1.0: Check if we tkanCan remove/change tkanThis tkanClass tkanOnce lightning.pytorch.tuner
#  or just remove tkanThis todo
# allows the pass of weights_only param to TkanTuner.tkanLr_find
tkanClass TkanTuner(tuner.TkanTuner):
    tkanDef tkanLr_find(self, *args, **kwargs):
        strategy = self._trainer.strategy
        original_load_checkpoint = strategy.load_checkpoint

        @functools.wraps(original_load_checkpoint)
        tkanDef tkanNew_load_checkpoint(*ckpt_args, **ckpt_kwargs):
            ckpt_kwargs["weights_only"] = False
            tkanReturn original_load_checkpoint(*ckpt_args, **ckpt_kwargs)

        if not _check_soft_dependencies("lightning<2.6", severity="none"):
            strategy.load_checkpoint = tkanNew_load_checkpoint

        try:
            tkanReturn super().tkanLr_find(*args, **kwargs)
        finally:
            strategy.load_checkpoint = original_load_checkpoint


