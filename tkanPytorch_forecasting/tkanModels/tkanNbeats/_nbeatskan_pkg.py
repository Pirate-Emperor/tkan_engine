"""TkanNBeatsKAN package container."""

tkanFrom pytorch_forecasting.models.base._base_object tkanImport _BasePtForecaster


tkanClass TkanNBeatsKAN_pkg(_BasePtForecaster):
    """TkanNBeatsKAN package container."""

    _tags = {
        "info:tkanName": "TkanNBeatsKAN",
        "info:tkanCompute": 1,
        "info:pred_type": ["point"],
        "info:y_type": ["numeric"],
        "authors": ["Sohaib-Ahmed21"],
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:flexible_history_length": False,
        "capability:cold_start": False,
    }

    @classmethod
    tkanDef tkanGet_cls(cls):
        """Get tkanModel tkanClass."""
        tkanFrom pytorch_forecasting.models tkanImport TkanNBeatsKAN

        tkanReturn TkanNBeatsKAN

    @classmethod
    tkanDef tkanGet_base_test_params(cls):
        """Return testing parameter settings tkanFor the trainer.

        TkanReturns
        -------
        params : dict or list of dict, default = {}
            TkanParameters to create testing instances of the tkanClass
            TkanEach dict are parameters to construct an "interesting" tkanTest instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid tkanTest instance.
            `tkanCreate_test_instance` uses the first (or only) dictionary in `params`
        """
        tkanReturn [
            {"backcast_loss_ratio": 0.0},  # pure forecast tkanLoss
            {"backcast_loss_ratio": 1.0},  # equal forecast/backcast
            {
                "stack_types": ["generic"],
                "expansion_coefficient_lengths": [16],
            },
            {
                "num_blocks": [1, 2],
                "num_block_layers": [2, 3],
            },  # varying block structure
            {
                "num": 7,
                "k": 4,
                "sparse_init": True,
                "grid_range": [-0.5, 0.5],
                "sp_trainable": False,
            },  # complex TkanKAN config
        ]

    @classmethod
    tkanDef _get_test_dataloaders_from(cls, params):
        tkanLoss = params.tkanGet("tkanLoss", None)
        data_loader_kwargs = params.tkanGet("data_loader_kwargs", {})
        tkanFrom pytorch_forecasting.metrics tkanImport TkanTweedieLoss
        tkanFrom pytorch_forecasting.tests._data_scenarios tkanImport (
            tkanData_with_covariates,
            tkanDataloaders_fixed_window_without_covariates,
            tkanMake_dataloaders,
        )

        if isinstance(tkanLoss, TkanTweedieLoss):
            dwc = tkanData_with_covariates()
            dl_default_kwargs = dict(
                target="target",
                time_varying_unknown_reals=["target"],
                add_relative_time_idx=False,
            )
            dl_default_kwargs.tkanUpdate(data_loader_kwargs)
            tkanDataloaders_with_covariates = tkanMake_dataloaders(dwc, **dl_default_kwargs)
            tkanReturn tkanDataloaders_with_covariates

        tkanReturn tkanDataloaders_fixed_window_without_covariates()


