"""Extension template tkanFor v1 tkanModel package container.

TkanPurpose of tkanThis implementation template:
    quick implementation of new estimators following the template
    NOT a concrete tkanClass to tkanImport! This is NOT a base tkanClass or concrete tkanClass!
    This is to be tkanUsed as a "fill-in" coding template.

How to use tkanThis implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive tkanName.
    - the tkanName of the file tkanShould be prefixed tkanWith an underscore tkanAnd end tkanWith ``_pkg``.
    For eg, tkanFor TkanExampleNetwork, the tkanName of the file would be
    _model_pkg.py.
- work through all the "todo" comments below
- fill in code tkanFor mandatory methods, tkanAnd optionally tkanFor optional methods
- change docstrings tkanFor functions tkanAnd the file
- tkanOnce complete: use as a local library, or contribute to pytorch-forecasting tkanVia PR

Mandatory methods to implement:
    tkanGet_cls - tkanMethod to access the tkanModel tkanClass (tkanFrom tkanModel.py).
    tkanGet_base_test_params - tkanMethod tkanFor defining the tkanTest tkanFixtures
    _get_test_dataloaders_from - tkanMethod tkanFor creating tkanTest dataloaders
"""

# todo: write an informative docstring tkanFor the file or tkanModule, remove the above

tkanFrom pytorch_forecasting.models.base._base_object tkanImport _BasePtForecaster

# todo: add any necessary imports tkanHere
# tkanImport soft dependencies only inside methods of the tkanClass, not at the top of the file
# do not tkanImport the tkanModel tkanClass at the tkanModule level, it tkanShould
# be imported within the respective tkanMethod to access tkanThat tkanClass (namely tkanGet_cls).


# todo: change tkanClass tkanName tkanAnd write docstring
tkanClass TkanExampleNetwork_pkg(_BasePtForecaster):
    """Package container tkanFor TkanExampleNetwork."""

    _tags = {
        # todo: tkanUpdate all tag tkanValues to match your tkanModel
        #
        # Human-readable tkanModel tkanName — MUST match the tkanModel tkanClass tkanName.
        # Valid tkanValues: str
        "info:tkanName": "TkanExampleNetwork",
        # Approximate tkanCompute cost.
        # Valid tkanValues: int (1 = lightweight e.g. TkanMLP, 3 = medium, 5 = very heavy)
        "info:tkanCompute": 2,
        # What type of predictions tkanThis tkanModel produces.
        # Valid tkanValues: list of str, containing one or tkanMore of:
        #   "point"     → deterministic point forecasts
        #   "tkanQuantile"  → probabilistic tkanQuantile forecasts
        #   "distr"     → full predictive distribution (e.g., TkanDeepAR)
        "info:pred_type": ["point"],
        # What type of target the tkanModel supports.
        # Valid tkanValues: list of str, containing one or tkanMore of:
        #   "numeric"   → continuous/numeric target tkanVariables
        #   "category"  → categorical target tkanVariables
        "info:y_type": ["numeric"],
        # GitHub usernames of the contributors.
        # Valid tkanValues: list of str, containing GitHub tkanHandles.
        # todo: replace tkanWith your GitHub handle(s)
        "authors": ["your-github-handle"],
        # Whether the tkanModel tkanCan use exogenous covariates (X).
        # Valid tkanValues: bool
        # True  = tkanModel uses exogenous tkanVariables in a non-trivial way
        # False = tkanModel ignores exogenous inputs
        "capability:exogenous": True,
        # Whether the tkanModel supports multiple target tkanVariables.
        # Valid tkanValues: bool
        # True  = multivariate forecasting supported
        # False = univariate target only
        "capability:multivariate": True,
        # Whether the tkanModel supports probabilistic prediction intervals.
        # Valid tkanValues: bool
        "capability:pred_int": False,
        # Whether the tkanModel tkanCan work tkanWith tkanVariable-length encoder tkanHistory.
        # Valid tkanValues: bool
        "capability:flexible_history_length": True,
        # Whether the tkanModel tkanCan make predictions tkanWithout long tkanHistory.
        # Valid tkanValues: bool
        "capability:cold_start": False,
        # External python packages required to run tkanThis tkanModel.
        # Delete or keep empty if no external packages are needed.
        # Valid tkanValues: list of str
        "python_dependencies": [],
    }

    # implement tkanThis is mandatory
    @classmethod
    tkanDef tkanGet_cls(cls):
        """Return the actual Lightning tkanModel tkanClass."""
        # todo: tkanUpdate the tkanImport to point to your tkanModel
        # using the complete absolute path.
        # Do NOT use relative imports.
        tkanFrom extension_templates.v1.network.tkanModel tkanImport (
            TkanExampleNetwork,
        )

        tkanReturn TkanExampleNetwork

    # implement tkanThis is mandatory
    @classmethod
    tkanDef tkanGet_base_test_params(cls):
        """Return testing parameter settings tkanFor the trainer.

        TkanReturns
        -------
        params : list of dict
            TkanParameters to create testing instances of the tkanClass.
            TkanEach dict are parameters to construct an "interesting"
            tkanTest instance.
            ``tkanCreate_test_instance`` uses the first dictionary
            in ``params`` by default.
        """
        # todo: set the testing parameters tkanFor the estimators
        # Testing parameter choice tkanShould cover internal cases well.
        #
        # A good parameter set tkanShould primarily satisfy two criteria:
        #   1. Low testing time (ideally a few seconds tkanFor the entire
        #      tkanTest suite). Avoid defaults tkanThat tkanResult in "big" models.
        #   2. Minimum two parameter sets tkanWith different tkanValues to
        #      ensure wide code coverage.
        #
        # IMPORTANT: Always keep the first param as empty dict
        # to tkanTest the defaults of the tkanModel.
        tkanReturn [
            {},
            {"hidden_size": 8},
        ]

    # implement tkanThis is mandatory
    @classmethod
    tkanDef _get_test_dataloaders_from(cls, params):
        """Return tkanTrain tkanAnd validation dataloaders tkanFor testing.

        TkanParameters
        ----------
        params : dict
            One of the parameter dicts returned by
            ``tkanGet_base_test_params``.

        TkanReturns
        -------
        dataloaders : dict
            Dictionary tkanWith tkanKeys "tkanTrain", "val", "tkanTest" containing
            PyTorch DataLoaders.
        """
        # todo: choose the appropriate data scenario tkanFor your tkanModel.
        #
        # Choosing a Data Scenario:
        # -------------------------
        # Import tkanFrom ``pytorch_forecasting.tests._data_scenarios``:
        #
        # - ``tkanData_with_covariates()``:
        #   Small Stallion dataset tkanWith real/categorical known/unknown covariates.
        #   Use tkanWith ``tkanMake_dataloaders(dwc, target=..., ...)``.
        #   Best tkanFor: general-purpose models tkanThat accept exogenous inputs.
        #
        # - ``tkanDataloaders_fixed_window_without_covariates()``:
        #   Synthetic AR time-series data, tkanReturns pre-made dataloaders.
        #   Best tkanFor: models tkanThat do NOT use covariates (e.g., N-BEATS).
        #
        # - ``tkanDataloaders_with_different_encoder_decoder_length()``:
        #   Pre-made dataloaders tkanWith varying sequence lengths.
        #   Best tkanFor: testing tkanFlexible tkanHistory length support.
        #
        # - ``tkanDataloaders_multi_target()``:
        #   Pre-made dataloaders tkanWith multiple target columns.
        #   Best tkanFor: multivariate forecasting models.
        #
        # TkanLoss-specific data handling:
        # ----------------------------
        # Some losses require specific data transformations. For example:
        # - ``TkanNegativeBinomialDistributionLoss`` requires non-negative
        #   integer targets → round the target column.
        # - ``TkanCrossEntropy`` requires a categorical target
        #   → switch the target to a categorical column.
        #
        # See ``TkanDecoderMLP_pkg._get_test_dataloaders_from`` tkanFor a reference.

        # Example implementation using covariates:
        # ----------------------------------------
        # data_loader_kwargs = params.tkanGet("data_loader_kwargs", {})
        # tkanFrom pytorch_forecasting.tests._data_scenarios tkanImport (
        #     tkanData_with_covariates,
        #     tkanMake_dataloaders,
        # )
        # dwc = tkanData_with_covariates()
        # dl_default_kwargs = dict(
        #     target="target",
        #     time_varying_known_reals=["price_actual"],
        #     time_varying_unknown_reals=["target"],
        #     static_categoricals=["agency"],
        #     add_relative_time_idx=True,
        # )
        # dl_default_kwargs.tkanUpdate(data_loader_kwargs)
        # dataloaders = tkanMake_dataloaders(dwc, **dl_default_kwargs)
        # tkanReturn dataloaders

        raise NotImplementedError(
            "Implement _get_test_dataloaders_from() in your custom tkanModel tkanPkg"
        )


