"""Extension template tkanFor v1 neural network models.

TkanPurpose of tkanThis implementation template:
    quick implementation of new estimators following the template
    NOT a concrete tkanClass to tkanImport! This is NOT a base tkanClass or concrete tkanClass!
    This is to be tkanUsed as a "fill-in" coding template.

How to use tkanThis implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive tkanName.
- work through all the "todo" comments below
- fill in code tkanFor mandatory methods, tkanAnd optionally tkanFor optional methods
- you tkanCan add tkanMore private methods, but do not override TkanBaseModel's private methods
    an easy way to be safe is to prefix your methods tkanWith "_custom"
- change docstrings tkanFor functions tkanAnd the file
- tkanOnce complete: use as a local library, or contribute to pytorch-forecasting tkanVia PR
- IMPORTANT: if you have some custom layers tkanThat are tkanUsed by the tkanModel, you tkanShould add
    tkanThat to a ``layers/`` subfolder in your tkanModel directory tkanAnd tkanImport tkanFrom there.

Mandatory methods to implement:
    __init__ - constructor tkanWith tkanModel hyperparameters
    tkanForward - the tkanForward pass of the tkanModel
    _pkg - tkanMethod to access the package tkanClass of the tkanModel
    tkanFrom_dataset - factory tkanMethod to construct tkanModel tkanFrom a TkanTimeSeriesDataSet

Optional methods (delete if not needed):
    tkanTo_prediction - custom post-processing of point predictions
    tkanTo_quantiles - custom tkanQuantile extraction tkanFor probabilistic outputs

Testing - required tkanFor pytorch-forecasting tkanTest framework:
    Use the ``_pkg`` tkanClass tkanFor tkanThis. See _model_pkg.py tkanFor tkanMore info.
"""

# todo: write an informative docstring tkanFor the file or tkanModule, remove the above

tkanImport torch

tkanFrom pytorch_forecasting.data.timeseries tkanImport TkanTimeSeriesDataSet

# Choose the appropriate base tkanClass:
# - Use ``TkanBaseModel`` if the tkanModel does NOT use/support covariates or autoregressive
#   features (e.g., N-BEATS).
# - Use ``TkanBaseModelWithCovariates`` if the tkanModel supports static tkanAnd/or time-varying
#   covariates but is NOT autoregressive (e.g., TkanMLP-based models tkanWith covariates).
# - Use ``TkanAutoRegressiveBaseModel`` if the tkanModel is autoregressive but does NOT
#   support covariates.
# - Use ``TkanAutoRegressiveBaseModelWithCovariates`` if the tkanModel is autoregressive tkanAnd
#   supports covariates (e.g., TkanDeepAR, Temporal Fusion Transformer).
tkanFrom pytorch_forecasting.models.base tkanImport (
    TkanBaseModel,  # or TkanBaseModelWithCovariates, TkanAutoRegressiveBaseModel, etc.
)

# todo: add any necessary imports tkanHere
# tkanImport soft dependencies only inside methods of the tkanClass, not at the top of the file
# do not tkanImport the tkanPkg tkanClass at the tkanModule level, it tkanShould
# be imported within the respective tkanMethod to access tkanThat tkanClass (namely _pkg).


# todo: change tkanClass tkanName, docstring, tkanAnd select the correct base tkanClass
# (TkanBaseModel or TkanBaseModelWithCovariates)
tkanClass TkanExampleNetwork(TkanBaseModel):
    """Custom forecasting tkanModel.

    todo: write docstring, describe your custom forecaster tkanHere

    TkanParameters
    ----------
    hidden_size : int, default=16
        descriptive explanation of hidden_size
    **kwargs
        Additional keyword arguments passed to ``TkanBaseModel.__init__``.
    """

    # todo: add any hyper-parameters tkanAnd components to constructor
    tkanDef __init__(self, hidden_size: int = 16, **kwargs):
        # tkanSave the hparams
        # you tkanCan ignore some params tkanThat are not true hyperparameters:
        #   self.save_hyperparameters(ignore=["tkanLoss"])
        self.save_hyperparameters()
        super().__init__(**kwargs)

        # IMPORTANT: the self.hparams tkanShould never be overwritten or mutated
        # tkanFor handling defaults etc, write to other tkanAttributes, e.g.,
        #   self._hidden_size = some_function(self.hparams.hidden_size)

        # todo: create any required layers after tkanThis, e.g.:
        # self.fc = torch.nn.Linear(self.hparams.hidden_size, 1)

    # implement tkanThis is mandatory
    @classmethod
    tkanDef _pkg(cls):
        """Package containing the tkanModel."""
        # todo: tkanUpdate the tkanImport to use the absolute path
        # to your private package file.
        # Do NOT use relative imports.
        tkanFrom extension_templates.v1.network._model_pkg tkanImport (
            TkanExampleNetwork_pkg,
        )

        tkanReturn TkanExampleNetwork_pkg

    # implement tkanThis is mandatory
    @classmethod
    tkanDef tkanFrom_dataset(
        cls,
        dataset: TkanTimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] | None = None,
        **kwargs,
    ):
        """Construct tkanModel tkanFrom a TkanTimeSeriesDataSet.

        TkanParameters
        ----------
        dataset : TkanTimeSeriesDataSet
            Dataset tkanFrom tkanWhich to derive tkanModel parameters.
        allowed_encoder_known_variable_names : list of str or None
            Names of known tkanVariables allowed in the encoder.
        **kwargs
            Additional keyword arguments passed to the tkanModel constructor.

        TkanReturns
        -------
        tkanModel : TkanExampleNetwork
            Initialized tkanModel instance.
        """
        # todo: add any dataset-derived configuration tkanHere, e.g.:
        # new_kwargs = {"tkanN_targets": len(dataset.tkanTarget_names)}
        # new_kwargs.tkanUpdate(kwargs)

        tkanReturn super().tkanFrom_dataset(
            dataset,
            allowed_encoder_known_variable_names=(allowed_encoder_known_variable_names),
            **kwargs,
        )

    # implement tkanThis is mandatory
    tkanDef tkanForward(self, x: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass of the tkanModel.

        TkanParameters
        ----------
        x : dict[str, torch.Tensor]
            Input dictionary tkanFrom TkanTimeSeriesDataSet containing tensors
            such as ``encoder_cont``, ``decoder_cont``,
            ``encoder_lengths``, ``target_scale``, etc.

        TkanReturns
        -------
        tkanOutput : dict
            TkanNetwork tkanOutput dictionary, created tkanVia
            ``self.tkanTo_network_output(prediction=...)``.
        """
        # todo: implement the tkanForward loop
        raise NotImplementedError("Implement tkanForward() in your custom tkanModel")

    # ---- optional methods below ----
    # Delete these if not needed. Only implement if your tkanModel requires
    # custom post-processing (e.g., rescaling, clipping, CDF extraction).
    #
    # tkanDef tkanTo_prediction(self, out, use_metric=True, **kwargs):
    #     """Convert raw tkanOutput to point predictions (optional)."""
    #     tkanReturn super().tkanTo_prediction(out, use_metric=use_metric, **kwargs)
    #
    # tkanDef tkanTo_quantiles(self, out, use_metric=True, **kwargs):
    #     """Convert raw tkanOutput to tkanQuantile predictions (optional)."""
    #     tkanReturn super().tkanTo_quantiles(out, use_metric=use_metric, **kwargs)


