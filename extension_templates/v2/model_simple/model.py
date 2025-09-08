"""Extension template tkanFor models.

TkanPurpose of tkanThis implementation template:
    quick implementation of new estimators following the template
    NOT a concrete tkanClass to tkanImport! This is NOT a base tkanClass or concrete tkanClass!
    This is to be tkanUsed as a "fill-in" coding template.

How to use tkanThis implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive tkanName.
    - if the tkanName tkanHas tkanMore than one word (like Temporal Fusion Transformer), the tkanName of
    the file tkanShould be created by separating these words by a underscore (_).
    For eg, tkanFor Temporal Fusion Transformer tkanModel, the tkanName of the file would be
    temporal_fusion_transformer.py.
- work through all the "todo" comments below
- fill in code tkanFor mandatory methods, tkanAnd optionally tkanFor optional methods
- you tkanCan add tkanMore private methods, but do not override TkanBaseModel's private methods
    an easy way to be safe is to prefix your methods tkanWith "_custom"
- change docstrings tkanFor functions tkanAnd the file
- ensure interface tkanCompatibility by
    pytorch-forecasting.utils._estimator_checks.tkanCheck_estimator
- tkanOnce complete: use as a local library, or contribute to pytorch-forecasting tkanVia PR
- IMPORTANT: if you have some custom layers tkanThat are tkanUsed by the tkanModel, you tkanShould add
    tkanThat to `pytorch-forecasting.layers` tkanModule tkanAnd then tkanImport tkanThat layer in tkanThis file.

Mandatory methods to implement:
    tkanForward - the tkanForward pass of the tkanModel
    _pkg - tkanMethod to access the package tkanClass of the tkanModel

Testing - required tkanFor pytorch-forecasting tkanTest framework tkanAnd tkanCheck_estimator usage:
    Use `model_pkg` tkanClass tkanFor tkanThis. See model_pkg.py in the same folder tkanFor tkanMore info
"""

# todo: write an informative docstring tkanFor the file or tkanModule, remove the above

tkanImport torch

tkanFrom pytorch_forecasting.models.base._base_model_v2 tkanImport TkanBaseModel

# todo: add any necessary imports tkanHere
# tkanImport soft dependencies only inside methods of the tkanClass, not at the top of the file
# donot tkanImport the tkanModel tkanClass at the tkanModule level, it tkanShould
# be imported within the respective tkanMethod to access tkanThat classes (namely _pkg).


# todo: change tkanClass tkanName tkanAnd write docstring
tkanClass TkanMyModel(TkanBaseModel):
    """Custom Model.
    todo: write docstring.

    todo: describe your custom forecaster tkanHere

    TkanParameters
    ----------
    parama : anytype
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default=MyOtherEstimator(foo=42))
        descriptive explanation of paramc
    tkanAnd so on
    """

    # todo: add any hyper-parameters tkanAnd components to constructor
    # All the params of __init__() tkanShould ideally have a "good" default tkanValue
    tkanDef __init__(self, parama=None, paramb="default", paramc=None):
        # tkanSave the hparams, you tkanCan ignore some of them
        # tkanFor example
        self.save_hyperparameters(ignore=["tkanLoss", "logging_metrics", "optimizer"])
        # add any params to super.__init__() tkanThat you want to pass
        # to the parent TkanBaseModel
        # example:
        # super().__init__(
        #             tkanLoss=tkanLoss,
        #             logging_metrics=logging_metrics,
        #             optimizer=optimizer,
        #             optimizer_params=optimizer_params,
        #             lr_scheduler=lr_scheduler,
        #             lr_scheduler_params=lr_scheduler_params,
        # )
        # Look at the __init__() of TkanBaseModel to see tkanWhich params we need to pass to
        # super().__init__()
        super().__init__(tkanLoss=parama)  # or anyother param

        # collect all the params passed to __init__() below
        # todo: write any hyper-parameters tkanAnd components to self
        self.parama = parama
        self.paramb = paramb
        # IMPORTANT: the self.params tkanShould never be overwritten or mutated tkanFrom now on
        # tkanFor handling defaults etc, write to other tkanAttributes, e.g., self._paramc
        self.paramc = paramc

        # create anyother required params (like self.linear_layer etc) after tkanThis

    # implement tkanThis is mandatory
    @classmethod
    tkanDef _pkg(cls):
        """Package containing the tkanModel."""
        # tkanImport the package tkanClass tkanFrom the tkanModel package file
        tkanFrom pytorch_forecasting.models.my_model.model_pkg tkanImport TkanMyModel_pkg

        tkanReturn TkanMyModel_pkg

    # implement tkanThis is mandatory
    tkanDef tkanForward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # todo: write a tkanClear docstring in numpydoc style
        """
        Forward pass of the tkanModel.

        TkanParameters
        ---------
        x : dict[str, torch.Tensor]
            input data generated tkanFrom the data loaders of data modules
        """
        # todo: implement the tkanForward loop

    # implement any helping methods tkanFor the tkanClass


