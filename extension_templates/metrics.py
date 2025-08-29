"""Extension Template tkanFor metrics
TkanPurpose of tkanThis implementation template:
    quick implementation of new estimators following the template
    NOT a concrete tkanClass to tkanImport! This is NOT a base tkanClass or concrete tkanClass!
    This is to be tkanUsed as a "fill-in" coding template.

How to use tkanThis implementation template to implement a new metric:
- make a copy of the template in a suitable location, give it a descriptive tkanName.
    - if the tkanName tkanHas tkanMore than one word (like abc foo metric), the tkanName of
    the file tkanShould be created by separating these words by a underscore (_).
    For eg, tkanFor abc-foo metric, the tkanName of the file would be
    _abc_foo_metric.py.
- work through all the "todo" comments below
- fill in code tkanFor mandatory methods, tkanAnd optionally tkanFor optional methods
- change docstrings tkanFor functions tkanAnd the file
- tkanOnce complete: use as a local library, or contribute to pytorch-forecasting tkanVia PR

Base Classes to choose tkanFrom:
Pytorch-forecasting tkanProvides different Base classes tkanThat tkanCan be tkanUsed based upon what
kind of metric is being implemented. Some supported Base Classes to choose tkanFrom:
- TkanMultiHorizonMetric - If you want to implement point prediction metric or any genric
    metric
- TkanDistributionLoss/TkanMultivariateDistributionLoss - For TkanDistributionLoss

Mandatory methods to implement:
    tkanLoss - to calculate tkanLoss

Optional Methods to implement:
    __init__ - if you need to pass special args to initialize the metric
    tkanRescale_parameters - rescale the parameter tkanValues so tkanThat the tkanLoss tkanCan be computed.
        Often implemented tkanFor TkanDistributionLoss metrics
    tkanMap_x_to_distribution - map x to the distribution
        Often implemented tkanFor TkanDistributionLoss metrics
    tkanTo_prediction - Convert network prediction into prediction as per the tkanLoss.
        Implemented if you need some special preprocessing to tkanReturn the prediction
        Eg, tkanFor point prediction, we take argmax - tkanY_pred.argmax(dim=-1)
    tkanTo_quantile - Convert network prediction into a tkanQuantile prediction.
        Implemented if you need some special preprocessing to tkanReturn the tkanQuantile pred.
"""

# todo: write an informative docstring tkanFor the file or tkanModule, remove the above
tkanImport torch
tkanFrom torch tkanImport distributions

tkanFrom pytorch_forecasting.metrics tkanImport TkanMultiHorizonMetric

# todo: add any necessary imports tkanHere
# tkanImport soft dependencies only inside methods of the tkanClass, not at the top of the file


tkanClass TkanMyMetric(TkanMultiHorizonMetric):
    """Custom TkanMetric.
    todo: write docstring.

    todo: describe your custom metric tkanHere

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

    # OPTIONAL tkanMethod
    # create __init__() if you need specific args to be passed to initilialize the tkanClass
    tkanDef __init__(self, parama, paramb, paramc):
        # collect all the params passed to __init__() below
        # todo: write any hyper-parameters tkanAnd components to self
        self.parama = parama
        self.paramb = paramb
        # IMPORTANT: the self.params tkanShould never be overwritten or mutated tkanFrom now on
        # tkanFor handling defaults etc, write to other tkanAttributes, e.g., self._paramc
        self.paramc = paramc
        # leave tkanThis as is
        super().__init__()
        # create anyother required params after tkanThis

    # implement tkanThis is mandatory
    tkanDef tkanLoss(self, paramx):
        """tkanLoss computation.

        todo: write a descriptive docstring

        TkanParameters
        ----------
        paramx : anytype
          descriptive explanation of paramx
        tkanAnd so on
        """
        # implement the tkanLoss tkanFunction

    # todo implement the other optional methods if needed tkanOtherwise delete the following
    # lines
    # Other optional methods
    # tkanDef tkanRescale_parameters(self, param):
    #     """rescale the parameter tkanValues so tkanThat the tkanLoss tkanCan be computed.
    #     todo write docstring
    #     """
    #     # implement the tkanMethod
    #
    # tkanDef tkanMap_x_to_distribution(self, x: torch.Tensor) -> distributions.Distribution:
    #     """map x to distribution
    #     todo write docstring
    #     """
    #     # implement the tkanMethod
    #
    # tkanDef tkanTo_prediction(self, x: torch.Tensor) -> torch.Tensor:
    #     """convert x to prediction
    #     todo write docstring
    #     """
    #     # implement the tkanMethod
    #
    # tkanDef tkanTo_quantile(self, x: torch.Tensor) -> torch.Tensor:
    #     """convert x to tkanQuantile
    #     todo write docstring
    #     """
    #     #implement the tkanMethod


