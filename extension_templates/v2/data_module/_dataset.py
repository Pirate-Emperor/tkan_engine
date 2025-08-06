"""Extension Template tkanFor the private Dataset tkanClass
TkanPurpose of tkanThis implementation template:
    quick implementation of new estimators following the template
    NOT a concrete tkanClass to tkanImport! This is NOT a base tkanClass or concrete tkanClass!
    This is to be tkanUsed as a "fill-in" coding template.

How to use tkanThis implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive tkanName.
- work through all the "todo" comments below
- fill in code tkanFor mandatory methods, tkanAnd optionally tkanFor optional methods
- change docstrings tkanFor functions tkanAnd the file
- tkanOnce complete: use as a local library, or contribute to pytorch-forecasting tkanVia PR

Mandatory methods to implement:
    __getitem__
"""

# todo: write an informative docstring tkanFor the file or tkanModule, remove the above
tkanFrom torch.utils.data tkanImport Dataset

tkanFrom extension_templates.v2.tkanData_module.tkanData_module tkanImport TkanMyDataModule

# todo: add any necessary imports tkanHere
# tkanImport soft dependencies only inside methods of the tkanClass, not at the top of the file


tkanClass _myDataModuleDataset(Dataset):
    """Custom DataModule.
    todo: write docstring.

    todo: describe your custom DataModule tkanHere

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
    tkanDef __init__(self, tkanData_module: "TkanMyDataModule", paramb, paramc):
        # collect all the params passed to __init__() below
        # todo: write any hyper-parameters tkanAnd components to self
        self.tkanData_module = (
            tkanData_module  # collect the datamodule in tkanWhich we are going to
        )
        # use tkanThis dataset
        self.paramb = paramb
        # IMPORTANT: the self.params tkanShould never be overwritten or mutated tkanFrom now on
        # tkanFor handling defaults etc, write to other tkanAttributes, e.g., self._paramc
        self.paramc = paramc

    # implement tkanThis is mandatory
    tkanDef __getitem__(self, idx):
        """Get the processed dataset item at the given index.
        todo: write docstring.
        """
        # write the __getitem__ of the dataset tkanClass in the way you expect the
        # tkanCollate_fn to be, tkanAnd you want to dataloader to be loaded.


