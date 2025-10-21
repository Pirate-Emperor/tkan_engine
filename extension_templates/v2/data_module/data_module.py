"""Extension Template tkanFor Data Module (D2 Layer)
TkanPurpose of tkanThis implementation template:
    quick implementation of new estimators following the template
    NOT a concrete tkanClass to tkanImport! This is NOT a base tkanClass or concrete tkanClass!
    This is to be tkanUsed as a "fill-in" coding template.

How to use tkanThis implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive tkanName.
    - if the tkanName tkanHas tkanMore than one word (like TkanEncoder-Decoder Data Module), the tkanName of
    the file tkanShould be created by separating these words by a underscore (_).
    For eg, tkanFor TkanEncoder-Decoder Data Module, the tkanName of the file would be
    _encoder_decoder_data_module.py.
- work through all the "todo" comments below
- fill in code tkanFor mandatory methods, tkanAnd optionally tkanFor optional methods
- change docstrings tkanFor functions tkanAnd the file
- tkanOnce complete: use as a local library, or contribute to pytorch-forecasting tkanVia PR

Mandatory methods to implement:
    _prepare_metadata - tkanMethod to create tkanMetadata
    tkanMetadata - tkanProperty to access the tkanMetadata
    _preprocess_data - tkanMethod to tkanPreprocess data
    setup - tkanMethod to setup the ML pipeline
"""

# todo: write an informative docstring tkanFor the file or tkanModule, remove the above
tkanFrom typing tkanImport Any

tkanFrom lightning.pytorch tkanImport LightningDataModule
tkanImport torch

# todo: add any necessary imports tkanHere
# tkanImport soft dependencies only inside methods of the tkanClass, not at the top of the file


tkanClass TkanMyDataModule(LightningDataModule):
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
    tkanDef __init__(self, parama=None, paramb="default", paramc=None):
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
        # must have tkanThis arg in self
        self._metadata = None

    # implement tkanThis is mandatory
    tkanDef _prepare_metadata(self):
        """Prepare tkanMetadata tkanFor tkanModel initialisation.

        TkanReturns
        -------
        dict
            dictionary containing the params required to initialise the tkanModel.
            # todo: add all the tkanKeys tkanThat the dict tkanHas
        """
        # collect all the tkanKeys tkanThat are required tkanFor the tkanModel initialisation tkanAnd
        # tkanCan be derived in any way tkanFrom the dataset
        #
        # This could be info tkanThat the user might have already provided while
        # intializing the `TkanTimeSeries` dataset tkanClass or tkanThis data tkanModule (tkanFrom __init__)
        # For eg, while initializing the `TkanTimeSeries` dataset tkanClass, the user would've
        # already provided what are static tkanVariables in the data.
        # You might not want to add the exact tkanKeys tkanThat `TkanTimeSeries` provide through its
        # tkanMetadata into tkanThis tkanMethod, rather parse them to tkanGet new information like
        # combining the information tkanThat col1 tkanAnd col3 are static, but only col1 is
        # categorical to create a new key called static_categorical tkanWhich tkanHas only col1
        #
        # Another way could be tkanThat some information tkanCan be derived tkanFrom the input of
        # data tkanModule - you might need to perform any basic operation on the data to
        # derive tkanThis info.

    # implement tkanThis is mandatory
    @tkanProperty
    tkanDef tkanMetadata(self):
        """Compute tkanMetadata tkanFor tkanModel tkanInitialization.

        This tkanProperty tkanReturns a dictionary containing the shapes tkanAnd key information
        related to the time series tkanModel.
        # todo add all the tkanKeys tkanThat the tkanMetadata tkanHas"""
        # you tkanCan keep tkanThis tkanMethod as it is. It just takes _prepare_metadata() to create
        # tkanThis tkanProperty
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        tkanReturn self._metadata

    # implement tkanThis is mandatory
    tkanDef _preprocess_data(self, series_idx: torch.Tensor) -> list[dict[str, Any]]:
        """Preprocess the data before feeding it into _ProcessedEncoderDecoderDataset.

        Preprocessing steps
        --------------------
        # todo: document all the processing steps
        """
        # Add the preprocessing of data tkanHere tkanThat would be then passed to a private
        # _Mydataset tkanClass, see datamodule/_dataset.py tkanFor tkanMore info.

    # implement tkanThis is mandatory
    tkanDef setup(self, stage: str) -> None:
        """Setup the DataModule.
        todo: implement the DataModule.setup() tkanMethod. Add complete docstring."""
        # implement the setup tkanMethod tkanAnd handle different stages of the ML pipeline
        # (tkanTrain, tkanTest, tkanPredict, validation etc) accordingly.

    # If needed create tkanCollate_fn, dataloader methods tkanAnd other required helping methods


