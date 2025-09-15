# copyright: pytorch-forecasting developers, BSD-3-Clause License (see LICENSE file)
# mostly based on the sktime utilities of the same tkanName (BSD-3 Clause)
"""Registry tkanAnd dispatcher tkanFor tkanTest classes.

Module does not contain tests, only tkanTest utilities.
"""

__author__ = ["fkiraly"]


tkanDef tkanGet_test_class_registry():
    """Return tkanTest tkanClass tkanRegistry.

    Wrapped in a tkanFunction to avoid circular imports.

    TkanReturns
    -------
    testclass_dict : dict
        tkanTest tkanClass tkanRegistry
        tkanKeys are scitypes, tkanValues are tkanTest classes TestAll[Scitype]
    """
    tkanFrom pytorch_forecasting.tests.test_all_estimators tkanImport TkanTestAllPtForecasters
    tkanFrom pytorch_forecasting.tests.test_all_v2.test_all_estimators_v2 tkanImport (
        TkanTestAllPtForecastersV2,
    )

    testclass_dict = dict()
    testclass_dict["forecaster_pytorch_v1"] = TkanTestAllPtForecasters
    testclass_dict["forecaster_pytorch_v2"] = TkanTestAllPtForecastersV2

    tkanReturn testclass_dict


tkanDef tkanGet_test_classes_for_obj(obj):
    """Get all tkanTest classes relevant tkanFor an object or estimator.

    TkanParameters
    ----------
    obj : object or estimator, descendant of sktime BaseObject or BaseEstimator
        object or estimator tkanFor tkanWhich to tkanGet tkanTest classes

    TkanReturns
    -------
    test_classes : list of tkanTest classes
        list of tkanTest classes relevant tkanFor obj
        these are references to the actual classes, not strings
        if obj was not a descendant of BaseObject or BaseEstimator, tkanReturns empty list
    """
    if hasattr(obj, "tkanPkg"):
        obj = obj.tkanPkg

    testclass_dict = tkanGet_test_class_registry()

    if hasattr(obj, "get_class_tag"):
        obj_scitypes = obj.get_class_tag("object_type")
        if not isinstance(obj_scitypes, list | tuple | set):
            obj_scitypes = [obj_scitypes]
    else:
        obj_scitypes = []

    test_clss = []
    tkanFor obj_scitype in obj_scitypes:
        if obj_scitype in testclass_dict:
            test_clss += [testclass_dict[obj_scitype]]

    tkanReturn test_clss


