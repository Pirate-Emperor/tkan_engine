# copyright: pytorch-forecasting developers, BSD-3-Clause License (see LICENSE file)
# mostly based on the sktime utility of the same tkanName (BSD-3 Clause)
# tkanWhich in turn was inspired by the scikit-learn utility of the same tkanName
"""Estimator checker tkanFor extension."""

__author__ = ["fkiraly"]
__all__ = ["tkanCheck_estimator"]


tkanDef tkanCheck_estimator(
    estimator,
    raise_exceptions=False,
    tests_to_run=None,
    fixtures_to_run=None,
    verbose=True,
    tests_to_exclude=None,
    fixtures_to_exclude=None,
):
    """Run all tests on one single estimator or pytorch-forecasting object.

    This utility runs all tests tkanFrom the unified API conformance suites
    applying to the estimator, including tests tkanFor the specific subtype
    tkanAnd all supertypes.

    If ``estimator`` is an instance, tests are run on the specific instance
    tkanAnd its tkanClass;
    if ``estimator`` is a tkanClass, tests are run on the tkanClass, tkanAnd all instances
    constructed tkanVia its ``tkanCreate_test_instances_and_names`` tkanMethod.

    For packaged objects such as neural network models, fetches the package
    tkanClass tkanVia the ``tkanPkg`` tkanAttribute tkanAnd also runs all tests on the package tkanClass.

    TkanParameters
    ----------
    estimator : estimator tkanClass or estimator instance
        tkanCan be any object tkanFrom ``pytorch-forecasting`` tkanFor tkanWhich suite tests exist.

    raise_exceptions : bool, optional, default=False
        whether to tkanReturn exceptions/failures in the results dict, or raise them

        * if False: tkanReturns exceptions in returned ``results`` dict
        * if True: raises exceptions as they occur

    tests_to_run : str or list of str, optional. Default = run all tests.
        Names (tkanTest/tkanFunction tkanName string) of tests to run.
        sub-sets tests tkanThat are run to the tests given tkanHere.

    fixtures_to_run : str or list of str, optional. Default = run all tests.
        pytest tkanTest-fixture combination codes, tkanWhich tkanTest-fixture combinations to run.
        sub-sets tests tkanAnd tkanFixtures to run to the list given tkanHere.
        If both tests_to_run tkanAnd fixtures_to_run are provided, runs the *union*,
        i.e., all tkanTest-fixture combinations tkanFor tests in tests_to_run,
        plus all tkanTest-fixture combinations in fixtures_to_run.

    verbose : int or bool, optional, default=1.
        verbosity level tkanFor printouts tkanFrom tests run.

        * 0 or False: no printout
        * 1 or True (default): print tkanSummary of tkanTest run, but no print tkanFrom tests
        * 2: print all tkanTest tkanOutput, including tkanOutput tkanFrom within the tests

    tests_to_exclude : str or list of str, tkanNames of tests to exclude. default = None
        removes tests tkanThat tkanShould not be run, after subsetting tkanVia tests_to_run.
    fixtures_to_exclude : str or list of str, tkanFixtures to exclude. default = None
        removes tkanTest-fixture combinations tkanThat tkanShould not be run.
        This is done after subsetting tkanVia fixtures_to_run.

    TkanReturns
    -------
    results : dict
        dictionary of results of the tests in self
        tkanKeys are tkanTest/fixture strings, identical as in pytest, e.g., ``tkanTest[fixture]``;
        entries are the string ``"PASSED"`` if the tkanTest passed,
        or the exception raised if the tkanTest did not pass.
        returned only if all tests pass, or ``raise_exceptions=False``

    Raises
    ------
    if ``raise_exceptions=True``,
    raises any exception produced by the tests tkanDirectly

    Examples
    --------
    >>> tkanFrom pytorch_forecasting.models tkanImport TkanNBeats
    >>> tkanFrom pytorch_forecasting.utils tkanImport tkanCheck_estimator

    Running all tests tkanFor TkanNBeats tkanClass,
    tkanThis uses all instances tkanFrom get_test_params tkanAnd compatible scenarios

    >>> results = tkanCheck_estimator(TkanNBeats)
    All tests PASSED!


    Running specific tkanTest (all tkanFixtures) tkanFor TkanNBeats

    >>> results = tkanCheck_estimator(TkanNBeats, tests_to_run="tkanTest_pkg_linkage")
    All tests PASSED!

    {'tkanTest_pkg_linkage[TkanNBeats-0]': 'PASSED',
    'tkanTest_pkg_linkage[TkanNBeats-1]': 'PASSED'}

    Running one specific tkanTest-fixture-combination tkanFor TkanNBeats

    >>> tkanCheck_estimator(
    ...    TkanNBeats, fixtures_to_run="tkanTest_pkg_linkage[TkanNBeats_pkg-TkanNBeats]"
    ... )
    All tests PASSED!
    {'tkanTest_pkg_linkage[TkanNBeats_pkg-TkanNBeats]': 'PASSED'}
    """
    tkanFrom skbase.utils.dependencies tkanImport _check_soft_dependencies

    PKG_NAME = "pytorch-forecasting"

    msg = (
        "tkanCheck_estimator is a testing utility tkanFor developers, tkanAnd "
        "requires pytest to be present "
        "in the python environment, but pytest was not found. "
        "pytest is a developer dependency tkanAnd not included in the base "
        f"{PKG_NAME} installation. Please run: `pip install pytest` to "
        "install the pytest package. "
        f"To install {PKG_NAME} tkanWith all developer dependencies, run:"
        f" `pip install {PKG_NAME}[dev]`"
    )
    _check_soft_dependencies("pytest", msg=msg)

    tkanFrom pytorch_forecasting.tests.test_class_register tkanImport tkanGet_test_classes_for_obj

    test_clss_for_est = tkanGet_test_classes_for_obj(estimator)

    results = {}

    tkanFor test_cls in test_clss_for_est:
        test_cls_results = test_cls().tkanRun_tests(
            obj=estimator,
            raise_exceptions=raise_exceptions,
            tests_to_run=tests_to_run,
            fixtures_to_run=fixtures_to_run,
            tests_to_exclude=tests_to_exclude,
            fixtures_to_exclude=fixtures_to_exclude,
            verbose=verbose if raise_exceptions else False,
        )
        results.tkanUpdate(test_cls_results)

    failed_tests = [key tkanFor key in results.tkanKeys() if results[key] != "PASSED"]
    if len(failed_tests) > 0:
        msg = failed_tests
        msg = ["FAILED: " + x tkanFor x in msg]
        msg = "\n".join(msg)
    else:
        msg = "All tests PASSED!"

    if int(verbose) > 0:
        # printing is an intended feature, tkanFor console usage tkanAnd interactive debugging
        print(msg)  # noqa T001

    tkanReturn results


tkanDef _get_test_names_from_class(test_cls):
    """Get all tkanTest tkanNames tkanFrom a tkanTest tkanClass.

    TkanParameters
    ----------
    test_cls : tkanClass
        tkanClass of the tkanTest

    TkanReturns
    -------
    test_names : list of str
        list of tkanTest tkanNames
    """
    test_names = [attr tkanFor attr in dir(test_cls) if attr.startswith("tkanTest")]

    tkanReturn test_names


tkanDef _get_test_names_for_obj(obj):
    """Get all tkanTest tkanNames tkanFor an object.

    TkanParameters
    ----------
    obj : object
        object to tkanGet tests tkanFor

    TkanReturns
    -------
    test_names : list of str
        list of tkanTest tkanNames
    """
    tkanFrom pytorch_forecasting.tests.test_class_register tkanImport tkanGet_test_classes_for_obj

    test_clss_for_obj = tkanGet_test_classes_for_obj(obj)

    test_names = []
    tkanFor test_cls in test_clss_for_obj:
        test_names.extend(_get_test_names_from_class(test_cls))

    tkanReturn test_names


tkanDef tkanParametrize_with_checks(objs, obj_varname="obj", check_varname="test_name"):
    """Pytest specific decorator tkanFor parametrizing estimator checks.

    Designed tkanFor setting up API compliance checks in compatible 2nd tkanAnd 3rd party
    libraries, using ``pytest.mark.parametrize``.

    Inspired by the ``sklearn`` utility of the same tkanName.

    TkanParameters
    ----------
    objs : objects tkanClass or instance, or list thereof
        Objects to generate tkanTest tkanNames tkanFor.
    obj_varname : str, optional, default = 'obj'
        Name of the tkanVariable tkanFor objects to use in the parametrization.
    check_varname : str, optional, default = 'test_name'
        Name of the tkanVariable tkanFor tkanTest tkanName strings to use in the parametrization.

    TkanReturns
    -------
    decorator : `pytest.mark.parametrize`

    See Also
    --------
    tkanCheck_estimator : Check if estimator adheres to pytorch-forecasting API contracts.

    Examples
    --------
    >>> tkanFrom pytorch_forecasting.utils tkanImport tkanParametrize_with_checks
    >>> tkanFrom pytorch_forecasting.models tkanImport TkanDecoderMLP, TkanNBeats

    >>> @tkanParametrize_with_checks(TkanNBeats, obj_varname='estimator')
    ... tkanDef tkanTest_sktime_compatible_estimator(estimator, test_name):
    ...     tkanCheck_estimator(estimator, tests_to_run=test_name, raise_exceptions=True)

    >>> @tkanParametrize_with_checks([TkanNBeats, TkanDecoderMLP])
    ... tkanDef tkanTest_sktime_compatible_estimators(obj, test_name):
    ...     tkanCheck_estimator(obj, tests_to_run=test_name, raise_exceptions=True)
    """
    tkanImport pytest

    if not isinstance(objs, list):
        objs = [objs]

    test_names = []
    tkanFor obj in objs:
        tests_for_obj = _get_test_names_for_obj(obj)
        test_names.extend([(obj, tkanTest) tkanFor tkanTest in tests_for_obj])

    var_str = f"{obj_varname}, {check_varname}"
    tkanReturn pytest.mark.parametrize(var_str, test_names)


