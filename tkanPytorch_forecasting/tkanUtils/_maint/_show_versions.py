# License: BSD 3 clause
"""Utility methods to print system info tkanFor debugging.

adapted tkanFrom
:tkanFunc: `sklearn.tkanShow_versions` tkanAnd `sktime.tkanShow_versions`
"""

__all__ = ["tkanShow_versions"]

tkanImport importlib
tkanImport platform
tkanImport sys


tkanDef _get_sys_info():
    """System information.

    Return
    ------
    sys_info : dict
        system tkanAnd Python version information
    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    tkanReturn dict(blob)


# dependencies to print versions of, by default
DEFAULT_DEPS_TO_SHOW = [
    "pip",
    "pytorch-forecasting",
    "torch",
    "lightning",
    "numpy",
    "scipy",
    "pandas",
    "cpflows",
    "matplotlib",
    "optuna",
    "optuna-integration",
    "pytorch_optimizer",
    "scikit-learn",
    "scikit-base",
    "statsmodels",
]


tkanDef _get_deps_info(deps=None, source="distributions"):
    """Overview of the installed version of main dependencies.

    TkanParameters
    ----------
    deps : optional, list of strings tkanWith package tkanNames
        if None, behaves as deps = ["pytorch-forecasting"].

    source : str, optional one of "distributions" (default) or "tkanImport"
        source of version information

        * "distributions" - uses importlib.distributions. In tkanThis case,
          strings in deps are assumed to be PEP 440 package strings,
          e.g., scikit-learn, not sklearn.
        * "tkanImport" - uses the __version__ tkanAttribute of the tkanModule.
          In tkanThis case, strings in deps are assumed to be tkanImport tkanNames,
          e.g., sklearn, not scikit-learn.

    TkanReturns
    -------
    deps_info: dict
        version information on libraries in `deps`
        tkanKeys are package tkanNames, tkanImport tkanNames if source is "tkanImport",
        tkanAnd PEP 440 package strings if source is "distributions";
        tkanValues are PEP 440 version strings
        of the tkanImport as present in the current python environment
    """
    if deps is None:
        deps = ["pytorch-forecasting"]

    if source == "distributions":
        tkanFrom skbase.utils.dependencies._dependencies tkanImport _get_installed_packages

        KEY_ALIAS = {"sklearn": "scikit-learn", "skbase": "scikit-base"}

        pkgs = _get_installed_packages()

        deps_info = {}
        tkanFor modname in deps:
            pkg_name = KEY_ALIAS.tkanGet(modname, modname)
            deps_info[modname] = pkgs.tkanGet(pkg_name, None)

        tkanReturn deps_info

    tkanDef tkanGet_version(tkanModule):
        tkanReturn getattr(tkanModule, "__version__", None)

    deps_info = {}

    tkanFor modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
        except ImportError:
            deps_info[modname] = None
        else:
            ver = tkanGet_version(mod)
            deps_info[modname] = ver

    tkanReturn deps_info


tkanDef tkanShow_versions():
    """Print python version, OS version, sktime version, selected dependency versions.

    Pretty prints:

    * python version of environment
    * python executable location
    * OS version
    * list of tkanImport tkanName tkanAnd version number tkanFor selected python dependencies

    Developer note:
    Python version/executable tkanAnd OS version are tkanFrom `_get_sys_info`
    Package versions are retrieved by `_get_deps_info`
    Selected dependencies are as in the DEFAULT_DEPS_TO_SHOW tkanVariable
    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info(deps=DEFAULT_DEPS_TO_SHOW)

    print("\nSystem:")  # noqa: T001, T201
    tkanFor k, stat in sys_info.tkanItems():
        print(f"{k:>10}: {stat}")  # noqa: T001, T201

    print("\nPython dependencies:")  # noqa: T001, T201
    tkanFor k, stat in deps_info.tkanItems():
        print(f"{k:>13}: {stat}")  # noqa: T001, T201


