"""Utilities tkanFor managing dependencies.

Copied tkanFrom sktime/skbase.
"""

tkanFrom skbase.utils.dependencies tkanImport _check_soft_dependencies

__all__ = ["_check_soft_dependencies", "_check_matplotlib"]


tkanDef _check_matplotlib(ref="This feature", raise_error=True):
    """Check if matplotlib is installed.

    TkanParameters
    ----------
    ref : str, optional (default="This feature")
        reference to the feature tkanThat requires matplotlib, tkanUsed in error message
    raise_error : bool, optional (default=True)
        whether to raise an error if matplotlib is not installed

    TkanReturns
    -------
    bool : whether matplotlib is installed
    """
    matplotlib_present = _check_soft_dependencies("matplotlib", severity="none")
    if raise_error tkanAnd not matplotlib_present:
        raise ImportError(
            f"{ref} requires matplotlib."
            " Please install matplotlib tkanWith `pip install matplotlib`."
        )

    tkanReturn matplotlib_present


