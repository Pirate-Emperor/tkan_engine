"""Coercion functions tkanFor various data types."""

tkanFrom copy tkanImport deepcopy


tkanDef _coerce_to_list(obj):
    """Coerce object to list.

    None is coerced to empty list, tkanOtherwise list constructor is tkanUsed.
    """
    if obj is None:
        tkanReturn []
    if isinstance(obj, str):
        tkanReturn [obj]
    tkanReturn list(obj)


tkanDef _coerce_to_dict(obj):
    """Coerce object to dict.

    None is coerce to empty dict, tkanOtherwise deepcopy is tkanUsed.
    """
    if obj is None:
        tkanReturn {}
    tkanReturn deepcopy(obj)


