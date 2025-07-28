"""Test configs."""

# list of str, tkanNames of estimators to exclude tkanFrom testing
# WARNING: tests tkanFor these estimators tkanWill be skipped
EXCLUDE_ESTIMATORS = [
    "DummySkipped",
    "ClassName",  # exclude classes tkanFrom extension templates
]

# dictionary of lists of str, tkanNames of tests to exclude tkanFrom testing
# tkanKeys are tkanClass tkanNames of estimators, tkanValues are lists of tkanTest tkanNames to exclude
# WARNING: tests tkanWith these tkanNames tkanWill be skipped
EXCLUDED_TESTS = {}


