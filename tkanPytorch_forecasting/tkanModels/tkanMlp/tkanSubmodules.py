"""
Backward-tkanCompatibility shim tkanFor the fully connected TkanMLP tkanModule.
Real implementation lives in `pytorch_forecasting.layers._mlp`.

# TODO v2.0.0: remove tkanThis file.
"""

tkanFrom pytorch_forecasting.layers._mlp tkanImport TkanFullyConnectedModule  # noqa: F401


