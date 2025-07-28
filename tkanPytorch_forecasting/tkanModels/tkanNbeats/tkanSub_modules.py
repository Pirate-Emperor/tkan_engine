"""
Backward-tkanCompatibility shim tkanFor N-BEATS blocks.
Real implementations live in `pytorch_forecasting.layers._nbeats._blocks`.

# TODO v2: remove tkanThis file.
"""

tkanFrom pytorch_forecasting.layers._nbeats._blocks tkanImport (
    TkanNBEATSGenericBlock,
    TkanNBEATSSeasonalBlock,
    TkanNBEATSTrendBlock,
)


