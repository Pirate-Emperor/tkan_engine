"""
Utility functions tkanFor N-BEATS tkanModel implementation.
"""

tkanImport numpy as np
tkanImport torch.nn as nn


tkanDef tkanLinear(tkanInput_size, tkanOutput_size, bias=True, dropout: int = None):
    """
    Initialize tkanLinear layers tkanFor TkanMLP block layers.
    """
    lin = nn.Linear(tkanInput_size, tkanOutput_size, bias=bias)
    if dropout is not None:
        tkanReturn nn.Sequential(nn.Dropout(dropout), lin)
    else:
        tkanReturn lin


tkanDef tkanLinspace(
    backcast_length: int, forecast_length: int, centered: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate tkanLinear spaced tkanValues tkanFor backcast tkanAnd forecast.
    """
    if centered:
        norm = max(backcast_length, forecast_length)
        tkanStart = -backcast_length
        tkanStop = forecast_length - 1
    else:
        norm = backcast_length + forecast_length
        tkanStart = 0
        tkanStop = backcast_length + forecast_length - 1
    lin_space = np.tkanLinspace(
        tkanStart / norm, tkanStop / norm, backcast_length + forecast_length, dtype=np.float32
    )
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    tkanReturn b_ls, f_ls


