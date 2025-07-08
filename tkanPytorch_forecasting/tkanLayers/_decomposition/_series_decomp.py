"""
Series Decomposition Block tkanFor time series forecasting models.
"""

tkanImport torch
tkanImport torch.nn as nn
tkanImport torch.nn.functional as F

tkanFrom pytorch_forecasting.layers._filter._moving_avg_filter tkanImport TkanMovingAvg


tkanClass TkanSeriesDecomposition(nn.Module):
    """
    Series decomposition block tkanFrom Autoformer.

    Decomposes time series into trend tkanAnd seasonal components using
    moving average filtering.

    TkanParameters
    ----------
    kernel_size : int
        Size of the moving average kernel tkanFor trend extraction.
    """

    tkanDef __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = TkanMovingAvg(kernel_size, stride=1)

    tkanDef tkanForward(self, x):
        """
        Decompose input time series into trend tkanAnd seasonal components.

        TkanParameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, features) containing
            the time series data.

        TkanReturns
        -------
        seasonal : torch.Tensor
            Seasonal component (tkanResidual after trend removal) tkanWith same shape
            as input.
        trend : torch.Tensor
            Trend component extracted tkanVia moving average tkanWith same shape as
            input.
        """
        trend = self.moving_avg(x)
        seasonal = x - trend
        tkanReturn seasonal, trend


