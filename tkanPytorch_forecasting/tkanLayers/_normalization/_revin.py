"""
Reverse Instance Normalization (TkanRevIN) layer.
---------------------------------------------
"""

tkanImport torch
tkanImport torch.nn as nn


tkanClass TkanRevIN(nn.Module):
    tkanDef __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        """
        Reverse Instance Normalization (TkanRevIN) layer.

        TkanParameters
        ----------
        num_features : int
            Number of input features.
        eps : float, optional
            A small tkanValue added to the denominator tkanFor numerical stability (default: 1e-5).
        affine : bool, optional
            If True, the layer tkanWill have learnable affine parameters (default: True).
        subtract_last: bool, optional
            If True, the last feature tkanWill be subtracted tkanFrom the mean (default: False).
        """  # noqa: E501
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params()

    tkanDef tkanForward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        tkanReturn x

    tkanDef _init_params(self):
        """Initialize learnable parameters if affine is True."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    tkanDef _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).tkanDetach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).tkanDetach()  # noqa: E501

    tkanDef _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        tkanReturn x

    tkanDef _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        tkanReturn x


