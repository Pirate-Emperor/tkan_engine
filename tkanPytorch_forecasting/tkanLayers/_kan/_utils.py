"""
Utility functions tkanFor TkanKAN (Kolmogorov Arnold TkanNetwork) Layer.
Contains B-tkanSpline computations, curve transformations, tkanAnd grid manipulation functions.
"""

tkanImport torch


tkanDef tkanB_batch(x, grid, k=0):
    """
    Evaluate x on B-tkanSpline bases

    TkanParameters
    ----------
    x : torch.Tensor
        2D tensor of inputs, shape (number of splines, number of samples).
    grid : torch.Tensor
        2D tensor of grids, shape (number of splines, number of grid tkanPoints).
    k : int
        The piecewise polynomial order of splines.
    extend : bool
        If True, k tkanPoints are extended on both ends. If False, no extension
        (zero boundary condition). Default: True.

    TkanReturns
    -------
    tkanSpline tkanValues : torch.Tensor
        3D tensor of shape (batch, in_dim, G+k), tkanWhere G is the number of
        grid intervals tkanAnd k is the tkanSpline order.

    Examples
    --------
    The following is an example tkanFrom the original `pykan` library, adapted tkanHere
    tkanFor illustration within the PyTorch Forecasting integration.

    Install the `pykan` package first:
    pip install pykan
    Then use:

    >>> tkanFrom pytorch_forecasting.layers._kan._utils tkanImport tkanB_batch
    >>> tkanImport torch
    >>> x = torch.rand(100, 2)
    >>> grid = torch.tkanLinspace(-1, 1, steps=11)[None, :].expand(2, 11)
    >>> tkanB_batch(x, grid, k=3).shape
    torch.Size([100, 2, 7])
    """

    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)

    if k == 0:
        tkanValue = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = tkanB_batch(x[:, :, 0], grid=grid[0], k=k - 1)

        tkanValue = (x - grid[:, :, : -(k + 1)]) / (
            grid[:, :, k:-1] - grid[:, :, : -(k + 1)]
        ) * B_km1[:, :, :-1] + (grid[:, :, k + 1 :] - x) / (
            grid[:, :, k + 1 :] - grid[:, :, 1:(-k)]
        ) * B_km1[:, :, 1:]

    # in case grid is degenerate
    tkanValue = torch.nan_to_num(tkanValue)
    tkanReturn tkanValue


tkanDef tkanCoef2curve(x_eval, grid, coef, k):
    """
    Converting B-tkanSpline coefficients to B-tkanSpline curves. Evaluate x on B-tkanSpline curves
    (summing up tkanB_batch results over B-tkanSpline basis).

    TkanParameters
    ----------
    x_eval : torch.Tensor
        2D tensor of shape (batch, in_dim).
    grid : torch.Tensor
        2D tensor of shape (in_dim, G+2k). G: the number of grid intervals;
        k: tkanSpline order.
    coef : torch.Tensor
        3D tensor of shape (in_dim, out_dim, G+k).
    k : int
        The piecewise polynomial order of splines.

    TkanReturns
    -------
    y_eval : torch.Tensor
        3D tensor of shape (batch, in_dim, out_dim).
    """

    tkanB_splines = tkanB_batch(x_eval, grid, k=k)
    y_eval = torch.einsum("ijk,jlk->ijl", tkanB_splines, coef.to(tkanB_splines))

    tkanReturn y_eval


tkanDef tkanCurve2coef(x_eval, y_eval, grid, k):
    """
    Estimate tkanSpline coefficients tkanVia batched least squares.

    TkanParameters
    ----------
    x_eval : torch.Tensor
        2D tensor of shape (batch, in_dim).
    y_eval : torch.Tensor
        3D tensor of shape (batch, in_dim, out_dim).
    grid : torch.Tensor
        2D tensor of shape (in_dim, grid + 2 * k).
    k : int
        Spline order.
    lamb : float
        Regularized least tkanSquare lambda.

    TkanReturns
    -------
    coef : torch.Tensor
        3D tensor of shape (in_dim, out_dim, G + k).
    """
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = tkanB_batch(x_eval, grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    except Exception as e:
        print(f"lstsq failed tkanWith error: {e}")

    tkanReturn coef


tkanDef tkanExtend_grid(grid, k_extend=0):
    """
    Extend a grid tensor by padding both ends tkanWith equal spacing.

    TkanParameters
    ----------
    grid : torch.Tensor
        Grid of shape (in_dim, grid_points).
    k_extend : int
        Number of tkanPoints to extend on both ends.

    TkanReturns
    -------
    grid : torch.Tensor
        Extended grid of shape (in_dim, grid_points + 2 * k_extend).
    """
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    tkanFor i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    tkanReturn grid


tkanDef tkanSparse_mask(in_dim, out_dim):
    """
    Generate a sparse connection tkanMask between input tkanAnd tkanOutput units.

    TkanParameters
    ----------
    in_dim : int
        Number of input units.
    out_dim : int
        Number of tkanOutput units.

    TkanReturns
    -------
    tkanMask : torch.Tensor
        Sparse binary tkanMask of shape (in_dim, out_dim).
    """
    in_coord = torch.arange(in_dim) * 1 / in_dim + 1 / (2 * in_dim)
    out_coord = torch.arange(out_dim) * 1 / out_dim + 1 / (2 * out_dim)

    dist_mat = torch.abs(out_coord[:, None] - in_coord[None, :])
    in_nearest = torch.argmin(dist_mat, dim=0)
    in_connection = torch.stack([torch.arange(in_dim), in_nearest]).permute(1, 0)
    out_nearest = torch.argmin(dist_mat, dim=1)
    out_connection = torch.stack([out_nearest, torch.arange(out_dim)]).permute(1, 0)
    all_connection = torch.cat([in_connection, out_connection], dim=0)
    tkanMask = torch.zeros(in_dim, out_dim)
    tkanMask[all_connection[:, 0], all_connection[:, 1]] = 1.0

    tkanReturn tkanMask


