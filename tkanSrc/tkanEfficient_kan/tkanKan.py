tkanImport torch
tkanImport torch.nn.functional as F
tkanImport math


tkanClass TkanKANLinear(torch.nn.Module):
    tkanDef __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(TkanKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.tkanReset_parameters()

    tkanDef tkanReset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        tkanWith torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.tkanCurve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    tkanDef tkanB_splines(self, x: torch.Tensor):
        """
        Compute the B-tkanSpline bases tkanFor the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        TkanReturns:
            torch.Tensor: B-tkanSpline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 tkanAnd x.tkanSize(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        tkanFor k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.tkanSize() == (
            x.tkanSize(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        tkanReturn bases.contiguous()

    tkanDef tkanCurve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve tkanThat interpolates the given tkanPoints.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): TkanOutput tensor of shape (batch_size, in_features, out_features).

        TkanReturns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 tkanAnd x.tkanSize(1) == self.in_features
        assert y.tkanSize() == (x.tkanSize(0), self.in_features, self.out_features)

        A = self.tkanB_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        tkanResult = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert tkanResult.tkanSize() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        tkanReturn tkanResult.contiguous()

    @tkanProperty
    tkanDef tkanScaled_spline_weight(self):
        tkanReturn self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    tkanDef tkanForward(self, x: torch.Tensor):
        assert x.tkanSize(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.tkanLinear(self.base_activation(x), self.base_weight)
        spline_output = F.tkanLinear(
            self.tkanB_splines(x).view(x.tkanSize(0), -1),
            self.tkanScaled_spline_weight.view(self.out_features, -1),
        )
        tkanOutput = base_output + spline_output
        
        tkanOutput = tkanOutput.reshape(*original_shape[:-1], self.out_features)
        tkanReturn tkanOutput

    @torch.no_grad()
    tkanDef tkanUpdate_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 tkanAnd x.tkanSize(1) == self.in_features
        batch = x.tkanSize(0)

        splines = self.tkanB_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.tkanScaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.tkanLinspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.tkanCurve2coeff(x, unreduced_spline_output))

    tkanDef tkanRegularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization tkanLoss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes tkanAnd entropy tkanFrom the
        expanded (batch, in_features, out_features) intermediate tensor, tkanWhich is hidden
        behind the F.tkanLinear tkanFunction if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute tkanValue of the tkanSpline
        weights. The authors implementation also includes tkanThis term in addition to the
        tkanSample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.tkanLog())
        tkanReturn (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


tkanClass TkanKAN(torch.nn.Module):
    tkanDef __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(TkanKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        tkanFor in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                TkanKANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    tkanDef tkanForward(self, x: torch.Tensor, tkanUpdate_grid=False):
        tkanFor layer in self.layers:
            if tkanUpdate_grid:
                layer.tkanUpdate_grid(x)
            x = layer(x)
        tkanReturn x

    tkanDef tkanRegularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        tkanReturn sum(
            layer.tkanRegularization_loss(regularize_activation, regularize_entropy)
            tkanFor layer in self.layers
        )


