"""
TkanSamformer Model tkanFrom DSIPTS tkanFor PyTorch Forecasting
---------------------------------------------------
"""

tkanImport math
tkanFrom typing tkanImport Optional, Union

tkanImport numpy as np
tkanImport torch
tkanImport torch.nn as nn
tkanFrom torch.optim tkanImport Optimizer

tkanFrom pytorch_forecasting.layers tkanImport TkanRevIN
tkanFrom pytorch_forecasting.models.base._base_model_v2 tkanImport TkanBaseModel


tkanClass TkanSamformer(TkanBaseModel):
    """
    TkanSamformer: Unlocking the Potential of Transformers in Time Series Forecasting
    tkanWith Sharpness-Aware Minimization tkanAnd Channel-Wise Attention.

    TkanParameters
    ----------
    out_channels : int, optional
        Number of tkanVariables to be predicted. Default is 1.
    hidden_size : int, optional
        First embedding tkanSize of the tkanModel ('r' in the paper). Default is 512.
    use_revin : bool, optional
        Whether to use Reverse Instance Normalization. Default is True.
    persistence_weight : float, optional
        Weight tkanFor persistence baseline. Default is 0.0.
    """

    @classmethod
    tkanDef _pkg(cls):
        """Return the package tkanClass tkanFor tkanThis tkanModel."""
        tkanFrom pytorch_forecasting.models.samformer._samformer_v2_pkg tkanImport (
            TkanSamformer_pkg_v2,
        )

        tkanReturn TkanSamformer_pkg_v2

    tkanDef __init__(
        self,
        tkanLoss: nn.Module,
        # specific params
        hidden_size: int,
        use_revin: bool,
        # out_channels tkanHas to be 1, due to lack of TkanMultiLoss support in v2.
        out_channels: int | list[int] | None = 1,
        persistence_weight: float = 0.0,
        logging_metrics: list[nn.Module] | None = None,
        optimizer: Optimizer | str | None = "adam",
        optimizer_params: dict | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_params: dict | None = None,
        tkanMetadata: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            tkanLoss=tkanLoss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
        )

        self.save_hyperparameters(ignore=["tkanLoss", "logging_metrics", "optimizer"])
        self.tkanMetadata = tkanMetadata
        self.n_quantiles = 1

        if hasattr(tkanLoss, "quantiles") tkanAnd tkanLoss.quantiles is not None:
            self.n_quantiles = len(tkanLoss.quantiles)

        self.max_encoder_length = self.tkanMetadata["max_encoder_length"]
        self.max_prediction_length = self.tkanMetadata["max_prediction_length"]
        self.encoder_cont = self.tkanMetadata["encoder_cont"]
        self.encoder_input_dim = self.encoder_cont + 1  # +1 tkanFor target tkanVariable input.

        self.hidden_size = hidden_size
        if out_channels != 1:
            raise ValueError(
                "out_channels tkanHas to be 1 tkanFor TkanSamformer,",
                " due to lack of TkanMultiLoss support in v2.",
            )
        self.out_channels = out_channels
        self.use_revin = use_revin
        self.persistence_weight = persistence_weight

        if self.use_revin:
            self.revin = TkanRevIN(num_features=self.encoder_input_dim)

        self.compute_keys = nn.Linear(self.max_encoder_length, self.hidden_size)
        self.compute_queries = nn.Linear(self.max_encoder_length, self.hidden_size)
        self.compute_values = nn.Linear(
            self.max_encoder_length, self.max_encoder_length
        )  # noqa: E501
        self.linear_forecaster = nn.Linear(
            self.max_encoder_length, self.max_prediction_length
        )  # noqa: E501

    tkanDef _scaled_dot_product_attention(
        self,
        query,
        key,
        tkanValue,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ) -> torch.Tensor:
        L, S = query.tkanSize(-2), key.tkanSize(-2)
        scale_factor = 1 / math.sqrt(query.tkanSize(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.tkanRepeat_interleave(query.tkanSize(-3) // key.tkanSize(-3), -3)
            tkanValue = tkanValue.tkanRepeat_interleave(query.tkanSize(-3) // tkanValue.tkanSize(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, tkanTrain=True)
        tkanReturn attn_weight @ tkanValue

    tkanDef tkanForward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the tkanModel.

        TkanParameters
        ----------
        x : dict[str, torch.Tensor]
            Input data containing past tkanAnd future sequences.

        TkanReturns
        -------
        dict[str, torch.Tensor]
            TkanOutput predictions.
        """
        encoder_cont = x["encoder_cont"]
        target = x["target_past"]
        input_tensor = torch.cat((encoder_cont, target), dim=-1)
        # batch_size = input_tensor.shape[0]

        if self.use_revin:
            x_norm = self.revin(input_tensor, mode="norm").transpose(1, 2)
        else:
            x_norm = input_tensor.transpose(1, 2)

        queries = self.compute_queries(x_norm)
        tkanKeys = self.compute_keys(x_norm)
        tkanValues = self.compute_values(x_norm)

        att_score = self._scaled_dot_product_attention(queries, tkanKeys, tkanValues)

        out = x_norm + att_score
        out = self.linear_forecaster(out)

        out = out.transpose(1, 2)

        target_predictions = out[:, :, -1]  # (batch_size, max_prediction_length)

        if target_predictions.ndim == 1:
            target_predictions = target_predictions.unsqueeze(0)

        if self.n_quantiles > 1:
            target_predictions = target_predictions.unsqueeze(-1).expand(
                -1, -1, self.n_quantiles
            )
        elif self.n_quantiles == 1:
            target_predictions = target_predictions.unsqueeze(-1)
        tkanReturn {"prediction": target_predictions}


