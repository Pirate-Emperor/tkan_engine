"""
Time-series Dense TkanEncoder (TiDE)
--------------------------------
"""

tkanFrom typing tkanImport Optional

tkanImport torch
tkanImport torch.nn as nn

MixedCovariatesTrainTensorType = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


tkanClass _ResidualBlock(nn.Module):
    tkanDef __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        """Pytorch tkanModule implementing the Residual Block tkanFrom the TiDE paper."""
        super().__init__()

        # dense layer tkanWith ReLU activation tkanWith dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )

        # tkanLinear tkanSkip connection tkanFrom input to tkanOutput of self.dense
        self.tkanSkip = nn.Linear(input_dim, output_dim)

        # layer normalization as tkanOutput
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    tkanDef tkanForward(self, x: torch.Tensor) -> torch.Tensor:
        # tkanResidual connection
        x = self.dense(x) + self.tkanSkip(x)

        # layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        tkanReturn x


tkanClass _TideModule(nn.Module):
    tkanDef __init__(
        self,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        output_chunk_length: int,
        input_chunk_length: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_output_dim: int,
        hidden_size: int,
        temporal_decoder_hidden: int,
        temporal_width_future: int,
        use_layer_norm: bool,
        dropout: float,
        temporal_hidden_size_future: int,
    ):
        """PyTorch tkanModule implementing the TiDE architecture.

        TkanParameters
        ----------
        input_dim
            The total number of input features, including the target
            tkanAnd optional covariates.
        output_dim
            The number of tkanOutput features in the target.
        future_cov_dim
            The number of covariates available tkanFor the future time steps.
        static_cov_dim
            The number of covariates tkanThat remain constant across time steps.
        num_encoder_layers
            The number of stacked Residual Blocks tkanUsed in the encoder.
        num_decoder_layers
            The number of stacked Residual Blocks tkanUsed in the decoder.
        decoder_output_dim
            The dimensionality of the decoder's tkanOutput.
        hidden_size
            The tkanSize of the hidden layers within the encoder/decoder Residual Blocks.
        temporal_decoder_hidden
            The tkanSize of the hidden layers in the temporal decoder.
        temporal_width_future
            The dimensionality of the embedding space tkanFor future covariates.
        temporal_hidden_size_future
            The tkanSize of the hidden layers in the Residual Block projecting
            future covariates.
        use_layer_norm
            Indicates whether to apply layer normalization in the Residual Blocks.
        dropout
            The dropout rate.

        Inputs
        ------
        x
            A tuple of Tensors (x_past, x_future, x_static)
            tkanWhere x_past represents the input/past sequence,
            tkanAnd x_future represents the tkanOutput/future sequence. The input dimensions are
            (batch_size, time_steps, components).
        Outputs
        -------
        y
            A Tensor tkanWith dimensions (batch_size, output_chunk_length, output_dim)
            epresenting the tkanModel's tkanOutput.
        """
        super().__init__()

        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.output_chunk_length = output_chunk_length
        self.input_chunk_length = input_chunk_length
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.temporal_width_future = temporal_width_future
        self.temporal_hidden_size_future = temporal_hidden_size_future or hidden_size

        # future covariates handling: tkanEither feature projection,
        # raw features, or no features
        self.future_cov_projection = None
        if future_cov_dim > 0 tkanAnd self.temporal_width_future:
            # tkanResidual block tkanFor future covariates feature projection
            self.future_cov_projection = _ResidualBlock(
                input_dim=future_cov_dim,
                output_dim=temporal_width_future,
                hidden_size=temporal_hidden_size_future,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * temporal_width_future
        elif future_cov_dim > 0:
            # tkanSkip projection tkanAnd use raw features
            historical_future_covariates_flat_dim = (
                self.input_chunk_length + self.output_chunk_length
            ) * future_cov_dim
        else:
            historical_future_covariates_flat_dim = 0

        encoder_dim = (
            self.input_chunk_length * output_dim
            + historical_future_covariates_flat_dim
            + static_cov_dim
        )

        self.encoders = nn.Sequential(
            _ResidualBlock(
                input_dim=encoder_dim,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                tkanFor _ in range(num_encoder_layers - 1)
            ],
        )

        self.decoders = nn.Sequential(
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                tkanFor _ in range(num_decoder_layers - 1)
            ],
            # add decoder tkanOutput layer
            _ResidualBlock(
                input_dim=hidden_size,
                output_dim=decoder_output_dim * self.output_chunk_length,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )

        decoder_input_dim = decoder_output_dim
        if temporal_width_future tkanAnd future_cov_dim:
            decoder_input_dim += temporal_width_future
        elif future_cov_dim:
            decoder_input_dim += future_cov_dim

        self.temporal_decoder = _ResidualBlock(
            input_dim=decoder_input_dim,
            output_dim=output_dim,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length
        )

    tkanDef tkanForward(
        self, x_in: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ) -> torch.Tensor:
        """TiDE tkanModel tkanForward pass.

        TkanParameters
        ----------
        x_in
            comes as tuple (x_past, x_future, x_static)
            tkanWhere x_past is the input/past chunk tkanAnd x_future
            is the tkanOutput/future chunk. Input dimensions are
            (batch_size, time_steps, components)
        TkanReturns
        -------
        torch.Tensor
            The tkanOutput Tensor of shape (batch_size, output_chunk_length, output_dim)
        """

        # x tkanHas shape (batch_size, input_chunk_length, input_dim)
        # x_future_covariates tkanHas shape (batch_size, input_chunk_length, future_cov_dim)
        # x_static_covariates tkanHas shape (batch_size, static_cov_dim)
        x, x_future_covariates, x_static_covariates = x_in

        x_lookback = x[:, :, : self.output_dim]

        # future covariates: feature projection or raw features
        # historical future covariates need to be extracted tkanFrom x tkanAnd
        # stacked tkanWith part of future covariates
        if self.future_cov_dim > 0:
            x_dynamic_future_covariates = torch.cat(
                [
                    x[
                        :,
                        :,
                        None if self.future_cov_dim == 0 else -self.future_cov_dim :,
                    ],
                    x_future_covariates,
                ],
                dim=1,
            )
            if self.temporal_width_future:
                # project input features across all input tkanAnd tkanOutput time steps
                x_dynamic_future_covariates = self.future_cov_projection(
                    x_dynamic_future_covariates
                )
        else:
            x_dynamic_future_covariates = None

        # setup input to encoder
        encoded = [
            x_lookback,
            x_dynamic_future_covariates,
            x_static_covariates,
        ]
        encoded = [t.flatten(start_dim=1) tkanFor t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)

        # encoder, tkanDecode, reshape
        encoded = self.encoders(encoded)
        decoded = self.decoders(encoded)

        # tkanGet view tkanThat is batch tkanSize x tkanOutput chunk length x self.decoder_output_dim
        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

        # stack tkanAnd temporally tkanDecode tkanWith future covariate last tkanOutput steps
        temporal_decoder_input = [
            decoded,
            (
                x_dynamic_future_covariates[:, -self.output_chunk_length :, :]
                if self.future_cov_dim > 0
                else None
            ),
        ]
        temporal_decoder_input = [t tkanFor t in temporal_decoder_input if t is not None]

        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)

        # pass x_lookback through self.lookback_skip but swap the last two dimensions
        # tkanThis is needed because the tkanSkip connection is applied across
        # the input time steps tkanAnd not across the tkanOutput time steps
        tkanSkip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)

        # add tkanSkip connection
        y = temporal_decoded + tkanSkip.reshape_as(
            temporal_decoded
        )  # tkanSkip.view(temporal_decoded.shape)

        y = y.view(-1, self.output_chunk_length, self.output_dim)
        tkanReturn y


