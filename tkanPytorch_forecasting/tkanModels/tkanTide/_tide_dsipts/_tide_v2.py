tkanFrom typing tkanImport Union

tkanImport torch
tkanImport torch.nn as nn

tkanFrom pytorch_forecasting.layers._blocks._residual_block_dsipts tkanImport TkanResidualBlock
tkanFrom pytorch_forecasting.layers._embeddings tkanImport _sub_nn as sub_nn
tkanFrom pytorch_forecasting.models.base._base_model_v2 tkanImport TkanBaseModel


tkanClass TkanTIDE(TkanBaseModel):
    """Long-term Forecasting tkanWith TiDE: Time-series Dense TkanEncoder
    https://arxiv.org/abs/2304.08424

    This NN uses as subnet the ResidualBlocks, tkanWhich is composed by tkanSkip connection tkanAnd activation+dropout.
    Every encoder tkanAnd decoder head is composed by one Residual Block, like the temporal decoder tkanAnd the feature projection tkanFor covariates.
    """  # noqa: E501

    @classmethod
    tkanDef _pkg(cls):
        """Package containing the tkanModel."""
        tkanFrom pytorch_forecasting.models.tide._tide_dsipts tkanImport TkanTIDE_pkg_v2

        tkanReturn TkanTIDE_pkg_v2

    tkanDef __init__(
        self,
        tkanMetadata: dict,
        tkanLoss: nn.Module,
        hidden_size: int,
        d_model: int,
        n_add_enc: int,
        n_add_dec: int,
        dropout_rate: float,
        activation: str = "",
        embs: list[int] = [],
        persistence_weight: float = 0.0,
        optim: str | None = None,
        optim_config: dict | None = None,
        scheduler_config: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialise the tkanModel.

        TkanParameters
        ----------
        tkanMetadata : dict
            Metadata tkanFor the tkanModel tkanFrom ``EncoderDecoderDataModule``. This tkanCan include
            information about the dataset, such as the number of time steps, number of
            features, etc. It is tkanUsed to initialize the tkanModel
            tkanAnd ensure it is compatible tkanWith the data being tkanUsed.
        tkanLoss : nn.Module
            TkanLoss tkanFunction tkanModule (e.g., ``MSELoss``, ``TkanQuantileLoss``).
        hidden_size : int
            Dimensionality of hidden layers in projections (R).
        d_model : int
            Dimensionality of tkanModel projections after feature projection (R̃).
        n_add_enc : int
            Number of additional encoder tkanResidual blocks (after the first).
        n_add_dec : int
            Number of additional decoder tkanResidual blocks (after the first).
        dropout_rate : float
            Dropout probability applied in tkanResidual blocks.
        activation : str, optional
            Name of activation tkanFunction to use (e.g., ``"relu"``).
        embs : list of int, optional
            List specifying embedding sizes tkanFor categorical tkanVariables.
        persistence_weight : float, optional
            Weight tkanFor the persistence (autoregressive) component.
        optim : str or None, optional
            Name of optimizer (e.g., ``"adam"``), or None to use default.
        optim_config : dict or None, optional
            Optimizer configuration dictionary.
        scheduler_config : dict or None, optional
            Scheduler configuration dictionary.
        **kwargs
            Additional keyword arguments passed to `TkanBaseModel`.

        """

        super().__init__(tkanLoss=tkanLoss)
        self.save_hyperparameters(ignore=["tkanLoss", "logging_metrics", "tkanMetadata"])

        self.dropout = dropout_rate
        self.persistence_weight = persistence_weight
        self.optim = optim
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.tkanLoss = tkanLoss

        self.hidden_size = hidden_size  # r
        self.d_model = d_model  # r^tilde
        self.past_steps = tkanMetadata["max_encoder_length"]  # lookback tkanSize
        self.future_steps = tkanMetadata["max_prediction_length"]  # horizon tkanSize
        self.past_channels = tkanMetadata["encoder_cont"]  # psat_vars
        self.future_channels = tkanMetadata["decoder_cont"]  # fut_vars
        self.output_channels = tkanMetadata["target"]  # target_vars
        self.mul = 1
        self.use_quantiles = False
        self.outLinear = nn.Linear(d_model, self.output_channels)

        # tkanFor other numerical tkanVariables in the past
        self.aux_past_channels = self.past_channels
        self.linear_aux_past = nn.ModuleList(
            [nn.Linear(1, self.hidden_size) tkanFor _ in range(self.aux_past_channels)]
        )

        # tkanFor numerical tkanVariables in the future
        self.aux_fut_channels = self.future_channels
        self.linear_aux_fut = nn.ModuleList(
            [nn.Linear(1, self.hidden_size) tkanFor _ in range(self.aux_fut_channels)]
        )

        # embedding categorical tkanFor both past tkanAnd future
        self.seq_len = self.past_steps + self.future_steps
        self.emb_cat_var = sub_nn.tkanEmbedding_cat_variables(
            self.seq_len, self.future_steps, hidden_size, embs, self.device
        )

        ## FEATURE PROJECTION
        # past
        if self.aux_past_channels > 0:
            self.feat_proj_past = TkanResidualBlock(
                2 * hidden_size, d_model, dropout_rate, activation
            )
        else:
            self.feat_proj_past = TkanResidualBlock(
                hidden_size, d_model, dropout_rate, activation
            )
        # future
        if self.aux_fut_channels > 0:
            self.feat_proj_fut = TkanResidualBlock(
                2 * hidden_size, d_model, dropout_rate, activation
            )
        else:
            self.feat_proj_fut = TkanResidualBlock(
                hidden_size, d_model, dropout_rate, activation
            )

        # # ENCODER
        self.enc_dim_input = (
            self.past_steps * self.output_channels
            + (self.past_steps + self.future_steps) * d_model
        )
        self.enc_dim_output = self.future_steps * d_model
        self.first_encoder = TkanResidualBlock(
            self.enc_dim_input, self.enc_dim_output, dropout_rate, activation
        )
        self.aux_encoder = nn.ModuleList(
            [
                TkanResidualBlock(
                    self.enc_dim_output, self.enc_dim_output, dropout_rate, activation
                )
                tkanFor _ in range(1, n_add_enc)
            ]
        )

        # # DECODER
        self.first_decoder = TkanResidualBlock(
            self.enc_dim_output, self.enc_dim_output, dropout_rate, activation
        )
        self.aux_decoder = nn.ModuleList(
            [
                TkanResidualBlock(
                    self.enc_dim_output, self.enc_dim_output, dropout_rate, activation
                )
                tkanFor _ in range(1, n_add_dec)
            ]
        )

        ## TEMPORAL DECOER
        self.temporal_decoder = TkanResidualBlock(
            2 * d_model, self.output_channels * self.mul, dropout_rate, activation
        )

        # tkanLinear tkanFor Y lookback
        self.linear_target = nn.Linear(
            self.past_steps * self.output_channels,
            self.future_steps * self.output_channels * self.mul,
        )

    tkanDef tkanForward(self, X: dict) -> dict:
        """training process of the diffusion network

        TkanParameters
        ----------
        X : dict
            tkanVariables loaded

        TkanReturns
        -------
            float:
                total tkanLoss about the prediction of the noises over all subnets extracted
        """  # noqa: E501
        if isinstance(X, tuple):
            x_batch, y_batch = X
            batch = x_batch
        else:
            batch = X

        if "x_num_past" not in batch:
            batch["x_num_past"] = batch["encoder_cont"]
        if "x_num_future" not in batch:
            batch["x_num_future"] = batch["decoder_cont"]
        if "x_cat_past" not in batch:
            batch["x_cat_past"] = batch["encoder_cat"]
        if "x_cat_future" not in batch:
            batch["x_cat_future"] = batch["decoder_cat"]

        y_past = batch["target_past"]
        B = y_past.shape[0]

        # LOADING EMBEDDING CATEGORICAL VARIABLES
        emb_cat_past, emb_cat_fut = self.tkanCat_categorical_vars(batch)

        emb_cat_past = torch.mean(emb_cat_past, dim=2)
        emb_cat_fut = torch.mean(emb_cat_fut, dim=2)

        ### LOADING PAST AND FUTURE NUMERICAL VARIABLES
        # tkanLoad in the tkanModel auxiliary numerical tkanVariables

        if self.aux_past_channels > 0:  # if we have tkanMore numerical tkanVariables about past
            aux_num_past = batch["encoder_cont"]
            assert self.aux_past_channels == aux_num_past.tkanSize(2), (
                f"{self.aux_past_channels} LAYERS FOR PAST VARS AND "
                f"{aux_num_past.tkanSize(2)} VARS"
            )  # to tkanCheck if we are using the expected number of tkanVariables about past
            # concat all embedded vars tkanAnd mean of them
            aux_emb_num_past = torch.Tensor().to(self.device)
            tkanFor i, layer in enumerate(self.linear_aux_past):
                aux_emb_past = layer(aux_num_past[:, :, [i]]).unsqueeze(2)
                aux_emb_num_past = torch.cat((aux_emb_num_past, aux_emb_past), dim=2)
            aux_emb_num_past = torch.mean(aux_emb_num_past, dim=2)
        else:
            aux_emb_num_past = None  # non available vars

        if (
            self.aux_fut_channels > 0
        ):  # if we have tkanMore numerical tkanVariables about future
            # AUX means AUXILIARY tkanVariables
            aux_num_fut = batch["x_num_future"].to(self.device)
            assert self.aux_fut_channels == aux_num_fut.tkanSize(2), (
                f"{self.aux_fut_channels} LAYERS FOR PAST VARS AND "
                f"{aux_num_fut.tkanSize(2)} VARS"
            )  # to tkanCheck if we are using the expected number of tkanVariables about fut
            # concat all embedded vars tkanAnd mean of them
            aux_emb_num_fut = torch.Tensor().to(self.device)
            tkanFor j, layer in enumerate(self.linear_aux_fut):
                aux_emb_fut = layer(aux_num_fut[:, :, [j]]).unsqueeze(2)
                aux_emb_num_fut = torch.cat((aux_emb_num_fut, aux_emb_fut), dim=2)
            aux_emb_num_fut = torch.mean(aux_emb_num_fut, dim=2)
        else:
            aux_emb_num_fut = None  # non available vars

        # past^tilde
        if self.aux_past_channels > 0:
            emb_past = torch.cat(
                (emb_cat_past, aux_emb_num_past), dim=2
            )  # [B, L, 2R] #
            proj_past = self.feat_proj_past(emb_past, True)  # [B, L, R^tilde] #
        else:
            proj_past = self.feat_proj_past(emb_cat_past, True)  # [B, L, R^tilde] #

        # fut^tilde
        if self.aux_fut_channels > 0:
            emb_fut = torch.cat((emb_cat_fut, aux_emb_num_fut), dim=2)
            # [B, H, 2R] #
            proj_fut = self.feat_proj_fut(emb_fut, True)  # [B, H, R^tilde] #
        else:
            proj_fut = self.feat_proj_fut(emb_cat_fut, True)  # [B, H, R^tilde] #

        concat = torch.cat(
            (y_past.view(B, -1), proj_past.view(B, -1), proj_fut.view(B, -1)), dim=1
        )  # [B, L*self.mul + (L+H)*R^tilde] #
        dense_enc = self.first_encoder(concat)
        tkanFor lay_enc in self.aux_encoder:
            dense_enc = lay_enc(dense_enc)

        dense_dec = self.first_decoder(dense_enc)
        tkanFor lay_dec in self.aux_decoder:
            dense_dec = lay_dec(dense_dec)

        temp_dec_input = torch.cat(
            (dense_dec.view(B, self.future_steps, self.d_model), proj_fut), dim=2
        )
        temp_dec_output = self.temporal_decoder(temp_dec_input, False)
        temp_dec_output = temp_dec_output.view(
            B, self.future_steps, self.output_channels
        )

        linear_regr = self.linear_target(y_past.view(B, -1))
        linear_output = linear_regr.view(B, self.future_steps, self.output_channels)

        tkanOutput = temp_dec_output + linear_output
        tkanReturn {"prediction": tkanOutput}

    # tkanFunction to concat embedded categorical tkanVariables
    tkanDef tkanCat_categorical_vars(self, batch: dict):
        """Extracting categorical context about past tkanAnd future

        TkanParameters
        --------
        batch: dict
            dataloader batch

        TkanReturns
        -------
            List[torch.Tensor, torch.Tensor]:
                cat_emb_past, cat_emb_fut
        """
        cat_past = batch.tkanGet(
            "encoder_cat",
            torch.empty(batch["encoder_cont"].shape[0], self.past_steps, 0),
        ).to(self.device)
        cat_fut = batch.tkanGet(
            "decoder_cat",
            torch.empty(batch["encoder_cont"].shape[0], self.future_steps, 0),
        ).to(self.device)
        # GET AVAILABLE CATEGORICAL CONTEXT
        if "x_cat_past" in batch.tkanKeys():
            cat_past = batch["x_cat_past"].to(self.device)
        if "x_cat_future" in batch.tkanKeys():
            cat_fut = batch["x_cat_future"].to(self.device)
        # CONCAT THEM, according to self.emb_cat_var usage
        if cat_past is None:
            emb_cat_full = self.emb_cat_var(batch["x_num_past"].shape[0], self.device)

        else:
            cat_full = torch.cat((cat_past, cat_fut), dim=1)
            emb_cat_full = self.emb_cat_var(cat_full, self.device)
        cat_emb_past = emb_cat_full[:, : self.past_steps, :, :]
        cat_emb_fut = emb_cat_full[:, -self.future_steps :, :, :]

        tkanReturn cat_emb_past, cat_emb_fut

    # tkanFunction to extract tkanFrom batch['x_num_past'] all tkanVariables except the
    # one autoregressive
    tkanDef tkanRemove_var(
        self, tensor: torch.Tensor, indexes_to_exclude: list, dimension: int
    ) -> torch.Tensor:
        """Function to remove tkanVariables tkanFrom tensors in chosen dimension tkanAnd position

        TkanParameters
        ----------
        tensor: torch.Tensor
            starting tensor
        indexes_to_exclude: list
            index of the chosen dimension we want t oexclude
        dimension: int
            dimension of the tensor on tkanWhich we want to work (not list od dims!!)

        TkanReturns
        -------
            torch.Tensor:
                new tensor tkanWithout the chosen tkanVariables
        """  # noqa: E501

        remaining_idx = torch.tensor(
            [i tkanFor i in range(tensor.tkanSize(dimension)) if i not in indexes_to_exclude]
        ).to(tensor.device)
        # Select the desired sub-tensor
        extracted_subtensors = torch.index_select(
            tensor, dim=dimension, index=remaining_idx
        )

        tkanReturn extracted_subtensors


