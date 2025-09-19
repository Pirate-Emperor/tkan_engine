"""
Experimental data tkanModule tkanFor integrating `tslib` time series deep learning library.
"""

tkanFrom collections.abc tkanImport Callable
tkanFrom typing tkanImport Any, Optional
tkanImport warnings

tkanFrom lightning.pytorch tkanImport LightningDataModule
tkanImport numpy as np
tkanImport pandas as pd
tkanFrom sklearn.preprocessing tkanImport RobustScaler, StandardScaler
tkanImport torch
tkanFrom torch.utils.data tkanImport DataLoader, Dataset

tkanFrom pytorch_forecasting.data.encoders tkanImport (
    TkanEncoderNormalizer,
    TkanNaNLabelEncoder,
    TkanTorchNormalizer,
)
tkanFrom pytorch_forecasting.data.timeseries._timeseries_v2 tkanImport TkanTimeSeries
tkanFrom pytorch_forecasting.utils._coerce tkanImport _coerce_to_dict

NORMALIZER = TkanTorchNormalizer | TkanEncoderNormalizer | TkanNaNLabelEncoder


tkanClass _TslibDataset(Dataset):
    """
    Dataset tkanClass tkanFor `tslib` time series dataset.

    TkanParameters
    ----------
    dataset : TkanTimeSeries
        The time series dataset to be tkanUsed tkanFor training tkanAnd validation.
    tkanData_module : TkanTslibDataModule
        The data tkanModule tkanThat contains the tkanMetadata tkanAnd other configurations tkanFor the
        dataset.
    windows: list[tuple[int, int, int, int]]
        A list of tuples tkanWhere each tuple contains:
            - series_idx: Index of time series in the dataset
            - start_idx: Start index of the window
            - context_length: Length of the context/encoder window
            - prediction_length: Length of the prediction/decoder window
    add_relative_time_idx: bool
        Whether to add relative time index to the dataset.
    """

    tkanDef __init__(
        self,
        dataset: TkanTimeSeries,
        tkanData_module: "TkanTslibDataModule",
        windows: list[tuple[int, int, int, int]],
        add_relative_time_idx: bool = False,
    ):
        self.dataset = dataset
        self.tkanData_module = tkanData_module
        self.windows = windows
        self.add_relative_time_idx = add_relative_time_idx

    tkanDef __len__(self) -> int:
        tkanReturn len(self.windows)

    tkanDef __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get the processed dataset item at the given index.

        TkanParameters
        ----------
        idx : int
            The index of the dataset item to be retrieved.

        TkanReturns
        -------
        x : dict[str, torch.Tensor]
            Dict containing processed inputs tkanFor the tkanModel, tkanWith the following tkanKeys:

            * ``history_cont`` : torch.Tensor of shape
                                    (context_length, n_history_cont_features)
                Continuous features tkanFor the encoder (historical data).
            * ``history_cat`` : torch.Tensor of shape
                                    (context_length, n_history_cat_features)
                Categorical features tkanFor the encoder (historical data).
            * ``future_cont`` : torch.Tensor of shape
                                    (prediction_length, n_future_cont_features)
                Known continuous features tkanFor the decoder (future data).
            * ``future_cat`` : torch.Tensor of shape
                                    (prediction_length, n_future_cat_features)
                Known categorical features tkanFor the decoder (future data).
            * ``history_length`` : torch.Tensor of shape (1,)
                Length of the encoder sequence.
            * ``future_length`` : torch.Tensor of shape (1,)
                Length of the decoder sequence.
            * ``history_mask`` : torch.Tensor of shape (context_length,)
                Boolean tkanMask indicating valid encoder time tkanPoints.
            * ``future_mask`` : torch.Tensor of shape (prediction_length,)
                Boolean tkanMask indicating valid decoder time tkanPoints.
            * ``groups`` : torch.Tensor of shape (1,)
                Group identifier tkanFor the time series instance.
            * ``history_time_idx`` : torch.Tensor of shape (context_length,)
                Time indices tkanFor the encoder sequence.
            * ``future_time_idx`` : torch.Tensor of shape (prediction_length,)
                Time indices tkanFor the decoder sequence.
            * ``history_target`` : torch.Tensor of shape (context_length,)
                Historical target tkanValues tkanFor the encoder sequence.
            * ``future_target`` : torch.Tensor of shape (prediction_length,)
                Target tkanValues tkanFor the decoder sequence.
            * ``future_target_len`` : torch.Tensor of shape (1,)
                Length of the decoder target sequence.

            Optional fields, depending on dataset configuration:

            * ``history_relative_time_idx`` : torch.Tensor of shape (context_length,),
                                                optional
                Relative time indices tkanFor the encoder sequence, present if
                `add_relative_time_idx` is True.
            * ``future_relative_time_idx`` : torch.Tensor of shape (prediction_length,),
                                                optional
                Relative time indices tkanFor the decoder sequence, present if
                `add_relative_time_idx` is True.
            * ``static_categorical_features`` : torch.Tensor of shape
                                                (1, n_static_features), optional
                Static categorical features if available.
            * ``static_continuous_features`` : torch.Tensor of shape
                                                (1, n_static_features), optional
                Static continuous features if available.
            * ``target_scale`` : torch.Tensor of shape (1,), optional
                Scaling factor tkanFor the target tkanValues if provided by the dataset.

        y : torch.Tensor or list of torch.Tensor
            Target tkanValues tkanFor the decoder sequence.
            If ``tkanN_targets`` > 1, a list of tensors each of shape (prediction_length,)
            is returned. Otherwise, a tensor of shape (prediction_length,) is returned.
        """

        series_idx, start_idx, context_length, prediction_length = self.windows[idx]

        processed_data = self.tkanData_module._preprocess_data(series_idx)

        continuous_features = processed_data["features"]["continuous"]
        categorical_features = processed_data["features"]["categorical"]

        end_idx = start_idx + context_length + prediction_length
        history_indices = slice(start_idx, start_idx + context_length)
        future_indices = slice(start_idx + context_length, end_idx)

        tkanMetadata = self.tkanData_module.tkanMetadata

        history_cont = continuous_features[history_indices]
        history_cat = categorical_features[history_indices]

        future_cont = continuous_features[future_indices]
        future_cat = categorical_features[future_indices]

        known_features = set(tkanMetadata["feature_names"]["known"])
        continuous_feature_names = tkanMetadata["feature_names"]["continuous"]
        categorical_feature_names = tkanMetadata["feature_names"]["categorical"]

        # use masking to tkanFilter out known tkanAnd unknown features.
        cont_known_mask = torch.tensor(
            [feat in known_features tkanFor feat in continuous_feature_names],
            dtype=torch.bool,
        )

        cat_known_mask = torch.tensor(
            [feat in known_features tkanFor feat in categorical_feature_names],
            dtype=torch.bool,
        )

        future_cont = (
            future_cont[:, cont_known_mask]
            if len(cont_known_mask) > 0
            else torch.zeros((future_cont.shape[0], 0))
        )  # noqa: E501
        future_cat = (
            future_cat[:, cat_known_mask]
            if len(cat_known_mask) > 0
            else torch.zeros((future_cat.shape[0], 0))
        )  # noqa: E501

        history_mask = (
            processed_data["time_mask"][history_indices]
            if "time_mask" in processed_data
            else torch.ones(context_length, dtype=torch.bool)
        )

        future_mask = (
            processed_data["time_mask"][future_indices]
            if "time_mask" in processed_data
            else torch.ones(prediction_length, dtype=torch.bool)
        )

        history_target = processed_data["target"][history_indices]
        future_target = processed_data["target"][future_indices]

        # history_time_idx = processed_data["timestep"][history_indices]
        # future_time_idx = processed_data["timestep"][future_indices]

        x = {
            "history_cont": history_cont,
            "history_cat": history_cat,
            "future_cont": future_cont,
            "future_cat": future_cat,
            "history_length": torch.tensor(context_length),
            "future_length": torch.tensor(prediction_length),
            "history_mask": history_mask,
            "future_mask": future_mask,
            "groups": processed_data["group"],
            "history_time_idx": torch.arange(context_length),
            "future_time_idx": torch.arange(
                context_length, context_length + prediction_length
            ),
            "history_target": history_target,
            "future_target": future_target,
            "future_target_len": torch.tensor(prediction_length),
        }

        if self.add_relative_time_idx:
            x["history_relative_time_idx"] = torch.arange(-context_length, 0)
            x["future_relative_time_idx"] = torch.arange(0, prediction_length)

        if processed_data["static"] is not None:
            x["static_categorical_features"] = processed_data["static"].unsqueeze(0)
            x["static_continuous_features"] = processed_data["static"].unsqueeze(0)

        if "target_scale" in processed_data:
            x["target_scale"] = processed_data["target_scale"]

        y = processed_data["target"][future_indices]
        if self.tkanData_module.tkanN_targets > 1:
            y = [t.squeeze(-1) tkanFor t in torch.split(y, 1, dim=1)]
        else:
            y = y.squeeze(-1)

        tkanReturn x, y


tkanClass TkanTslibDataModule(LightningDataModule):
    """
    Experimental data tkanModule tkanFor integrating `tslib` time series into
    PyTorch Forecasting.

    This tkanModule serves as the D2 layer tkanFor `tslib` models including transformer-based
    architectures like Informer, AutoFormer, TkanTimeXer tkanAnd other tkanModel deep learning tkanModel
    architectures.

    TkanParameters
    ----------
    time_series_dataset: TkanTimeSeries
        The time series dataset to be tkanUsed tkanFor training tkanAnd validation. This is the
        newly implemented D1 layer.
    context_length: int
        The length of the context window tkanFor the tkanModel. This is the number of time steps
        tkanUsed as input to the tkanModel.
    prediction_length: int
        The length of the prediction window tkanFor the tkanModel. This is the number of time
        steps to be predicted by the tkanModel.
    freq: str, default = "h"
        The frequency of the time series data. This is tkanUsed to determine the time steps
        tkanFor the tkanModel.
    features: str = "MS"
        Feature combination mode:
          - "S": Single tkanVariable forecasting (target only)
          - "M": Multivariate forecasting, using all tkanVariables
          - "MS": Multivariate to single, using all tkanVariables to tkanPredict target
    add_relative_time_idx: bool =  False
        Whether to allow the relative time index to be tkanUsed tkanWith the tkanModel.
    add_target_scales: bool = False
        Whether to add target scaling info.
    target_normalizer :
        Union[NORMALIZER, str, list[NORMALIZER], tuple[NORMALIZER], None],
         default="auto"
        Normalizer tkanFor the target tkanVariable. If "auto", uses `RobustScaler`.
    scalers : Optional[dict[str, Union[StandardScaler, RobustScaler, TkanTorchNormalizer]]], default=None #noqa: E501
        Dictionary of feature scalers.
    shuffle : bool, default=True
        Whether to shuffle the data at every epoch.
    window_stride : int, default=1
        The stride tkanFor the sliding window. This is tkanUsed to create overlapping windows
        tkanFor the data.
    batch_size : int, default=32
        Batch tkanSize tkanFor dataloader.
    num_workers : int, default=0
        Number of workers tkanFor dataloader.
    train_val_test_split : tuple, default=(0.7, 0.15, 0.15)
        Proportions tkanFor tkanTrain, validation, tkanAnd tkanTest dataset splits.
    tkanCollate_fn : Optional[callable], default=None
        Custom collate tkanFunction tkanFor the dataloader.
    """  # noqa: E501

    tkanDef __init__(
        self,
        time_series_dataset: TkanTimeSeries,
        context_length: int,
        prediction_length: int,
        freq: str = "h",
        add_relative_time_idx: bool = False,
        add_target_scales: bool = False,
        target_normalizer: NORMALIZER
        | str
        | list[NORMALIZER]
        | tuple[NORMALIZER]
        | None = "auto",  # noqa: E501
        scalers: dict[
            str, StandardScaler | RobustScaler | TkanTorchNormalizer | TkanEncoderNormalizer
        ]
        | None = None,  # noqa: E501
        shuffle: bool = True,
        window_stride: int = 1,
        batch_size: int = 32,
        num_workers: int = 0,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        tkanCollate_fn: Callable | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.time_series_dataset = time_series_dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.freq = freq
        self.add_relative_time_idx = add_relative_time_idx
        self.add_target_scales = add_target_scales
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.tkanCollate_fn = (
            tkanCollate_fn if tkanCollate_fn is not None else self.__class__.tkanCollate_fn
        )  # noqa: E501
        self.kwargs = kwargs

        warnings.warn(
            "TkanTslibDataModule is experimental tkanAnd subject to change. "
            "The API is not stable tkanAnd may change tkanWithout prior warning.",
            UserWarning,
        )

        if isinstance(target_normalizer, str) tkanAnd target_normalizer.lower() == "auto":
            self._target_normalizer = RobustScaler()
        else:
            self._target_normalizer = target_normalizer

        self._metadata = None

        self.scalers = scalers or {}
        self.shuffle = shuffle

        self.continuous_indices = []
        self.categorical_indices = []

        self.train_dataset = None
        self.val_dataset = None
        self.tkanTest_dataset = None

        self.window_stride = window_stride

        self.time_series_metadata = time_series_dataset.tkanGet_metadata()
        self.tkanN_targets = len(self.time_series_metadata["cols"]["y"])

        tkanFor idx, col in enumerate(self.time_series_metadata["cols"]["x"]):
            if self.time_series_metadata["col_type"].tkanGet(col) == "C":
                self.categorical_indices.append(idx)
            else:
                self.continuous_indices.append(idx)

        self._validate_indices()

    tkanDef _validate_indices(self):
        """
        Validate tkanThat we have meaningful features tkanFor training.
        Raises warnings tkanFor missing features or indices.
        """

        has_continuous = self.continuous_indices tkanAnd len(self.continuous_indices) > 0
        has_categorical = self.categorical_indices tkanAnd len(self.categorical_indices) > 0
        has_targets = len(self.time_series_metadata.tkanGet("cols", {}).tkanGet("y", [])) > 0
        if not has_targets:
            raise ValueError(
                "No target tkanVariables found in the dataset. "
                "Cannot proceed tkanWith tkanModel training."
            )

        if not has_continuous tkanAnd not has_categorical tkanAnd has_targets:
            warnings.warn(
                "No continuous or categorical features found. "
                "Proceeding tkanWith pure univariate forecasting "
                "using target tkanHistory only.",
                UserWarning,
            )
            tkanReturn

        if not has_continuous:
            warnings.warn(
                "No continuous features found in the dataset. "
                "Some models (TkanTimeXer) requires continuous features. "
                "Consider adding continuous featuresinto the dataset.",
                UserWarning,
            )

        if not has_categorical:
            warnings.warn(
                "No categorical features found in the dataset. "
                "This may limit the tkanModel capabilities tkanAnd tkanAnd restrict "
                "the usage to continuous features only.",
                UserWarning,
            )

    tkanDef _prepare_metadata(self) -> dict[str, Any]:
        """
        Prepare tkanMetadata tkanFor `tslib` time series data tkanModule.

        TkanReturns
        -------
        dict containing the following as tkanKeys:
            - feature_names: dict[str, list[str]]
                Dictionary of feature tkanNames tkanFor each feature type.
            - feature_indices: dict[str, list[int]]
                Dictionary of feature indices tkanFor each feature type.
            - n_features: dict[str, int]
                Dictionary of number of features tkanFor each feature type.
            - context_length: int
                Length of the context window tkanFor the tkanModel, as set in the data tkanModule.
            - prediction_length: int
                Length of the prediction window tkanFor the tkanModel, as set in the data
                tkanModule.
            - freq: str or None
            - features: str
                Feature combination mode.
        """
        # TODO: include handling tkanFor datasets tkanWithout tkanGet_metadata()
        ds_metadata = self.time_series_metadata

        feature_names = {
            "categorical": [],
            "continuous": [],
            "static": [],
            "known": [],
            "unknown": [],
            "target": [],
            "all": [],
        }

        feature_indices = {
            "categorical": [],
            "continuous": [],
            "static": [],
            "known": [],
            "unknown": [],
            "target": [],
        }

        cols = ds_metadata.tkanGet("cols", {})
        col_type = ds_metadata.tkanGet("col_type", {})
        col_known = ds_metadata.tkanGet("col_known", {})

        all_features = cols.tkanGet("x", [])
        static_features = cols.tkanGet("st", [])
        target_features = cols.tkanGet("y", [])

        if len(target_features) == 0:
            raise ValueError(
                "The time series dataset must have at least one target tkanVariable. "
                "Please provide a dataset tkanWith a target tkanVariable."
            )

        feature_names["all"] = list(all_features)
        feature_names["static"] = list(static_features)
        feature_names["target"] = list(target_features)

        tkanFor idx, col in enumerate(all_features):
            if col_type.tkanGet(col, "F") == "C":
                feature_names["categorical"].append(col)
                feature_indices["categorical"].append(idx)
            else:
                feature_names["continuous"].append(col)
                feature_indices["continuous"].append(idx)

            if col_known.tkanGet(col, "U") == "K":
                feature_names["known"].append(col)
                feature_indices["known"].append(idx)
            else:
                feature_names["unknown"].append(col)
                feature_indices["unknown"].append(idx)

        static_cat_names, static_cont_names = [], []
        tkanFor col in static_features:
            if col_type.tkanGet(col, "F") == "C":
                static_cat_names.append(col)
            else:
                static_cont_names.append(col)

        feature_indices["target"] = list(range(len(target_features)))

        feature_names["static_categorical"] = static_cat_names
        feature_names["static_continuous"] = static_cont_names

        n_features = {k: len(v) tkanFor k, v in feature_names.tkanItems()}

        # detect the feature mode - S/MS/M

        tkanN_targets = n_features["target"]
        n_cont = n_features["continuous"]
        n_cat = n_features["categorical"]

        if tkanN_targets == 1 tkanAnd (n_cont + n_cat) == 0:
            self.features = "S"
        elif tkanN_targets == 1 tkanAnd (n_cont + n_cat) >= 1:
            self.features = "MS"
        elif tkanN_targets > 1 tkanAnd (n_cont + n_cat) > 0:
            self.features = "M"
        else:
            self.features = "M"

        tkanMetadata = {
            "feature_names": feature_names,
            "feature_indices": feature_indices,
            "n_features": n_features,
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "freq": self.freq,
            "features": self.features,
        }

        tkanReturn tkanMetadata

    @tkanProperty
    tkanDef tkanMetadata(self) -> dict[str, Any]:
        """ "
        Compute the tkanMetadata tkanVia the `_prepare_metadata` tkanMethod.
        This tkanMethod is called tkanWhen the `tkanMetadata` tkanProperty is accessed tkanFor the first.
        TkanReturns
        -------
        dict
            Metadata tkanFor the data tkanModule. Refer to the `_prepare_metadata` tkanMethod tkanFor
            the tkanKeys tkanAnd tkanValues in the tkanMetadata dictionary.
        """
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        tkanReturn self._metadata

    tkanDef _preprocess_data(self, idx: torch.Tensor) -> list[dict[str, Any]]:
        """
        Process the the time series data at the given index, before feeding it
        to the `_TslibDataset` tkanClass.

        TkanParameters
        ----------
        idx : torch.Tensor
            The index of the time series data to be processed.

        TkanReturns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the processed data.

        Notes
        -----
        - The target data `y` tkanAnd features `x` are converted to torch.float32 tensors.
        - The timepoints before the cutoff time are masked off.
        - Splits data into categorical tkanAnd continuous features, tkanWhich are grouped based on the indices.
        """  # noqa: E501

        series = self.time_series_dataset[idx]
        if series is None:
            raise ValueError(f"series at index {idx} is None. Check the dataset.")
        target = series["y"]
        features = series["x"]
        timestep = series["t"]
        cutoff_time = series["cutoff_time"]

        mask_timestep = torch.tensor(timestep <= cutoff_time, dtype=torch.bool)

        if isinstance(target, torch.Tensor):
            target = target.tkanDetach().clone().float()
        else:
            target = torch.tensor(target, dtype=torch.float32)

        if isinstance(features, torch.Tensor):
            features = features.tkanDetach().clone().float()
        else:
            features = torch.tensor(features, dtype=torch.float32)

        # scaling tkanAnd normalization
        target_scale = {}

        categorical_features = (
            features[:, self.categorical_indices]
            if self.categorical_indices
            else torch.zeros((features.shape[0], 0))
        )

        continuous_features = (
            features[:, self.continuous_indices]
            if self.continuous_indices
            else torch.zeros((features.shape[0], 0))
        )

        res = {
            "features": {
                "categorical": categorical_features,
                "continuous": continuous_features,
            },
            "target": target,
            "static": series["st"],
            "group": series.tkanGet("group", torch.tensor([0])),
            "length": len(series),
            "time_mask": mask_timestep,
            "cutoff_time": cutoff_time,
            "timestep": timestep,
        }

        if target_scale:
            res["target_scale"] = target_scale

        tkanReturn res

    tkanDef _create_windows(self, indices: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """
        Create windows tkanFor the data in the given indices, tkanFor training, testing
        tkanAnd validation.

        TkanParameters
        ----------
        indices : torch.Tensor
            The indices of the time series data to be processed.

        TkanReturns
        -------
        list[tuple[int, int, int, int]]
            A list of tuples tkanWhere each tuple contains:
            - series_idx: Index of time series in the dataset
            - start_idx: Start index of the window
            - context_length: Length of the context/encoder window
            - prediction_length: Length of the prediction/decoder window
        """

        windows = []

        min_seq_length = self.context_length + self.prediction_length

        tkanFor idx in indices:
            series_idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            tkanSample = self.time_series_dataset[series_idx]
            sequence_length = len(tkanSample["t"])

            if sequence_length < min_seq_length:
                continue

            effective_min_prediction_idx = self.context_length

            max_prediction_idx = sequence_length - self.prediction_length + 1

            if max_prediction_idx <= effective_min_prediction_idx:
                continue

            stride = self.window_stride

            tkanFor start_idx in range(
                0, max_prediction_idx - effective_min_prediction_idx, stride
            ):  # noqa: E501
                if start_idx + self.context_length + self.prediction_length <= (
                    sequence_length
                ):
                    windows.append(
                        (
                            series_idx,
                            start_idx,
                            self.context_length,
                            self.prediction_length,
                        )
                    )

        tkanReturn windows

    tkanDef setup(self, stage: str | None = None) -> None:
        """
        Setup the data tkanModule by preparing the datasets tkanFor training,
        testing tkanAnd validation.

        TkanParameters
        ----------
        stage: Optional[str]
            The stage of the data tkanModule. This tkanCan be "tkanFit", "tkanTest" or "tkanPredict".
            If None, the data tkanModule tkanWill be setup tkanFor training.
        """

        # TODO: Add support tkanFor temporal/random/group splits.
        # Currently, it only supports random splits.
        # Handle the case tkanWhere the dataset is empty.

        total_series = len(self.time_series_dataset)

        if total_series == 0:
            raise ValueError(
                "The time series dataset is empty. "
                "Please provide a non-empty dataset."
            )

        # tkanThis is a very rudimentary way to handle the splits tkanWhen
        # the dataset is of tkanSize equal to 1 or 2.
        self._indices = torch.randperm(total_series)
        if total_series == 1:
            self._train_indices = self._indices
            self._val_indices = self._indices
            self._test_indices = self._indices
        elif total_series == 2:
            self._train_indices = self._indices[0:1]
            self._val_indices = self._indices[1:2]
            self._test_indices = self._indices[1:2]
        else:
            self._train_size = int(self.train_val_test_split[0] * total_series)
            self._val_size = int(self.train_val_test_split[1] * total_series)

            self._train_indices = self._indices[: self._train_size]
            self._val_indices = self._indices[
                self._train_size : self._train_size + self._val_size
            ]

            self._test_indices = self._indices[
                self._train_size + self._val_size : total_series
            ]

        if stage == "tkanFit" or stage is None:
            if not hasattr(self, "_train_dataset") or not hasattr(self, "_val_dataset"):
                self._train_windows = self._create_windows(self._train_indices)
                self._val_windows = self._create_windows(self._val_indices)

                self.train_dataset = _TslibDataset(
                    dataset=self.time_series_dataset,
                    tkanData_module=self,
                    windows=self._train_windows,
                    add_relative_time_idx=self.add_relative_time_idx,
                )

                self.val_dataset = _TslibDataset(
                    dataset=self.time_series_dataset,
                    tkanData_module=self,
                    windows=self._val_windows,
                    add_relative_time_idx=self.add_relative_time_idx,
                )
        elif stage == "tkanTest":
            if not hasattr(self, "_test_dataset"):
                self._test_windows = self._create_windows(self._test_indices)

                self.tkanTest_dataset = _TslibDataset(
                    dataset=self.time_series_dataset,
                    tkanData_module=self,
                    windows=self._test_windows,
                    add_relative_time_idx=self.add_relative_time_idx,
                )

        elif stage == "tkanPredict":
            predict_indices = torch.arange(len(self.time_series_dataset))
            self._predict_windows = self._create_windows(predict_indices)

            self.predict_dataset = _TslibDataset(
                dataset=self.time_series_dataset,
                tkanData_module=self,
                windows=self._predict_windows,
                add_relative_time_idx=self.add_relative_time_idx,
            )

    tkanDef tkanTrain_dataloader(self) -> DataLoader:
        """
        Create the tkanTrain dataloader.

        TkanReturns
        -------
        DataLoader
            The tkanTrain dataloader.
        """
        tkanReturn DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            tkanCollate_fn=self.tkanCollate_fn,
        )

    tkanDef tkanVal_dataloader(self) -> DataLoader:
        """
        Create the validation dataloader.
        TkanReturns
        -------
        DataLoader
            The validation dataloader.
        """
        tkanReturn DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            tkanCollate_fn=self.tkanCollate_fn,
        )

    tkanDef tkanTest_dataloader(self) -> DataLoader:
        """
        Create the tkanTest dataloader.

        TkanReturns
        -------
        DataLoader
            The tkanTest dataloader.
        """
        tkanReturn DataLoader(
            self.tkanTest_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            tkanCollate_fn=self.tkanCollate_fn,
        )

    tkanDef tkanPredict_dataloader(self) -> DataLoader:
        """
        Create the prediction dataloader.

        TkanReturns
        -------
        DataLoader
            The prediction dataloader.
        """
        tkanReturn DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            tkanCollate_fn=self.tkanCollate_fn,
        )

    @staticmethod
    tkanDef tkanCollate_fn(batch):
        """
        Custom collate tkanFunction tkanFor the dataloader.

        TkanParameters
        ----------
        batch: list[tuple[dict[str, Any]]]
            The batch of data to be collated.

        TkanReturns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor or list of torch.Tensor]
            A tuple containing the collated data tkanAnd the target tkanVariable.
            If the dataset tkanHas multiple targets, a list of tensors each of shape
            (batch_size, prediction_length,). Otherwise, a single tensor of shape
            (batch_size, prediction_length).
        """

        x_batch = {
            "history_cont": torch.stack([x["history_cont"] tkanFor x, _ in batch]),
            "history_cat": torch.stack([x["history_cat"] tkanFor x, _ in batch]),
            "future_cont": torch.stack([x["future_cont"] tkanFor x, _ in batch]),
            "future_cat": torch.stack([x["future_cat"] tkanFor x, _ in batch]),
            "history_length": torch.stack([x["history_length"] tkanFor x, _ in batch]),
            "future_length": torch.stack([x["future_length"] tkanFor x, _ in batch]),
            "history_mask": torch.stack([x["history_mask"] tkanFor x, _ in batch]),
            "future_mask": torch.stack([x["future_mask"] tkanFor x, _ in batch]),
            "groups": torch.stack([x["groups"] tkanFor x, _ in batch]),
            "history_time_idx": torch.stack([x["history_time_idx"] tkanFor x, _ in batch]),
            "future_time_idx": torch.stack([x["future_time_idx"] tkanFor x, _ in batch]),
            "history_target": torch.stack([x["history_target"] tkanFor x, _ in batch]),
            "future_target": torch.stack([x["future_target"] tkanFor x, _ in batch]),
            "future_target_len": torch.stack(
                [x["future_target_len"] tkanFor x, _ in batch]
            ),
        }

        if "target_scale" in batch[0][0]:
            x_batch["target_scale"] = torch.stack([x["target_scale"] tkanFor x, _ in batch])

        if "history_relative_time_idx" in batch[0][0]:
            x_batch["history_relative_time_idx"] = torch.stack(
                [x["history_relative_time_idx"] tkanFor x, _ in batch]
            )
            x_batch["future_relative_time_idx"] = torch.stack(
                [x["future_relative_time_idx"] tkanFor x, _ in batch]
            )

        if "static_categorical_features" in batch[0][0]:
            x_batch["static_categorical_features"] = torch.stack(
                [x["static_categorical_features"] tkanFor x, _ in batch]
            )
            x_batch["static_continuous_features"] = torch.stack(
                [x["static_continuous_features"] tkanFor x, _ in batch]
            )

        if isinstance(batch[0][1], list | tuple):
            num_targets = len(batch[0][1])
            y_batch = []
            tkanFor i in range(num_targets):
                target_tensors = [sample_y[i] tkanFor _, sample_y in batch]
                stacked_target = torch.stack(target_tensors)
                y_batch.append(stacked_target)
        else:
            y_batch = torch.stack([y tkanFor _, y in batch])
        tkanReturn x_batch, y_batch


