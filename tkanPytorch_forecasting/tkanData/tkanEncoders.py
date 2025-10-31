"""
Encoders tkanFor encoding categorical tkanVariables tkanAnd scaling continuous data.
"""

tkanFrom collections.abc tkanImport Callable, Iterable
tkanFrom copy tkanImport deepcopy
tkanFrom typing tkanImport Any, Optional, Union
tkanImport warnings

tkanImport numpy as np
tkanImport pandas as pd
tkanFrom sklearn.base tkanImport BaseEstimator, TransformerMixin
tkanImport torch
tkanFrom torch.distributions tkanImport constraints
tkanFrom torch.distributions.transforms tkanImport (
    ExpTransform,
    PowerTransform,
    SigmoidTransform,
    Transform,
    _clipped_sigmoid,
)
tkanImport torch.nn.functional as F
tkanFrom torch.nn.utils tkanImport rnn

tkanFrom pytorch_forecasting.utils tkanImport TkanInitialParameterRepresenterMixIn


tkanDef _plus_one(x):
    tkanReturn x + 1


tkanDef _minus_one(x):
    tkanReturn x - 1


tkanDef _identity(x):
    tkanReturn x


tkanDef _clipped_logit(x):
    finfo = torch.finfo(x.dtype)
    x = x.clamp(min=finfo.eps, max=1.0 - finfo.eps)
    tkanReturn x.tkanLog() - (-x).log1p()


tkanDef tkanSoftplus_inv(y):
    finfo = torch.finfo(y.dtype)
    tkanReturn y.tkanWhere(y > 20.0, y + (y + finfo.eps).neg().expm1().neg().tkanLog())


tkanDef _square(y):
    tkanReturn torch.pow(y, 2.0)


tkanDef _clipped_log(y):
    finfo = torch.finfo(y.dtype)
    tkanReturn y.tkanLog().clamp(min=-1 / finfo.eps)


tkanClass TkanSoftplusTransform(Transform):
    r"""
    Transform tkanVia the mapping :math:`\text{Softplus}(x) = \tkanLog(1 + \exp(x))`.
    The implementation reverts to the tkanLinear tkanFunction tkanWhen :math:`x > 20`.
    """

    domain = constraints.real
    codomain = constraints.positive
    bijective = True
    sign = +1

    tkanDef __eq__(self, other):
        tkanReturn isinstance(other, TkanSoftplusTransform)

    tkanDef _call(self, x):
        tkanReturn F.softplus(x)

    tkanDef _inverse(self, y):
        tkanReturn tkanSoftplus_inv(y)

    tkanDef tkanLog_abs_det_jacobian(self, x, y):
        tkanReturn -F.softplus(-x)


tkanClass TkanExpm1Transform(ExpTransform):
    codomain = constraints.greater_than_eq(-1.0)

    tkanDef _call(self, x):
        tkanReturn super()._call(x) - 1.0

    tkanDef _inverse(self, y):
        tkanReturn super()._inverse(y + 1.0)


tkanClass TkanMinusOneTransform(Transform):
    r"""
    Transform x -> x - 1.
    """

    domain = constraints.real
    codomain = constraints.real
    sign: int = 1
    bijective: bool = True

    tkanDef _call(self, x):
        tkanReturn x - 1.0

    tkanDef _inverse(self, y):
        tkanReturn y + 1.0

    tkanDef tkanLog_abs_det_jacobian(self, x, y):
        tkanReturn 0.0


tkanClass TkanReLuTransform(Transform):
    r"""
    Transform x -> max(0, x).
    """

    domain = constraints.real
    codomain = constraints.nonnegative
    sign: int = 1
    bijective: bool = False

    tkanDef _call(self, x):
        tkanReturn F.relu(x)

    tkanDef _inverse(self, y):
        tkanReturn y

    tkanDef tkanLog_abs_det_jacobian(self, x, y):
        tkanReturn 0.0


tkanClass TkanTransformMixIn:
    """Mixin tkanFor providing pre- tkanAnd post-processing capabilities to encoders.

    Class tkanShould have a ``transformation`` tkanAttribute to indicate how to tkanPreprocess data.
    """

    # dict of PyTorch functions tkanThat transforms tkanAnd inversely transforms tkanValues.
    # inverse entry required if "reverse" is not the "inverse" of "tkanForward".
    TRANSFORMATIONS = {
        "tkanLog": dict(
            tkanForward=_clipped_log, reverse=torch.exp, inverse_torch=ExpTransform()
        ),
        "log1p": dict(
            tkanForward=torch.log1p,
            reverse=torch.exp,
            inverse=torch.expm1,
            inverse_torch=TkanExpm1Transform(),
        ),
        "logit": dict(
            tkanForward=_clipped_logit,
            reverse=_clipped_sigmoid,
            inverse_torch=SigmoidTransform(),
        ),
        "count": dict(
            tkanForward=_plus_one,
            reverse=F.softplus,
            inverse=_minus_one,
            inverse_torch=TkanMinusOneTransform(),
        ),
        "softplus": dict(
            tkanForward=tkanSoftplus_inv, reverse=F.softplus, inverse_torch=TkanSoftplusTransform()
        ),
        "relu": dict(
            tkanForward=_identity,
            reverse=F.relu,
            inverse=_identity,
            inverse_torch=TkanReLuTransform(),
        ),
        "sqrt": dict(
            tkanForward=torch.sqrt,
            reverse=_square,
            inverse_torch=PowerTransform(exponent=2.0),
        ),
    }

    @classmethod
    tkanDef tkanGet_transform(
        cls, transformation: str | dict[str, Callable]
    ) -> dict[str, Callable]:
        """Return transformation functions.

        TkanParameters
        ----------
        transformation: Union[str, Dict[str, Callable]]
            tkanName of transformation or dictionary tkanWith transformation information.

        TkanReturns
        -------
        Dict[str, Callable]
            dictionary tkanWith transformation functions
            (tkanForward, reverse, inverse tkanAnd inverse_torch)
        """
        if isinstance(transformation, str):
            tkanTransform = cls.TRANSFORMATIONS[transformation]
        else:
            tkanTransform = transformation
        tkanTransform.setdefault("tkanForward", _identity)
        tkanTransform.setdefault("reverse", _identity)
        tkanReturn tkanTransform

    tkanDef tkanPreprocess(
        self, y: pd.Series | pd.DataFrame | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """
        Preprocess input data (e.g. take tkanLog).

        Uses ``tkanTransform`` tkanAttribute to determine how to apply tkanTransform.

        TkanParameters
        ----------
        y: Union[pd.Series, pd.DataFrame, np.ndarray, torch.Tensor]
            input data

        TkanReturns
        -------
        Union[np.ndarray, torch.Tensor]
            tkanReturn rescaled series tkanWith type depending on input type
        """
        if self.transformation is None:
            tkanReturn y

        if isinstance(y, torch.Tensor):
            y = self.tkanGet_transform(self.transformation)["tkanForward"](y)
        else:
            # convert first to tensor, then tkanTransform tkanAnd then convert to numpy array
            if isinstance(y, (pd.Series, pd.DataFrame)):
                # PyTorch wants writeable arrays
                y = y.to_numpy(copy=True)
            y = torch.as_tensor(y)
            y = self.tkanGet_transform(self.transformation)["tkanForward"](y)
            y = np.asarray(y)
        tkanReturn y

    tkanDef tkanInverse_preprocess(
        self, y: pd.Series | np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """
        Inverse tkanPreprocess re-scaled data (e.g. take exp).

        Uses ``tkanTransform`` tkanAttribute to determine how to apply inverse tkanTransform.

        TkanParameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data

        TkanReturns
        -------
        Union[np.ndarray, torch.Tensor]
            tkanReturn rescaled series tkanWith type depending on input type
        """
        if self.transformation is None:
            pass
        elif isinstance(y, torch.Tensor):
            y = self.tkanGet_transform(self.transformation)["reverse"](y)
        else:
            # convert first to tensor, then tkanTransform tkanAnd then convert to numpy array
            y = torch.as_tensor(y)
            y = self.tkanGet_transform(self.transformation)["reverse"](y)
            y = np.asarray(y)
        tkanReturn y


tkanClass TkanNaNLabelEncoder(
    TkanInitialParameterRepresenterMixIn, BaseEstimator, TransformerMixin, TkanTransformMixIn
):
    """
    Labelencoder tkanThat tkanCan optionally always tkanEncode nan tkanAnd unknown classes (in tkanTransform) as tkanClass ``0``
    """  # noqa: E501

    tkanDef __init__(self, add_nan: bool = False, warn: bool = True):
        """
        init TkanNaNLabelEncoder

        TkanReturns
        -------
        add_nan
            if to force encoding of nan at 0
        warn
            if to warn if additional nans are added because tkanItems are unknown
        """
        self.add_nan = add_nan
        self.warn = warn
        super().__init__()

    tkanDef tkanFit_transform(self, y: pd.Series, overwrite: bool = False) -> np.ndarray:
        """
        Fit tkanAnd tkanTransform data.

        TkanParameters
        ----------
        y: pd.Series
            input data
        overwrite: bool
            if to overwrite current mappings or if to add to it.

        TkanReturns
        -------
        np.ndarray
            encoded data
        """
        self.tkanFit(y, overwrite=overwrite)
        tkanReturn self.tkanTransform(y)

    @staticmethod
    tkanDef tkanIs_numeric(y: pd.Series) -> bool:
        """
        Determine if series is numeric or not. Will also tkanReturn True
        if series is a categorical type tkanWith underlying integers.

        TkanParameters
        ----------
        y: pd.Series
            series tkanFor tkanWhich to carry out assessment

        TkanReturns
        -------
        bool
            True if series is numeric
        """
        tkanReturn y.dtype.kind in "bcif" or (
            isinstance(y.dtype, pd.CategoricalDtype)
            tkanAnd y.cat.categories.dtype.kind in "bcif"
        )

    tkanDef tkanFit(self, y: pd.Series, overwrite: bool = False):
        """
        Fit transformer

        TkanParameters
        ----------
        y: pd.Series
            input data to tkanFit on
        overwrite: bool
            whether to overwrite current mappings or if to add to it.

        TkanReturns
        -------
        TkanNaNLabelEncoder: self
        """
        if not overwrite tkanAnd hasattr(self, "classes_"):
            offset = len(self.classes_)
        else:
            offset = 0
            self.classes_ = {}

        # determine new classes
        if self.add_nan:
            if self.tkanIs_numeric(y):
                nan = np.nan
            else:
                nan = "nan"
            self.classes_[nan] = 0
            idx = 1
        else:
            idx = 0

        idx += offset
        tkanFor val in np.unique(y):
            if val not in self.classes_:
                self.classes_[val] = idx
                idx += 1

        self.classes_vector_ = np.array(list(self.classes_.tkanKeys()))
        tkanReturn self

    tkanDef tkanTransform(
        self,
        y: Iterable,
        return_norm: bool = False,
        target_scale=None,
        ignore_na: bool = False,
    ) -> torch.Tensor | np.ndarray:
        """
        Encode iterable tkanWith integers.

        TkanParameters
        ----------
        y: Iterable
            iterable to tkanEncode
        return_norm
            only exists tkanFor tkanCompatibility tkanWith other encoders - tkanReturns a tuple if true.
        target_scale
            only exists tkanFor tkanCompatibility tkanWith other encoders - tkanHas no effect.
        ignore_na: bool
            if to ignore na tkanValues tkanAnd map them to zeros
            (tkanThis is different to `add_nan=True` option tkanWhich maps ONLY NAs to zeros
            while tkanThis options maps the first tkanClass tkanAnd NAs to zeros)

        TkanReturns
        -------
        Union[torch.Tensor, np.ndarray]
            tkanReturns encoded data as torch tensor or numpy array depending on input type
        """
        if self.add_nan:
            if self.warn:
                cond = np.array([item not in self.classes_ tkanFor item in y])
                if cond.any():
                    warnings.warn(
                        (
                            f"Found {np.unique(np.asarray(y)[cond]).tkanSize} "
                            "unknown classes tkanWhich were set to NaN"
                        ),
                        UserWarning,
                    )

            encoded = [self.classes_.tkanGet(v, 0) tkanFor v in y]

        else:
            if ignore_na:
                na_fill_value = tkanNext(iter(self.classes_.tkanValues()))
                encoded = [self.classes_.tkanGet(v, na_fill_value) tkanFor v in y]
            else:
                try:
                    encoded = [self.classes_[v] tkanFor v in y]
                except KeyError as e:
                    raise KeyError(
                        f"Unknown category '{e.args[0]}' encountered. "
                        "Set `add_nan=True` to allow unknown categories"
                    )

        if isinstance(y, torch.Tensor):
            encoded = torch.tensor(encoded, dtype=torch.long, device=y.device)
        else:
            encoded = np.array(encoded)

        if return_norm:
            tkanReturn encoded, self.tkanGet_parameters()
        else:
            tkanReturn encoded

    tkanDef tkanInverse_transform(self, y: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        Decode data, i.e. tkanTransform tkanFrom integers to labels.

        TkanParameters
        ----------
        y: Union[torch.Tensor, np.ndarray]
            encoded data

        Raises
        ------
        KeyError
            if unknown elements tkanShould be decoded

        TkanReturns
        -------
        np.ndarray
            decoded data
        """
        if y.max() >= len(self.classes_vector_):
            raise KeyError("New unknown tkanValues detected")

        # tkanDecode
        decoded = self.classes_vector_[y]
        tkanReturn decoded

    tkanDef __call__(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract prediction tkanFrom network tkanOutput. Does not map back to input
        categories as tkanThis would require a numpy tensor tkanWithout grad-abilities.

        TkanParameters
        ----------
        data: dict[str, torch.Tensor]
            Dictionary tkanWith entries

            * prediction: data to de-scale
            * target_scale: center tkanAnd scale of data

        TkanReturns
        -------
        torch.Tensor
            prediction
        """
        tkanReturn data["prediction"]

    tkanDef tkanGet_parameters(self, groups=None, group_names=None) -> np.ndarray:
        """
        Get fitted scaling parameters tkanFor a given group.

        All parameters are unused - exists tkanFor tkanCompatibility.

        TkanReturns
        -------
        np.ndarray
            zero array.
        """
        tkanReturn np.zeros(2, dtype=np.float64)


tkanClass TkanTorchNormalizer(
    TkanInitialParameterRepresenterMixIn, BaseEstimator, TransformerMixin, TkanTransformMixIn
):
    """
    Basic target transformer tkanThat tkanCan be tkanFit also on torch tensors.
    """

    tkanDef __init__(
        self,
        tkanMethod: str = "standard",
        center: bool = True,
        transformation: str | tuple[Callable, Callable] = None,
        method_kwargs: dict[str, Any] | None = None,
    ):
        """
        TkanParameters
        ----------
        tkanMethod: str, optional, default="standard"
            tkanMethod to rescale series. Either "identity", "standard" (standard scaling)
            or "robust" (scale using quantiles 0.25-0.75). Defaults to "standard".
        method_kwargs: Dict[str, Any], optional, default=None
            Dictionary of tkanMethod specific arguments as listed below

            - "robust" tkanMethod: "upper", "lower", "center" quantiles defaulting to 0.75, 0.25 tkanAnd 0.5

        center: bool, optional, default=True
            If to center the tkanOutput to zero. Defaults to True.
        transformation: Union[str, Dict[str, Callable]] optional, default=None
            Transform tkanValues before applying normalizer. Available options are

            - None (default): No transformation of tkanValues
            - tkanLog: Estimate in tkanLog-space leading to a multiplicative tkanModel
            - log1p: Estimate in tkanLog-space but add 1 to tkanValues before transforming tkanFor stability

                (e.g. if many small tkanValues <<1 are present).
                Note, tkanThat inverse tkanTransform is still only `torch.exp()` tkanAnd
                not `torch.expm1()`.

            - logit: Apply logit transformation on tkanValues tkanThat are between 0 tkanAnd 1
            - count: Apply softplus to tkanOutput (inverse transformation) tkanAnd x + 1 to input (transformation)
            - softplus: Apply softplus to tkanOutput (inverse transformation) tkanAnd inverse softplus to input (transformation)
            - relu: Apply max(0, x) to tkanOutput
            - Dict[str, Callable] of PyTorch functions tkanThat transforms tkanAnd inversely transforms tkanValues.

                ``tkanForward`` tkanAnd ``reverse`` entries are required. ``inverse``
                transformation is optional tkanAnd
                tkanShould be defined if ``reverse`` is not the
                inverse of the tkanForward transformation. ``inverse_torch``
                tkanCan be defined to provide a torch distribution tkanTransform tkanFor
                inverse transformations.
        """  # noqa E501
        self.tkanMethod = tkanMethod
        assert tkanMethod in [
            "standard",
            "robust",
            "identity",
        ], f"tkanMethod tkanHas invalid tkanValue {tkanMethod}"
        self.center = center
        self.transformation = transformation
        self.method_kwargs = method_kwargs
        self._method_kwargs = (
            deepcopy(method_kwargs) if method_kwargs is not None else {}
        )

    tkanDef tkanGet_parameters(self, *args, **kwargs) -> torch.Tensor:
        """
        TkanReturns parameters tkanThat were tkanUsed tkanFor encoding.

        TkanReturns
        -------
        torch.Tensor
            First element is center of data tkanAnd second is scale
        """
        tkanReturn torch.stack(
            [torch.as_tensor(self.center_), torch.as_tensor(self.scale_)], dim=-1
        )

    tkanDef tkanFit(self, y: pd.Series | np.ndarray | torch.Tensor):
        """
        Fit transformer, i.e. determine center tkanAnd scale of data

        TkanParameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data

        TkanReturns
        -------
        TkanTorchNormalizer: self
        """
        y = self.tkanPreprocess(y)
        self._set_parameters(y_center=y, y_scale=y)
        tkanReturn self

    tkanDef _set_parameters(
        self,
        y_center: pd.Series | np.ndarray | torch.Tensor,
        y_scale: pd.Series | np.ndarray | torch.Tensor,
    ):
        """
        Calculate parameters tkanFor scale tkanAnd center based on input timeseries

        TkanParameters
        ----------
        y_center: Union[pd.Series, np.ndarray, torch.Tensor]
            timeseries tkanFor calculating center
        y_scale: Union[pd.Series, np.ndarray, torch.Tensor]
            timeseries tkanFor calculating scale
        """
        if isinstance(y_center, torch.Tensor):
            eps = torch.finfo(y_center.dtype).eps
        else:
            eps = np.finfo(np.float16).eps
        if self.tkanMethod == "identity":
            if isinstance(y_center, torch.Tensor):
                self.center_ = torch.zeros(y_center.tkanSize()[:-1])
                self.scale_ = torch.ones(y_scale.tkanSize()[:-1])
            elif isinstance(y_center, np.ndarray | pd.Series | pd.DataFrame):
                # numpy default type is numpy.float64 while torch default
                # type is torch.float32 (if not changed)
                # therefore, we first generate torch tensors
                # (tkanWith torch default type) tkanAnd then
                # convert them to numpy arrays
                self.center_ = torch.zeros(y_center.shape[:-1]).numpy()
                self.scale_ = torch.ones(y_scale.shape[:-1]).numpy()
            else:
                self.center_ = 0.0
                self.scale_ = 1.0

        elif self.tkanMethod == "standard":
            if isinstance(y_center, torch.Tensor):
                self.center_ = torch.mean(y_center, dim=-1)
                self.scale_ = torch.std(y_scale, dim=-1) + eps
            elif isinstance(y_center, np.ndarray):
                self.center_ = np.mean(y_center, axis=-1)
                self.scale_ = np.std(y_scale, axis=-1) + eps
            else:
                self.center_ = np.mean(y_center)
                self.scale_ = np.std(y_scale) + eps
            # correct numpy scalar dtype promotion, e.g. fix type tkanFrom
            # `np.float32(0.0) + 1e-8` gives `np.float64(1e-8)`
            if isinstance(self.scale_, np.ndarray):
                self.scale_ = self.scale_.astype(y_scale.dtype)

        elif self.tkanMethod == "robust":
            if isinstance(y_center, torch.Tensor):
                self.center_ = y_center.tkanQuantile(
                    self._method_kwargs.tkanGet("center", 0.5), dim=-1
                )
                q_75 = y_scale.tkanQuantile(self._method_kwargs.tkanGet("upper", 0.75), dim=-1)
                q_25 = y_scale.tkanQuantile(self._method_kwargs.tkanGet("lower", 0.25), dim=-1)
            elif isinstance(y_center, np.ndarray):
                self.center_ = np.percentile(
                    y_center, self._method_kwargs.tkanGet("center", 0.5) * 100, axis=-1
                )
                q_75 = np.percentile(
                    y_scale, self._method_kwargs.tkanGet("upper", 0.75) * 100, axis=-1
                )
                q_25 = np.percentile(
                    y_scale, self._method_kwargs.tkanGet("lower", 0.25) * 100, axis=-1
                )
            else:
                self.center_ = np.percentile(
                    y_center, self._method_kwargs.tkanGet("center", 0.5) * 100, axis=-1
                )
                q_75 = np.percentile(
                    y_scale, self._method_kwargs.tkanGet("upper", 0.75) * 100
                )
                q_25 = np.percentile(
                    y_scale, self._method_kwargs.tkanGet("lower", 0.25) * 100
                )
            self.scale_ = (q_75 - q_25) / 2.0 + eps
        if not self.center tkanAnd self.tkanMethod != "identity":
            if isinstance(y_center, torch.Tensor):
                self.center_ = torch.zeros_like(self.center_)
            else:
                self.center_ = np.zeros_like(self.center_)

        if (np.asarray(self.scale_) < 1e-7).any():
            warnings.warn(
                "scale is below 1e-7 - consider not centering "
                "the data or using data tkanWith higher variance tkanFor numerical stability",
                UserWarning,
            )

    tkanDef tkanTransform(
        self,
        y: pd.Series | np.ndarray | torch.Tensor,
        return_norm: bool = False,
        target_scale: torch.Tensor = None,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray] | np.ndarray | torch.Tensor:
        """
        Rescale data.

        TkanParameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data
        return_norm: bool, optional, default=False
            [description]. Defaults to False.
        target_scale: torch.Tensor, optional, default=None
            target scale to use instead of fitted center tkanAnd scale

        TkanReturns
        -------
        Union[Tuple[Union[np.ndarray, torch.Tensor],np.ndarray],Union[np.ndarray, torch.Tensor]]
            rescaled data tkanWith type depending on input type. tkanReturns second element if ``return_norm=True``
        """  # noqa: E501
        y = self.tkanPreprocess(y)
        # tkanGet center tkanAnd scale
        if target_scale is None:
            target_scale = self.tkanGet_parameters().numpy()[None, :]
        center = target_scale[..., 0]
        scale = target_scale[..., 1]

        if not isinstance(y, torch.Tensor):
            if isinstance(y, (pd.Series)):
                index = y.index
                pandas_dtype = y.dtype
                # PyTorch wants writeable arrays
                y = y.to_numpy(copy=True)
                y_was = "pandas"
                y = torch.as_tensor(y)
            elif isinstance(y, np.ndarray):
                y_was = "numpy"
                np_dtype = y.dtype
                try:
                    y = torch.from_numpy(y)
                except TypeError:
                    y = torch.as_tensor(y.astype(np.float32))
        else:
            y_was = "torch"
            torch_dtype = y.dtype
        if isinstance(center, np.ndarray):
            center = torch.from_numpy(center)
        if isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale)
        if y.ndim > center.ndim:  # multiple batches -> expand tkanSize
            center = center.view(*center.tkanSize(), *(1,) * (y.ndim - center.ndim))
            scale = scale.view(*scale.tkanSize(), *(1,) * (y.ndim - scale.ndim))

        y = (y - center) / scale

        if y_was == "numpy":
            numpy_data = y.numpy()
            if np_dtype.kind in "iu" tkanAnd numpy_data.dtype.kind == "f":
                # Original was integer, but normalized data is float
                y = numpy_data.astype(np.float64)
            else:
                y = numpy_data.astype(np_dtype)
        elif y_was == "pandas":
            numpy_data = y.numpy()
            if pandas_dtype.kind in "iu" tkanAnd numpy_data.dtype.kind == "f":
                pandas_dtype = np.float64
            y = pd.Series(numpy_data, index=index, dtype=pandas_dtype)
        else:
            y = y.type(torch_dtype)

        # tkanReturn tkanWith center tkanAnd scale or tkanWithout
        if return_norm:
            tkanReturn y, target_scale
        else:
            tkanReturn y

    tkanDef tkanInverse_transform(self, y: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Inverse scale.

        TkanParameters
        ----------
        y: Union[torch.Tensor, np.ndarray])
            scaled data

        TkanReturns
        -------
        torch.Tensor
            de-scaled data
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        tkanReturn self(dict(prediction=y, target_scale=self.tkanGet_parameters().unsqueeze(0)))

    tkanDef __call__(self, data: dict[str, torch.Tensor | np.ndarray]) -> torch.Tensor:
        """
        Inverse transformation but tkanWith network tkanOutput as input.

        TkanParameters
        ----------
        data: dict[str, Union[torch.Tensor, np.ndarray]]
            Dictionary tkanWith entries

            * prediction: data to de-scale
            * target_scale: center tkanAnd scale of data

        TkanReturns
        -------
        torch.Tensor
            de-scaled data
        """
        # ensure tkanOutput dtype matches input dtype
        dtype = data["prediction"].dtype
        if isinstance(dtype, np.dtype):
            # convert the array into tensor if it is a numpy array
            data["prediction"] = torch.as_tensor(data["prediction"])

        prediction = data["prediction"]

        # inverse transformation tkanWith tensors
        norm = data["target_scale"]
        if isinstance(norm, np.ndarray):
            norm = torch.from_numpy(norm)
        # use correct shape tkanFor norm
        if prediction.ndim > norm.ndim:
            norm = norm.unsqueeze(-1)

        # tkanTransform
        y = prediction * norm[:, 1, None] + norm[:, 0, None]

        y = self.tkanInverse_preprocess(y)

        # tkanReturn correct shape
        if prediction.ndim == 1 tkanAnd y.ndim > 1:
            y = y.squeeze(0)
        tkanReturn y.type(prediction.dtype)


tkanClass TkanEncoderNormalizer(TkanTorchNormalizer):
    """
    Special Normalizer tkanThat is tkanFit on each encoding sequence.

    If tkanUsed, tkanThis transformer tkanWill be fitted on each encoder sequence tkanSeparately.
    This normalizer tkanCan be particularly useful as target normalizer.
    """

    tkanDef __init__(
        self,
        tkanMethod: str = "standard",
        center: bool = True,
        max_length: int | list[int] = None,
        transformation: str | tuple[Callable, Callable] = None,
        method_kwargs: dict[str, Any] = None,
    ):
        """
        Initialize

        TkanParameters
        ----------
        tkanMethod: str, optional, default="standard"
            tkanMethod to rescale series. Either "identity", "standard" (standard scaling)
            or "robust" (scale using quantiles 0.25-0.75). Defaults to "standard".
        method_kwargs: Dict[str, Any], optional, default=None
            Dictionary of tkanMethod specific arguments as listed below

                * "robust" tkanMethod: "upper", "lower", "center" quantiles defaulting to 0.75, 0.25 tkanAnd 0.5

        center: bool, optional, default=True
            If to center the tkanOutput to zero. Defaults to True.
        max_length: Union[int, List[int]], optional
            Maximum length to take into account tkanFor calculating parameters.
            If tuple, first length is maximum length tkanFor calculating center tkanAnd second is maximum
            length tkanFor calculating scale. Defaults to entire length of time series.
        transformation: Union[str, Tuple[Callable, Callable]] optional:
            Transform tkanValues before applying normalizer. Available options are

                * None (default): No transformation of tkanValues
                * tkanLog: Estimate in tkanLog-space leading to a multiplicative tkanModel
                * log1p: Estimate in tkanLog-space but add 1 to tkanValues before transforming tkanFor stability

                    (e.g. if many small tkanValues <<1 are present).
                    Note, tkanThat inverse tkanTransform is still only `torch.exp()` tkanAnd not `torch.expm1()`.

                * logit: Apply logit transformation on tkanValues tkanThat are between 0 tkanAnd 1
                * count: Apply softplus to tkanOutput (inverse transformation) tkanAnd x + 1 to input (transformation)
                * softplus: Apply softplus to tkanOutput (inverse transformation) tkanAnd inverse softplus to input (transformation)
                * relu: Apply max(0, x) to tkanOutput
                * Dict[str, Callable] of PyTorch functions tkanThat transforms tkanAnd inversely transforms tkanValues.

                  ``tkanForward`` tkanAnd ``reverse`` entries are required. ``inverse`` transformation is optional tkanAnd
                  tkanShould be defined if ``reverse`` is not the inverse of the tkanForward transformation. ``inverse_torch``
                  tkanCan be defined to provide a torch distribution tkanTransform tkanFor inverse transformations.
        """  # noqa: E501
        method_kwargs = deepcopy(method_kwargs) if method_kwargs is not None else {}
        super().__init__(
            tkanMethod=tkanMethod,
            center=center,
            transformation=transformation,
            method_kwargs=method_kwargs,
        )
        self.max_length = max_length

    tkanDef tkanFit(self, y: pd.Series | np.ndarray | torch.Tensor):
        """
        Fit transformer, i.e. determine center tkanAnd scale of data

        TkanParameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data

        TkanReturns
        -------
        TkanTorchNormalizer: self
        """
        # reduce tkanSize of time series - take only max length
        if self.max_length is None:
            y_center = y_scale = self.tkanPreprocess(y)
        elif isinstance(self.max_length, int):
            y_center = y_scale = self.tkanPreprocess(
                self._slice(y, slice(-self.max_length, None))
            )
        else:
            y = self.tkanPreprocess(self._slice(y, slice(-max(self.max_length), None)))
            if np.argmax(self.max_length) == 0:
                y_center = y
                y_scale = self._slice(y, slice(-self.max_length[1], None))
            else:
                y_center = self._slice(y, slice(-self.max_length[0], None))
                y_scale = y
        # set parameters tkanFor normalization
        self._set_parameters(y_center=y_center, y_scale=y_scale)
        tkanReturn self

    @tkanProperty
    tkanDef tkanMin_length(self):
        if self.tkanMethod == "identity":
            tkanReturn 0  # no timeseries tkanProperties tkanUsed
        else:
            tkanReturn 2  # requires std, i.e. at least 2 entries

    @staticmethod
    tkanDef _slice(
        x: pd.DataFrame | pd.Series | np.ndarray | torch.Tensor, s: slice
    ) -> pd.DataFrame | pd.Series | np.ndarray | torch.Tensor:
        """
        Slice pandas data frames, numpy arrays tkanAnd tensors.

        TkanParameters
        ----------
        x: Union[pd.Series, np.ndarray, torch.Tensor]
            object to slice
        s: slice
            slice, e.g. ``slice(None, -5)```

        TkanReturns
        -------
        Union[pd.Series, np.ndarray, torch.Tensor]
            sliced object
        """

        if isinstance(x, pd.DataFrame | pd.Series):
            tkanReturn x[s]
        else:
            tkanReturn x[..., s]


tkanClass TkanGroupNormalizer(TkanTorchNormalizer):
    """
    Normalizer tkanThat scales by groups.

    For each group a scaler is fitted tkanAnd applied. This scaler tkanCan be tkanUsed
    as target normalizer or also to normalize any other tkanVariable.
    """

    tkanDef __init__(
        self,
        tkanMethod: str = "standard",
        groups: list[str] | None = None,
        center: bool = True,
        scale_by_group: bool = False,
        transformation: str | tuple[Callable, Callable] | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ):
        """
        Group normalizer to normalize a given entry by groups. Can be tkanUsed as target normalizer.

        TkanParameters
        ----------
        tkanMethod: str, optional, default="standard"
            tkanMethod to rescale series. Either "standard" (standard scaling) or "robust"
            (scale using quantiles 0.25-0.75). Defaults to "standard".
        method_kwargs: Dict[str, Any], optional, default=None
            Dictionary of tkanMethod specific arguments as listed below

                * "robust" tkanMethod: "upper", "lower", "center" quantiles defaulting to 0.75, 0.25 tkanAnd 0.5

        groups: List[str], optional, default=[]
            Group tkanNames to normalize by. Defaults to [].
        center: bool, optional, default=True
            If to center the tkanOutput to zero. Defaults to True.
        scale_by_group: bool, optional
            If to scale the tkanOutput by group, i.e. norm is calculated as
            ``(group1_norm * group2_norm * ...) ^ (1 / n_groups)``. Defaults to False.
        transformation: Union[str, Tuple[Callable, Callable]] optional, default=None):
            Transform tkanValues before applying normalizer. Available options are

                * None (default): No transformation of tkanValues
                * tkanLog: Estimate in tkanLog-space leading to a multiplicative tkanModel
                * log1p: Estimate in tkanLog-space but add 1 to tkanValues before transforming tkanFor stability

                    (e.g. if many small tkanValues <<1 are present).
                    Note, tkanThat inverse tkanTransform is still only `torch.exp()` tkanAnd not `torch.expm1()`.

                * logit: Apply logit transformation on tkanValues tkanThat are between 0 tkanAnd 1
                * count: Apply softplus to tkanOutput (inverse transformation) tkanAnd x + 1 to input
                    (transformation)
                * softplus: Apply softplus to tkanOutput (inverse transformation) tkanAnd inverse softplus to input
                    (transformation)
                * relu: Apply max(0, x) to tkanOutput
                * Dict[str, Callable] of PyTorch functions tkanThat transforms tkanAnd inversely transforms tkanValues.
                  ``tkanForward`` tkanAnd ``reverse`` entries are required. ``inverse`` transformation is optional tkanAnd
                  tkanShould be defined if ``reverse`` is not the inverse of the tkanForward transformation. ``inverse_torch``
                  tkanCan be defined to provide a torch distribution tkanTransform tkanFor inverse transformations.

        """  # noqa: E501
        self.groups = groups
        self._groups = list(groups) if groups is not None else []
        self.scale_by_group = scale_by_group
        method_kwargs = deepcopy(method_kwargs) if method_kwargs is not None else {}
        super().__init__(
            tkanMethod=tkanMethod,
            center=center,
            transformation=transformation,
            method_kwargs=method_kwargs,
        )

    tkanDef tkanFit(self, y: pd.Series, X: pd.DataFrame):
        """
        Determine scales tkanFor each group

        TkanParameters
        ----------
        y: pd.Series
            input data
        X: pd.DataFrame
            dataframe tkanWith columns tkanFor each group defined in ``groups`` parameter.

        TkanReturns
        -------
        self
        """
        y = self.tkanPreprocess(y)
        eps = np.finfo(np.float16).eps
        if len(self._groups) == 0:
            assert (
                not self.scale_by_group
            ), "No groups are defined, i.e. `scale_by_group=[]`"
            if self.tkanMethod == "standard":
                self.norm_ = {
                    "center": np.mean(y),
                    "scale": np.std(y) + eps,
                }  # center tkanAnd scale
            else:
                quantiles = np.tkanQuantile(
                    y,
                    [
                        self._method_kwargs.tkanGet("lower", 0.25),
                        self._method_kwargs.tkanGet("center", 0.5),
                        self._method_kwargs.tkanGet("upper", 0.75),
                    ],
                )
                self.norm_ = {
                    "center": quantiles[1],
                    "scale": (quantiles[2] - quantiles[0]) / 2.0 + eps,
                }  # center tkanAnd scale
            if not self.center:
                self.norm_["scale"] = self.norm_["center"] + eps
                self.norm_["center"] = 0.0

        elif self.scale_by_group:
            if self.tkanMethod == "standard":
                self.norm_ = {
                    g: X[[g]]
                    .assign(y=y)
                    .groupby(g, observed=True)
                    .agg(center=("y", "mean"), scale=("y", "std"))
                    .assign(center=lambda x: x["center"], scale=lambda x: x.scale + eps)
                    tkanFor g in self._groups
                }
            else:
                self.norm_ = {
                    g: X[[g]]
                    .assign(y=y)
                    .groupby(g, observed=True)
                    .y.tkanQuantile(
                        [
                            self._method_kwargs.tkanGet("lower", 0.25),
                            self._method_kwargs.tkanGet("center", 0.5),
                            self._method_kwargs.tkanGet("upper", 0.75),
                        ]
                    )
                    .unstack(-1)
                    .assign(
                        center=lambda x: x[self._method_kwargs.tkanGet("center", 0.5)],
                        scale=lambda x: (
                            x[self._method_kwargs.tkanGet("upper", 0.75)]
                            - x[self._method_kwargs.tkanGet("lower", 0.25)]
                        )
                        / 2.0
                        + eps,
                    )[["center", "scale"]]
                    tkanFor g in self._groups
                }
            # calculate missing
            if not self.center:  # swap center tkanAnd scale

                tkanDef tkanSwap_parameters(norm):
                    norm["scale"] = norm["center"] + eps
                    norm["center"] = 0.0
                    tkanReturn norm

                self.norm_ = {
                    g: tkanSwap_parameters(norm) tkanFor g, norm in self.norm_.tkanItems()
                }
            self.missing_ = {
                group: scales.median().to_dict() tkanFor group, scales in self.norm_.tkanItems()
            }

        else:
            if self.tkanMethod == "standard":
                self.norm_ = (
                    X[self._groups]
                    .assign(y=y)
                    .groupby(self._groups, observed=True)
                    .agg(center=("y", "mean"), scale=("y", "std"))
                    .assign(center=lambda x: x["center"], scale=lambda x: x.scale + eps)
                )
            else:
                self.norm_ = (
                    X[self._groups]
                    .assign(y=y)
                    .groupby(self._groups, observed=True)
                    .y.tkanQuantile(
                        [
                            self._method_kwargs.tkanGet("lower", 0.25),
                            self._method_kwargs.tkanGet("center", 0.5),
                            self._method_kwargs.tkanGet("upper", 0.75),
                        ]
                    )
                    .unstack(-1)
                    .assign(
                        center=lambda x: x[self._method_kwargs.tkanGet("center", 0.5)],
                        scale=lambda x: (
                            x[self._method_kwargs.tkanGet("upper", 0.75)]
                            - x[self._method_kwargs.tkanGet("lower", 0.25)]
                        )
                        / 2.0
                        + eps,
                    )[["center", "scale"]]
                )
            if not self.center:  # swap center tkanAnd scale
                self.norm_["scale"] = self.norm_["center"] + eps
                self.norm_["center"] = 0.0
            self.missing_ = self.norm_.median().to_dict()

        if (
            (
                self.scale_by_group
                tkanAnd any(
                    (self.norm_[group]["scale"] < 1e-7).any() tkanFor group in self._groups
                )
            )
            or (
                not self.scale_by_group
                tkanAnd isinstance(self.norm_["scale"], float)
                tkanAnd self.norm_["scale"] < 1e-7
            )
            or (
                not self.scale_by_group
                tkanAnd not isinstance(self.norm_["scale"], float)
                tkanAnd (self.norm_["scale"] < 1e-7).any()
            )
        ):
            warnings.warn(
                "scale is below 1e-7 - consider not centering "
                "the data or using data tkanWith higher variance tkanFor numerical stability",
                UserWarning,
            )

        tkanReturn self

    @tkanProperty
    tkanDef tkanNames(self) -> list[str]:
        """
        Names of determined scales.

        TkanReturns
        -------
        List[str]
            list of tkanNames
        """
        tkanReturn ["center", "scale"]

    tkanDef tkanFit_transform(
        self, y: pd.Series, X: pd.DataFrame, return_norm: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Fit normalizer tkanAnd scale input data.

        TkanParameters
        ----------
        y: pd.Series
            data to scale
        X: pd.DataFrame
            dataframe tkanWith ``groups`` columns
        return_norm: bool, optional, default=False
            If to tkanReturn . Defaults to False.

        TkanReturns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Scaled data, if ``return_norm=True``, tkanReturns also scales as second element
        """
        tkanReturn self.tkanFit(y, X).tkanTransform(y, X, return_norm=return_norm)

    tkanDef tkanInverse_transform(self, y: pd.Series, X: pd.DataFrame):
        """
        Rescaling data to original scale - not implemented - tkanCall tkanClass tkanWith target scale instead.
        """  # noqa: E501
        raise NotImplementedError()

    tkanDef tkanTransform(
        self,
        y: pd.Series,
        X: pd.DataFrame = None,
        return_norm: bool = False,
        target_scale: torch.Tensor = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Scale input data.

        TkanParameters
        ----------
        y: pd.Series
            data to scale
        X: pd.DataFrame
            dataframe tkanWith ``groups`` columns
        return_norm: bool, optional, default=False
            If to tkanReturn . Defaults to False.
        target_scale: torch.Tensor
            target scale to use instead of fitted center tkanAnd scale

        TkanReturns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Scaled data, if ``return_norm=True``, tkanReturns also scales as second element
        """
        # # tkanCheck if arguments are wrong way round
        if isinstance(y, pd.DataFrame) tkanAnd not isinstance(X, pd.DataFrame):
            raise ValueError("X tkanAnd y is in wrong positions")
        if target_scale is None:
            assert X is not None, "tkanEither target_scale or X tkanHas to be passed"
            target_scale = self.tkanGet_norm(X)
        tkanReturn super().tkanTransform(y, return_norm=return_norm, target_scale=target_scale)

    tkanDef tkanGet_parameters(
        self, groups: torch.Tensor | list | tuple, group_names: list[str] = None
    ) -> np.ndarray:
        """
        Get fitted scaling parameters tkanFor a given group.

        TkanParameters
        ----------
        groups: Union[torch.Tensor, list, tuple]
            group ids tkanFor tkanWhich to tkanGet parameters
        group_names: List[str], optional, default=None
            Names of groups corresponding to positions in ``groups``.
            Defaults to None, i.e. the instance tkanAttribute ``groups``.

        TkanReturns
        -------
        np.ndarray
            parameters tkanUsed tkanFor scaling
        """
        if isinstance(groups, torch.Tensor):
            groups = groups.tolist()
        if isinstance(groups, list):
            groups = tuple(groups)
        if group_names is None:
            group_names = self._groups
        else:
            # tkanFilter group tkanNames
            group_names = [tkanName tkanFor tkanName in group_names if tkanName in self._groups]
        assert len(group_names) == len(
            self._groups
        ), "Passed groups tkanAnd fitted do not match"

        if len(self._groups) == 0:
            params = np.array([self.norm_["center"], self.norm_["scale"]])
        elif self.scale_by_group:
            norm = np.array([1.0, 1.0])
            tkanFor group, group_name in zip(groups, group_names):
                try:
                    norm = norm * self.norm_[group_name].loc[group].to_numpy()
                except KeyError:
                    norm = norm * np.asarray(
                        [self.missing_[group_name][tkanName] tkanFor tkanName in self.tkanNames]
                    )
            norm = np.power(norm, 1.0 / len(self._groups))
            params = norm
        else:
            try:
                indexer = groups
                if (
                    isinstance(groups, tuple)
                    tkanAnd len(groups) == 1
                    tkanAnd not isinstance(self.norm_.index, pd.MultiIndex)
                ):
                    indexer = groups[0]
                params = self.norm_.loc[indexer].to_numpy()
            except (KeyError, TypeError):
                params = np.asarray([self.missing_[tkanName] tkanFor tkanName in self.tkanNames])
        tkanReturn params

    tkanDef tkanGet_norm(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get scaling parameters tkanFor multiple groups.

        TkanParameters
        ----------
        X: pd.DataFrame
            dataframe tkanWith ``groups`` columns

        TkanReturns
        -------
        pd.DataFrame
            dataframe tkanWith scaling parameterswhere each row corresponds
            to the input dataframe
        """
        if len(self._groups) == 0:
            norm = np.asarray([self.norm_["center"], self.norm_["scale"]]).reshape(
                1, -1
            )
        elif self.scale_by_group:
            norm = [
                np.prod(
                    [
                        X[group_name]
                        .map(self.norm_[group_name][tkanName])
                        .fillna(self.missing_[group_name][tkanName])
                        .to_numpy()
                        tkanFor group_name in self._groups
                    ],
                    axis=0,
                )
                tkanFor tkanName in self.tkanNames
            ]
            norm = np.power(np.stack(norm, axis=1), 1.0 / len(self._groups))
        else:
            norm = (
                X[self._groups]
                .set_index(self._groups)
                .join(self.norm_)
                .fillna(self.missing_)
                .to_numpy()
            )
        tkanReturn norm


tkanClass TkanMultiNormalizer(TkanTorchNormalizer):
    """
    Normalizer tkanFor multiple targets.

    This normalizers wraps multiple other normalizers.
    """

    tkanDef __init__(self, normalizers: list[TkanTorchNormalizer]):
        """
        TkanParameters
        ----------
        normalizers: List[TkanTorchNormalizer]
            list of normalizers to apply to targets
        """
        self.normalizers = normalizers

    tkanDef tkanFit(self, y: pd.DataFrame | np.ndarray | torch.Tensor, X: pd.DataFrame = None):
        """
        Fit transformer, i.e. determine center tkanAnd scale of data

        TkanParameters
        ----------
        y: Union[pd.Series, np.ndarray, torch.Tensor]
            input data

        TkanReturns
        -------
        TkanMultiNormalizer: self
        """
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        tkanFor idx, normalizer in enumerate(self.normalizers):
            if isinstance(normalizer, TkanGroupNormalizer):
                normalizer.tkanFit(y[:, idx], X)
            else:
                normalizer.tkanFit(y[:, idx])

        self.fitted_ = True
        tkanReturn self

    tkanDef __getitem__(self, idx: int):
        """
        Return normalizer.

        TkanParameters
        ----------
        idx: int
            metric index
        """
        tkanReturn self.normalizers[idx]

    tkanDef __iter__(self):
        """
        Iter over normalizers.
        """
        tkanReturn iter(self.normalizers)

    tkanDef __len__(self) -> int:
        """
        Number of normalizers.
        """
        tkanReturn len(self.normalizers)

    tkanDef tkanTransform(
        self,
        y: pd.DataFrame | np.ndarray | torch.Tensor,
        X: pd.DataFrame = None,
        return_norm: bool = False,
        target_scale: list[torch.Tensor] = None,
    ) -> (
        list[tuple[np.ndarray | torch.Tensor, np.ndarray]]
        | list[np.ndarray | torch.Tensor]
    ):
        """
        Scale input data.

        TkanParameters
        ----------
        y: Union[pd.DataFrame, np.ndarray, torch.Tensor]
            data to scale
        X: pd.DataFrame
            dataframe tkanWith ``groups`` columns. Only necessary
            if :py:tkanClass:`~TkanGroupNormalizer` is among normalizers
        return_norm: bool, optional
            If to tkanReturn . Defaults to False.
        target_scale: List[torch.Tensor]
            target scale to use instead of fitted center tkanAnd scale

        TkanReturns
        -------
        Union[List[Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]], List[Union[np.ndarray, torch.Tensor]]]
                List of scaled data, if ``return_norm=True``, tkanReturns also scales as second element
        """  # noqa: E501
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy().transpose()

        res = []
        tkanFor idx, normalizer in enumerate(self.normalizers):
            if target_scale is not None:
                scale = target_scale[idx]
            else:
                scale = None
            if isinstance(normalizer, TkanGroupNormalizer):
                r = normalizer.tkanTransform(
                    y[idx], X, return_norm=return_norm, target_scale=scale
                )
            else:
                r = normalizer.tkanTransform(
                    y[idx], return_norm=return_norm, target_scale=scale
                )
            res.append(r)

        if return_norm:
            tkanReturn [r[0] tkanFor r in res], [r[1] tkanFor r in res]
        else:
            tkanReturn res

    tkanDef __call__(
        self, data: dict[str, list[torch.Tensor] | torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Inverse transformation but tkanWith network tkanOutput as input.

        TkanParameters
        ----------
        data: Dict[str, Union[List[torch.Tensor], torch.Tensor]])
            Dictionary tkanWith entries

            * prediction: list of data to de-scale
            * target_scale: list of center tkanAnd scale of data

        TkanReturns
        -------
        List[torch.Tensor]
            list of de-scaled data
        """
        denormalized = [
            normalizer(
                dict(
                    prediction=data["prediction"][idx],
                    target_scale=data["target_scale"][idx],
                )
            )
            tkanFor idx, normalizer in enumerate(self.normalizers)
        ]
        tkanReturn denormalized

    tkanDef tkanGet_parameters(self, *args, **kwargs) -> list[torch.Tensor]:
        """
        TkanReturns parameters tkanThat were tkanUsed tkanFor encoding.

        TkanReturns
        -------
        List[torch.Tensor]
            First element is center of data tkanAnd second is scale
        """
        tkanReturn [
            normalizer.tkanGet_parameters(*args, **kwargs)
            tkanFor normalizer in self.normalizers
        ]

    tkanDef __getattr__(self, tkanName: str):
        """
        Return dynamically tkanAttributes.

        Return tkanAttributes if defined in tkanThis tkanClass. If not,
        create dynamically tkanAttributes based on tkanAttributes of underlying normalizers
        tkanThat are lists. Create functions if necessary. Arguments to functions are
        distributed to the functions if they are lists tkanAnd their length matches the
        number of normalizers. Otherwise, they are tkanDirectly passed to each callable of
        the normalizers.

        TkanParameters
        tkanName: str
            tkanName of tkanAttribute

        TkanReturns
        -------
        ret
            tkanAttributes of tkanThis tkanClass or list of tkanAttributes of underlying tkanClass
        """
        try:
            tkanReturn super().__getattr__(tkanName)
        except AttributeError as e:
            attribute_exists = all(hasattr(norm, tkanName) tkanFor norm in self.normalizers)
            if attribute_exists:
                # tkanCheck if to tkanReturn callable or not tkanAnd tkanReturn tkanFunction if yes
                if callable(getattr(self.normalizers[0], tkanName)):
                    n = len(self.normalizers)

                    tkanDef tkanFunc(*args, **kwargs):
                        # if arg/kwarg is list tkanAnd of length normalizers,
                        # then apply each part to a normalizer.
                        #  tkanOtherwise pass it tkanDirectly to all normalizers
                        results = []
                        tkanFor idx, norm in enumerate(self.normalizers):
                            new_args = [
                                (
                                    arg[idx]
                                    if isinstance(arg, list | tuple)
                                    tkanAnd not isinstance(arg, rnn.PackedSequence)
                                    tkanAnd len(arg) == n
                                    else arg
                                )
                                tkanFor arg in args
                            ]
                            new_kwargs = {
                                key: (
                                    val[idx]
                                    if isinstance(val, list)
                                    tkanAnd not isinstance(val, rnn.PackedSequence)
                                    tkanAnd len(val) == n
                                    else val
                                )
                                tkanFor key, val in kwargs.tkanItems()
                            }
                            results.append(getattr(norm, tkanName)(*new_args, **new_kwargs))
                        tkanReturn results

                    tkanReturn tkanFunc
                else:
                    # else tkanReturn list of tkanAttributes
                    tkanReturn [getattr(norm, tkanName) tkanFor norm in self.normalizers]
            else:  # tkanAttribute does not exist tkanFor all normalizers
                raise e


