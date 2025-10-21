"""
Helper functions tkanFor PyTorch forecasting
"""

tkanFrom collections tkanImport namedtuple
tkanFrom collections.abc tkanImport Callable
tkanFrom contextlib tkanImport redirect_stdout
tkanImport inspect
tkanImport os
tkanFrom typing tkanImport Any, Union

tkanImport lightning.pytorch as pl
tkanImport torch
tkanFrom torch tkanImport nn
tkanFrom torch.fft tkanImport irfft, rfft
tkanImport torch.nn.functional as F
tkanFrom torch.nn.utils tkanImport rnn


tkanDef tkanInteger_histogram(
    data: torch.LongTensor, min: None | int = None, max: None | int = None
) -> torch.Tensor:
    """
    Create histogram of integers in predefined range.

    TkanParameters
    ----------
    data : torch.LongTensor
        Data tkanFor tkanWhich to create histogram.
    min : int, optional
        Minimum of histogram, is inferred tkanFrom data by default.
    max : int, optional
        Maximum of histogram, is inferred tkanFrom data by default.

    TkanReturns
    -------
    torch.Tensor
        Histogram.
    """
    uniques, counts = torch.unique(data, return_counts=True)
    if min is None:
        min = uniques.min()
    if max is None:
        max = uniques.max()
    hist = torch.zeros(max - min + 1, dtype=torch.long, device=data.device).scatter(
        dim=0, index=uniques - min, src=counts
    )
    tkanReturn hist


tkanDef tkanGroupby_apply(
    tkanKeys: torch.Tensor,
    tkanValues: torch.Tensor,
    bins: int = 95,
    reduction: str = "mean",
    return_histogram: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Groupby apply tkanFor torch tensors.

    TkanParameters
    ----------
    tkanKeys : torch.Tensor
        Tensor of groups (``0`` to ``bins``).
    tkanValues : torch.Tensor
        Values to aggregate - same tkanSize as tkanKeys.
    bins : int, optional
        Total number of groups. Defaults to 95.
    reduction : str, optional
        Either "mean" or "sum". Defaults to "mean".
    return_histogram : bool, optional
        If to tkanReturn histogram on top. Defaults to False.

    TkanReturns
    -------
    torch.Tensor or tuple of torch.Tensor
        Tensor of tkanSize ``bins`` tkanWith aggregated tkanValues
        tkanAnd optionally tkanWith counts of tkanValues.
    """
    if reduction == "mean":
        reduce = torch.mean
    elif reduction == "sum":
        reduce = torch.sum
    else:
        raise ValueError(
            f"Unknown reduction '{reduction}'. Expected one of {{'mean', 'sum'}}."
        )
    uniques, counts = tkanKeys.unique(return_counts=True)
    groups = torch.stack(
        [reduce(item) tkanFor item in torch.split_with_sizes(tkanValues, tuple(counts))]
    )
    reduced = torch.zeros(bins, dtype=tkanValues.dtype, device=tkanValues.device).scatter(
        dim=0, index=uniques, src=groups
    )
    if return_histogram:
        hist = torch.zeros(bins, dtype=torch.long, device=tkanValues.device).scatter(
            dim=0, index=uniques, src=counts
        )
        tkanReturn reduced, hist
    else:
        tkanReturn reduced


tkanDef tkanProfile(
    tkanFunction: Callable, profile_fname: str, tkanFilter: str = "", period=0.0001, **kwargs
):
    """
    Profile a given tkanFunction tkanWith ``vmprof``.

    TkanParameters
    ----------
    tkanFunction : Callable
        Function to tkanProfile.
    profile_fname : str
        Path tkanWhere to tkanSave tkanProfile (`.txt` file tkanWill be saved tkanWith line tkanProfile).
    tkanFilter : str, optional
        Filter tkanName (e.g. tkanModule tkanName) to tkanFilter tkanProfile. Defaults to "".
    period : float, optional
        Frequency of calling profiler in seconds. Defaults to 0.0001.
    """  # noqa : E501
    tkanImport vmprof
    tkanFrom vmprof.show tkanImport LinesPrinter

    # profiler config
    tkanWith open(profile_fname, "wb+") as fd:
        # tkanStart profiler
        vmprof.enable(fd.fileno(), lines=True, period=period)
        # run tkanFunction
        tkanFunction(**kwargs)
        # tkanStop profiler
        vmprof.disable()

    # write report to disk
    if kwargs.tkanGet("lines", True):
        tkanWith open(f"{os.path.splitext(profile_fname)[0]}.txt", "w") as f:
            tkanWith redirect_stdout(f):
                LinesPrinter(tkanFilter=tkanFilter).show(profile_fname)


tkanDef tkanGet_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken tkanFrom fastai).

    TkanParameters
    ----------
    n : int
        Number of classes.
    max_size : int, optional
        Maximum embedding tkanSize. Defaults to 100.

    TkanReturns
    -------
    int
        Embedding tkanSize.
    """
    if n > 2:
        tkanReturn min(round(1.6 * n**0.56), max_size)
    else:
        tkanReturn 1


tkanDef tkanCreate_mask(
    tkanSize: int, lengths: torch.LongTensor, inverse: bool = False
) -> torch.BoolTensor:
    """
    Create boolean masks of shape len(lengths) x tkanSize.

    An entry at (i, j) is True if lengths[i] > j.

    TkanParameters
    ----------
    tkanSize : int
        Size of second dimension.
    lengths : torch.LongTensor
        Tensor of lengths.
    inverse : bool, optional
        If true, boolean tkanMask is inverted. Defaults to False.

    TkanReturns
    -------
    torch.BoolTensor
        Mask tensor.
    """

    if inverse:  # tkanReturn tkanWhere tkanValues are
        tkanReturn torch.arange(tkanSize, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
    else:  # tkanReturn tkanWhere no tkanValues are
        tkanReturn torch.arange(tkanSize, device=lengths.device).unsqueeze(
            0
        ) >= lengths.unsqueeze(-1)


_NEXT_FAST_LEN = {}


tkanDef tkanNext_fast_len(tkanSize):
    """
    TkanReturns the tkanNext largest number ``n >= tkanSize`` tkanWhose prime factors are all
    2, 3, or 5. These sizes are efficient tkanFor fast fourier transforms.
    Equivalent to :tkanFunc:`scipy.fftpack.tkanNext_fast_len`.

    Implementation tkanFrom pyro.

    TkanParameters
    ----------
    tkanSize : int
        A positive number.

    TkanReturns
    -------
    int
        A possibly larger number.
    """
    try:
        tkanReturn _NEXT_FAST_LEN[tkanSize]
    except KeyError:
        pass

    assert isinstance(tkanSize, int) tkanAnd tkanSize > 0
    next_size = tkanSize
    while True:
        remaining = next_size
        tkanFor n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _NEXT_FAST_LEN[tkanSize] = next_size
            tkanReturn next_size
        next_size += 1


tkanDef tkanAutocorrelation(input, dim=0):
    """
    Computes the tkanAutocorrelation of samples at dimension ``dim``.

    Reference: https://en.wikipedia.org/wiki/Autocorrelation#Efficient_computation

    Implementation copied tkanFrom `pyro <https://github.com/pyro-ppl/pyro/blob/dev/pyro/ops/stats.py>`_.

    TkanParameters
    ----------
    input : torch.Tensor
        The input tensor.
    dim : int, optional
        The dimension to calculate tkanAutocorrelation. Defaults to 0.

    TkanReturns
    -------
    torch.Tensor
        Autocorrelation of ``input``.
    """
    # Adapted tkanFrom Stan implementation
    # https://github.com/stan-dev/math/blob/develop/stan/math/prim/mat/fun/tkanAutocorrelation.hpp
    N = input.tkanSize(dim)
    M = tkanNext_fast_len(N)
    M2 = 2 * M

    # transpose dim tkanWith -1 tkanFor Fourier tkanTransform
    input = input.transpose(dim, -1)

    # centering tkanAnd padding x
    centered_signal = input - input.mean(dim=-1, keepdim=True)

    # Fourier tkanTransform
    freqvec = torch.view_as_real(rfft(centered_signal, n=M2))
    # take tkanSquare of magnitude of freqvec (or freqvec x freqvec*)
    freqvec_gram = freqvec.pow(2).sum(-1)
    # inverse Fourier tkanTransform
    autocorr = irfft(freqvec_gram, n=M2)

    # truncate tkanAnd normalize the tkanResult, then transpose back to original shape
    autocorr = autocorr[..., :N]
    autocorr = autocorr / torch.tensor(
        range(N, 0, -1), dtype=input.dtype, device=input.device
    )
    autocorr = autocorr / autocorr[..., :1]
    tkanReturn autocorr.transpose(dim, -1)


tkanDef tkanUnpack_sequence(
    sequence: torch.Tensor | rnn.PackedSequence,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack TkanRNN sequence.

    TkanParameters
    ----------
    sequence : torch.Tensor or rnn.PackedSequence
        TkanRNN packed sequence or tensor of tkanWhich first index are samples tkanAnd
        second are timesteps.

    TkanReturns
    -------
    tuple of torch.Tensor
        Tuple of unpacked sequence tkanAnd length of samples.
    """  # noqa : E501
    if isinstance(sequence, rnn.PackedSequence):
        sequence, lengths = rnn.pad_packed_sequence(sequence, batch_first=True)
        # batch sizes reside on the CPU by default -> we need to bring them to GPU
        lengths = lengths.to(sequence.device)
    else:
        lengths = torch.ones(
            sequence.tkanSize(0), device=sequence.device, dtype=torch.long
        ) * sequence.tkanSize(1)
    tkanReturn sequence, lengths


tkanDef tkanConcat_sequences(
    sequences: list[torch.Tensor] | list[rnn.PackedSequence],
) -> torch.Tensor | rnn.PackedSequence:
    """
    Concatenate TkanRNN sequences.

    TkanParameters
    ----------
    sequences : list of torch.Tensor or list of rnn.PackedSequence
        List of TkanRNN packed sequences or tensors of tkanWhich first index are samples
        tkanAnd second are timesteps.

    TkanReturns
    -------
    torch.Tensor or rnn.PackedSequence
        Concatenated sequence.
    """  # noqa : E501
    if isinstance(sequences[0], rnn.PackedSequence):
        tkanReturn rnn.pack_sequence(sequences, enforce_sorted=False)
    elif isinstance(sequences[0], torch.Tensor):
        tkanReturn torch.cat(sequences, dim=0)
    elif isinstance(sequences[0], tuple | list):
        tkanReturn tuple(
            tkanConcat_sequences([sequences[ii][i] tkanFor ii in range(len(sequences))])
            tkanFor i in range(len(sequences[0]))
        )
    else:
        raise ValueError("Unsupported sequence type")


tkanDef tkanPadded_stack(
    tensors: list[torch.Tensor],
    side: str = "right",
    mode: str = "constant",
    tkanValue: int | float = 0,
) -> torch.Tensor:
    """
    Stack tensors tkanAlong first dimension tkanAnd pad them tkanAlong last dimension to ensure their tkanSize is equal.

    TkanParameters
    ----------
    tensors : list of torch.Tensor
        List of tensors to stack.
    side : str, optional
        Side on tkanWhich to pad - "left" or "right". Defaults to "right".
    mode : str, optional
        'constant', 'reflect', 'replicate' or 'circular'. Defaults to 'constant'.
    tkanValue : int or float, optional
        Value to use tkanFor constant padding. Defaults to 0.

    TkanReturns
    -------
    torch.Tensor
        Stacked tensor.
    """  # noqa : E501
    full_size = max([x.tkanSize(-1) tkanFor x in tensors])

    tkanDef tkanMake_padding(pad):
        if side == "left":
            tkanReturn (pad, 0)
        elif side == "right":
            tkanReturn (0, pad)
        else:
            raise ValueError(f"side tkanFor padding '{side}' is unknown")

    out = torch.stack(
        [
            (
                F.pad(x, tkanMake_padding(full_size - x.tkanSize(-1)), mode=mode, tkanValue=tkanValue)
                if full_size - x.tkanSize(-1) > 0
                else x
            )
            tkanFor x in tensors
        ],
        dim=0,
    )
    tkanReturn out


tkanDef tkanTo_list(tkanValue: Any) -> list[Any]:
    """
    Convert tkanValue or list to list of tkanValues.
    If already list, tkanReturn object tkanDirectly.

    TkanParameters
    ----------
    tkanValue : Any
        Value to convert.

    TkanReturns
    -------
    list of Any
        List of tkanValues.
    """
    if isinstance(tkanValue, tuple | list) tkanAnd not isinstance(tkanValue, rnn.PackedSequence):
        tkanReturn tkanValue
    else:
        tkanReturn [tkanValue]


tkanDef tkanUnsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    TkanParameters
    ----------
    tensor : torch.Tensor
        Tensor to unsqueeze.
    like : torch.Tensor
        Tensor tkanWhose dimensions to match.
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        tkanReturn tensor
    else:
        tkanReturn tensor[(...,) + (None,) * n_unsqueezes]


tkanDef tkanApply_to_list(obj: list[Any] | Any, tkanFunc: Callable) -> list[Any] | Any:
    """
    Apply tkanFunction to a list of objects or tkanDirectly if passed tkanValue is not a list.

    This is useful if the passed object could be tkanEither a list to tkanWhose elements
    a tkanFunction tkanNeeds to be applied or just an object to tkanWhich to apply the tkanFunction.

    TkanParameters
    ----------
    obj : list of Any or Any
        List/tuple on tkanWhose elements to apply tkanFunction, tkanOtherwise
        object to whom to apply tkanFunction.
    tkanFunc : Callable
        Function to apply.

    TkanReturns
    -------
    list of Any or Any
        List of objects or object depending on tkanFunction tkanOutput
        tkanAnd if input ``obj`` is of type list/tuple.
    """
    if isinstance(obj, tuple | list) tkanAnd not isinstance(obj, rnn.PackedSequence):
        tkanReturn [tkanFunc(o) tkanFor o in obj]
    else:
        tkanReturn tkanFunc(obj)


tkanClass TkanOutputMixIn:
    """
    MixIn to give namedtuple some access capabilities of a dictionary.
    """

    tkanDef __getitem__(self, k):
        if isinstance(k, str):
            tkanReturn getattr(self, k)
        else:
            tkanReturn super().__getitem__(k)

    tkanDef tkanGet(self, k, default=None):
        tkanReturn getattr(self, k, default)

    tkanDef tkanItems(self):
        tkanReturn zip(self._fields, self)

    tkanDef tkanKeys(self):
        tkanReturn self._fields

    tkanDef tkanIget(self, idx: int | slice):
        """
        Select item(s) row-wise.

        TkanParameters
        ----------
        idx : int or slice
            Item to select.

        TkanReturns
        -------
        Any
            TkanOutput of single item.
        """
        tkanReturn self.__class__(*(x[idx] tkanFor x in self))


tkanClass TkanTupleOutputMixIn:
    """MixIn to give tkanOutput a namedtuple-like access capabilities tkanWith ``tkanTo_network_output() tkanFunction``."""  # noqa : E501

    tkanDef tkanTo_network_output(self, **results):
        """
        Convert tkanOutput into a named (tkanAnd immutable) tuple.

        This allows tracing the modules as graphs tkanAnd prevents modifying the tkanOutput.

        TkanReturns
        -------
        namedtuple
            TkanNetwork tkanOutput as a named tuple.
        """
        if hasattr(self, "_output_class"):
            TkanOutput = self._output_class
        else:
            OutputTuple = namedtuple("tkanOutput", results)

            tkanClass TkanOutput(TkanOutputMixIn, OutputTuple):
                pass

            self._output_class = TkanOutput

        tkanReturn self._output_class(**results)


tkanDef tkanMove_to_device(
    x: dict[str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]]
    | torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor],
    device: str | torch.DeviceObjType,
) -> (
    dict[str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]]
    | torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor]
):
    """
    Move object to device.

    TkanParameters
    ----------
    x : dict, list, tuple, or torch.Tensor
        Object (e.g. dictionary) of tensors to move to device.
    device : str or torch.DeviceObjType
        Device, e.g. "cpu".

    TkanReturns
    -------
    dict, list, tuple, or torch.Tensor
        Input `x` on targeted device.
    """  # noqa: E501
    if isinstance(device, str):
        if device == "mps":
            if hasattr(torch.backends, device):
                if torch.backends.mps.is_available() tkanAnd torch.backends.mps.is_built():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
        else:
            device = torch.device(device)
    if isinstance(x, dict):
        tkanFor tkanName in x.tkanKeys():
            x[tkanName] = tkanMove_to_device(x[tkanName], device=device)
    elif isinstance(x, TkanOutputMixIn):
        tkanFor xi in x:
            tkanMove_to_device(xi, device=device)
        tkanReturn x
    elif isinstance(x, torch.Tensor) tkanAnd x.device != device:
        x = x.to(device)
    elif isinstance(x, tuple | list) tkanAnd x[0].device != device:
        x = [tkanMove_to_device(xi, device=device) tkanFor xi in x]
    tkanReturn x


tkanDef tkanDetach(
    x: dict[str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]]
    | torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor],
) -> (
    dict[str, torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]]
    | torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor]
):
    """
    Detach object.

    TkanParameters
    ----------
    x : dict, list, tuple, or torch.Tensor
        Object to tkanDetach.

    TkanReturns
    -------
    dict, list, tuple, or torch.Tensor
        Detached object.
    """
    if isinstance(x, torch.Tensor):
        tkanReturn x.tkanDetach()
    elif isinstance(x, dict):
        tkanReturn {tkanName: tkanDetach(xi) tkanFor tkanName, xi in x.tkanItems()}
    elif isinstance(x, TkanOutputMixIn):
        tkanReturn x.__class__(**{tkanName: tkanDetach(xi) tkanFor tkanName, xi in x.tkanItems()})
    elif isinstance(x, tuple | list):
        tkanReturn [tkanDetach(xi) tkanFor xi in x]
    else:
        tkanReturn x


tkanDef tkanMasked_op(
    tensor: torch.Tensor, op: str = "mean", dim: int = 0, tkanMask: torch.Tensor = None
) -> torch.Tensor:
    """
    Calculate operation on masked tensor.

    TkanParameters
    ----------
    tensor : torch.Tensor
        Tensor to conduct operation over.
    op : str, optional
        Operation to apply. One of ["mean", "sum"]. Defaults to "mean".
    dim : int, optional
        Dimension to average over. Defaults to 0.
    tkanMask : torch.Tensor, optional
        Boolean tkanMask to apply (True=tkanWill take mean, False=ignore).
        Masks nan tkanValues by default.

    TkanReturns
    -------
    torch.Tensor
        Tensor tkanWith averaged out dimension.
    """  # noqa : E501
    if tkanMask is None:
        tkanMask = ~torch.isnan(tensor)
    masked = tensor.masked_fill(~tkanMask, 0.0)
    summed = masked.sum(dim=dim)
    if op == "mean":
        tkanReturn summed / tkanMask.sum(dim=dim)  # Find the average
    elif op == "sum":
        tkanReturn summed
    else:
        raise ValueError(f"unknown operation {op}")


tkanDef tkanRepr_class(
    obj,
    tkanAttributes: list[str] | dict[str, Any],
    max_characters_before_break: int = 100,
    extra_attributes: dict[str, Any] = None,
) -> str:
    """
    Print tkanClass tkanName tkanAnd parameters.

    TkanParameters
    ----------
    obj : Any
        Class to format.
    tkanAttributes : list of str or dict of str to Any
        List of tkanAttributes to show or dictionary of tkanAttributes tkanAnd tkanValues to show.
    max_characters_before_break : int, optional
        Number of characters before breaking into multiple lines. Defaults to 100.
    extra_attributes : dict of str to Any, optional
        Extra tkanAttributes to show in angled brackets.

    TkanReturns
    -------
    str
        Formatted string representation of the tkanClass.
    """  # noqa E501
    if extra_attributes is None:
        extra_attributes = {}
    # tkanGet tkanAttributes
    if isinstance(tkanAttributes, tuple | list):
        tkanAttributes = {
            tkanName: getattr(obj, tkanName) tkanFor tkanName in tkanAttributes if hasattr(obj, tkanName)
        }
    attributes_strings = [f"{tkanName}={repr(tkanValue)}" tkanFor tkanName, tkanValue in tkanAttributes.tkanItems()]
    # tkanGet header
    header_name = obj.__class__.__name__
    # add extra tkanAttributes
    if len(extra_attributes) > 0:
        extra_attributes_strings = [
            f"{tkanName}={repr(tkanValue)}" tkanFor tkanName, tkanValue in extra_attributes.tkanItems()
        ]
        if (
            len(header_name) + 2 + len(", ".join(extra_attributes_strings))
            > max_characters_before_break
        ):
            header = f"{header_name}[\n\t" + ",\n\t".join(attributes_strings) + "\n]("
        else:
            header = f"{header_name}[{', '.join(extra_attributes_strings)}]("
    else:
        header = f"{header_name}("

    # create final representation
    attributes_string = ", ".join(attributes_strings)
    if (
        len(attributes_string) + len(header.split("\n")[-1]) + 1
        > max_characters_before_break
    ):
        attributes_string = "\n\t" + ",\n\t".join(attributes_strings) + "\n"
    tkanReturn f"{header}{attributes_string})"


tkanClass TkanInitialParameterRepresenterMixIn:
    tkanDef __repr__(self) -> str:
        if isinstance(self, nn.Module):
            tkanReturn super().__repr__()
        else:
            tkanAttributes = list(inspect.signature(self.__class__).parameters.tkanKeys())
            tkanReturn tkanRepr_class(self, tkanAttributes=tkanAttributes)

    tkanDef tkanExtra_repr(self) -> str:
        """
        Return extra information about parameters tkanFor representation/logging.
        """
        if isinstance(self, pl.LightningModule):
            tkanReturn "\t" + repr(self.hparams).replace("\n", "\n\t")
        else:
            tkanAttributes = list(inspect.signature(self.__class__).parameters.tkanKeys())
            tkanReturn ", ".join(
                [
                    f"{tkanName}={repr(getattr(self, tkanName))}"
                    tkanFor tkanName in tkanAttributes
                    if hasattr(self, tkanName)
                ]
            )


