from itertools import starmap
import tempfile

import einops
import numpy
import numpy as np
import panel
from PIL import Image
import torch

from fragile.fragile_typing import BooleanValue, DictValues, Value, Walkers


def numpy_dtype_to_torch_dtype(dtype):
    """Convert a numpy data type to a corresponding torch data type.
    If a torch data type is provided, return it as is.

    Args:
        dtype (numpy.dtype or torch.dtype): The data type to convert.

    Returns:
        torch.dtype: The corresponding torch data type.

    """
    if isinstance(dtype, torch.dtype):
        return dtype

    dtype_mapping = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.float16: torch.float16,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }
    numpy_to_torch_dtype = {
        np.dtypes.BoolDType: torch.bool,
        np.dtypes.ByteDType: torch.uint8,
        np.dtypes.BytesDType: None,  # No direct equivalent in PyTorch
        np.dtypes.CLongDoubleDType: torch.complex128,  # Closest equivalent
        np.dtypes.Complex128DType: torch.complex128,
        np.dtypes.Complex64DType: torch.complex64,
        np.dtypes.DateTime64DType: None,  # No equivalent in PyTorch
        np.dtypes.Float16DType: torch.float16,
        np.dtypes.Float32DType: torch.float32,
        np.dtypes.Float64DType: torch.float64,
        np.dtypes.Int16DType: torch.int16,
        np.dtypes.Int32DType: torch.int32,
        np.dtypes.Int64DType: torch.int64,
        np.dtypes.Int8DType: torch.int8,
        np.dtypes.IntDType: torch.int32,  # NumPy's int is usually int32, but platform-dependent
        np.dtypes.LongDType: torch.int64,  # NumPy long maps to int64
        np.dtypes.LongDoubleDType: torch.float64,  # Closest match
        np.dtypes.LongLongDType: torch.int64,  # Closest match
        np.dtypes.ObjectDType: None,  # No equivalent in PyTorch
        np.dtypes.ShortDType: torch.int16,
        np.dtypes.StrDType: None,  # No equivalent in PyTorch
        np.dtypes.StringDType: None,  # No equivalent in PyTorch
        np.dtypes.TimeDelta64DType: None,  # No equivalent in PyTorch
        np.dtypes.UByteDType: torch.uint8,
        np.dtypes.UInt16DType: None,  # PyTorch does not support unsigned integers
        np.dtypes.UInt32DType: None,  # PyTorch does not support unsigned integers
        np.dtypes.UInt64DType: None,  # PyTorch does not support unsigned integers
        np.dtypes.UInt8DType: torch.uint8,  # Closest match
        np.dtypes.UIntDType: None,  # No equivalent in PyTorch
        np.dtypes.ULongDType: None,  # No equivalent in PyTorch
        np.dtypes.ULongLongDType: None,  # No equivalent in PyTorch
        np.dtypes.UShortDType: None,  # No equivalent in PyTorch
        np.dtypes.VoidDType: None,  # No equivalent in PyTorch
        **dtype_mapping,
    }
    torch_value = numpy_to_torch_dtype.get(dtype, None)
    if torch_value is None:
        torch_value = numpy_to_torch_dtype.get(type(dtype), None)
    return torch_value


def create_gif(data, filename=None, fps=10, optimize=False):
    duration = int((len(data) / fps) * 20)
    filename = tempfile.NamedTemporaryFile(suffix="a.gif") if filename is None else filename
    images = [Image.fromarray(v) for v in data]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=duration,
        loop=0,
    )
    return filename


def show_root_game(swarm, fps=10, optimize=False):
    vals, *_ = zip(*list(swarm.tree.iterate_root_path(names=["rgb"])))
    f = create_gif([x[0] for x in vals], fps=fps, optimize=optimize)
    return panel.pane.GIF(f)


def shape_is_vector(v):
    from fragile.core.config import ValueConfig  # noqa:PLC0415

    shape = v.shape if isinstance(v, ValueConfig) else v.get("shape", ())
    return not (shape is None or (isinstance(shape, tuple) and len(shape) > 0))


def get_n_walkers(state: DictValues) -> int | None:
    """Return the batch size which is the first dimension of the first key of the state."""
    if state is None:
        return None
    if isinstance(state, dict) and len(state) == 0:
        return 0
    if not isinstance(state, dict):
        msg = f"The state must be a dict but {type(state)} was passed."
        raise ValueError(msg)
    first_value = next(iter(state.values()))
    if not isinstance(first_value, Value):
        msg = f"The state must contain values but {type(first_value)} was passed. state: {state}"
        raise ValueError(msg)
    return len(first_value) if isinstance(first_value, list) else first_value.shape[0]


def is_single_value(state: DictValues) -> bool:
    """Return True if the state and kwargs contain a single value."""
    return get_n_walkers(state) == 1


def combine_values(state: DictValues = None, **kwargs: Value) -> DictValues:
    """Combine the state and kwargs into a single state dictionary."""
    if state is None:
        return kwargs or None
    if not kwargs:
        return state or None
    n_walkers_state = get_n_walkers(state)
    n_walkers_kwargs = get_n_walkers(kwargs)
    if n_walkers_state != n_walkers_kwargs:
        msg = (
            "The state must contain a the same number of walkers as kwargs. "
            f"Got n_state {n_walkers_state}!={n_walkers_kwargs} n_kwargs."
        )
        raise ValueError(msg)
    return {**state, **kwargs}


def running_in_ipython() -> bool:
    """Return ``True`` if the code is this function is being called from an IPython kernel."""
    try:
        from IPython import get_ipython  # noqa:PLC0415

        return get_ipython() is not None
    except ImportError:
        return False


def remove_notebook_margin(output_width_pct: int = 80):
    """Make the notebook output wider."""
    from IPython.core.display import HTML  # noqa:PLC0415

    html = (
        "<style>.container { width:" + str(output_width_pct) + "% !important; }"
        ".input{ width:70% !important; }"
        ".text_cell{ width:70% !important;"
        " font-size: 16px;}"
        ".title {align:center !important;}"
        "</style>"
    )
    return HTML(html)


def statistics_from_array(x: numpy.ndarray):
    """Return the (mean, std, max, min) of an array."""
    try:
        return (
            einops.asnumpy(x).mean(),
            einops.asnumpy(x).std(),
            einops.asnumpy(x).max(),
            einops.asnumpy(x).min(),
        )
    except (AttributeError, TypeError, ValueError):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan


def batch_to_walkers(state: DictValues, as_items: bool = False) -> Walkers:
    """Transform a state representing a population of walkers, into a list of single walkers.

    Args:
        state: A dictionary containing the state of a population of walkers.
        as_items: If true return the elements instead of batches of size 1.

    Returns:
        A list of dictionaries, each containing the state of a single walker.

    """

    def _get_index(x, ix):
        val = x[ix]
        if isinstance(x, list):
            return val if as_items else [val]
        if not as_items:
            return val[None]
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()
        if isinstance(val, numpy.ndarray):
            val = val.tolist()
        return val

    walkers = []
    for i in range(get_n_walkers(state)):
        single_walker = {}
        for k in state.keys():
            value = _get_index(state[k], i)
            single_walker[k] = value
        walkers.append(single_walker)
    return walkers


def walkers_to_batch(walkers: list[Walkers]) -> DictValues:
    """Transform a list of walkers into a batch of walkers.

    Args:
        walkers: A list of dictionaries, each containing the state of a single walker.

    Returns:
        A dictionary containing the state of a population of walkers.

    """
    batch = {}
    for key in walkers[0].keys():
        first_value = walkers[0][key]
        if isinstance(first_value, list):
            batch[key] = [value for walker in walkers for value in walker[key]]
        elif isinstance(first_value, numpy.ndarray):
            batch[key] = numpy.concatenate([walker[key] for walker in walkers])
        elif isinstance(first_value, torch.Tensor):
            batch[key] = torch.cat([walker[key] for walker in walkers])
        else:
            msg = f"Unsupported value type: {type(first_value)}"
            raise ValueError(msg)
    return batch


def get_index_value(value: Value, index: BooleanValue) -> Value:
    if index.all().item():
        return value
    if isinstance(value, list):
        return [v for v, ix in zip(value, einops.asnumpy(index)) if ix]
    if isinstance(value, numpy.ndarray):
        return value[einops.asnumpy(index)]
    if isinstance(value, torch.Tensor):
        return value[index]
    return None


def select_index(state: DictValues, index: BooleanValue) -> DictValues:
    """Select the elements in the dictionary corresponding to the provided index."""
    return {k: get_index_value(v, index) for k, v in state.items()}


def set_index_value(value: Value, index: BooleanValue, new_value: Value) -> Value:
    if index.all().item():
        return new_value
    if isinstance(value, list):
        index = einops.asnumpy(index)
        true_indices = numpy.where(index)[0]
        for i, v in zip(true_indices, new_value):
            value[i] = v
        return value
    if isinstance(value, numpy.ndarray):
        value[einops.asnumpy(index)] = new_value
        return value
    if isinstance(value, torch.Tensor):
        try:
            value[index] = new_value
        except Exception:  # TODO: Make sure this is right or I should always .detach().clone()
            value[index] = new_value.detach().clone()
        return value
    return None


def set_index(
    state: DictValues, index: BooleanValue, new_state: DictValues, ignore_names=()
) -> DictValues:
    """Set the elements in the dictionary corresponding to the provided index."""
    try:
        return {
            k: set_index_value(v, index, new_state[k]) if k in new_state else v
            for k, v in state.items()
            if k not in ignore_names
        }
    except Exception as e:
        msg = f"Error setting index in state: {state}\nindex: {index}\nnew_state: {new_state}"
        raise ValueError(msg) from e


def values_are_equal(value_1: Value, value_2: Value) -> bool:
    if not isinstance(value_1, type(value_2)):
        return False
    if isinstance(value_1, list):
        return all(starmap(values_are_equal, zip(value_1, value_2)))
    if isinstance(value_1, numpy.ndarray):
        try:
            return numpy.allclose(value_1, value_2, equal_nan=True)
        except TypeError:
            return numpy.all(value_1 == value_2)
    if isinstance(value_1, torch.Tensor):
        return torch.allclose(value_1, value_2, equal_nan=True)
    return value_1 == value_2


def states_are_equal(state_1: DictValues, state_2: DictValues) -> bool:
    """Return ``True`` if the two states are equal."""
    if not isinstance(state_1, dict) or not isinstance(state_2, dict):
        return False
    if set(state_1.keys()) != set(state_2.keys()):
        return False
    for k in state_1.keys():
        if not values_are_equal(state_1[k], state_2[k]):
            return False
    return True
