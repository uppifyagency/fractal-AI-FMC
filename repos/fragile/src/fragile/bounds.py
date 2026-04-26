from collections.abc import Iterable, Iterable as _Iterable

import einops
import numpy
import torch
from torch import Tensor

from fragile.fragile_typing import Scalar
from fragile.utils import numpy_dtype_to_torch_dtype


def where(cond, a, b, *args, **kwargs):  # noqa: ARG001
    was_bool = False
    if a.dtype == torch.bool:
        a = a.to(torch.int32)
        was_bool = True
    if b.dtype == torch.bool:
        b = b.to(torch.int32)
        was_bool = True
    res = torch.where(cond, a, b)
    return res.to(torch.bool) if was_bool else res


class Bounds:
    """The :class:`Bounds` implements the logic for defining and managing closed intervals."""

    def __new__(cls, high, low, *args, **kwargs):
        """Instantiate a :class:`Bounds`."""
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        if isinstance(high, list):
            high = torch.tensor(high)
        if isinstance(low, list):
            low = torch.tensor(low)

        if isinstance(dtype, numpy.dtype):
            return NumpyBounds(high, low, *args, **kwargs)
        if isinstance(dtype, torch.dtype):
            return TorchBounds(high, low, *args, **kwargs)

        if device is not None:
            return TorchBounds(high, low, *args, **kwargs)
        if isinstance(high, numpy.ndarray) or isinstance(low, numpy.ndarray):
            return NumpyBounds(high, low, *args, **kwargs)
        if isinstance(high, torch.Tensor) or isinstance(low, torch.Tensor | int | float):
            return TorchBounds(high, low, *args, **kwargs)
        msg = "Inputs must be either numpy arrays or torch tensors."
        raise TypeError(msg)

    @classmethod
    def from_tuples(cls, bounds: Iterable[tuple]) -> "NumpyBounds | TorchBounds":
        """Instantiate a :class:`Bounds` from a collection of tuples containing \
        the higher and lower bounds for every dimension as a tuple.

        Args:
            bounds: Iterable that returns tuples containing the higher and lower \
                    bound for every dimension of the target bounds.

        Returns:
                :class:`Bounds` instance.

        Examples:
            >>> intervals = ((-1., 1.), (-2., 1.), (2, 3))
            >>> bounds = Bounds.from_tuples(intervals)
            >>> print(bounds)
            TorchBounds shape torch.float32 dtype torch.Size([3]) \
low tensor([-1., -2.,  2.]) high tensor([1., 1., 3.])

        """
        low, high = [], []
        for lo, hi in bounds:
            low.append(lo)
            high.append(hi)
        return Bounds(low=low, high=high)

    @classmethod
    def from_space(cls, space: "gym.spaces.box.Box") -> "Bounds":  # noqa: F821
        """Initialize a :class:`Bounds` from a :class:`Box` gym action space."""
        return Bounds(low=space.low, high=space.high, dtype=space.dtype)

    @classmethod
    def from_array(cls, x: Tensor | numpy.ndarray, scale: float = 1.0) -> "Bounds":
        """Instantiate a bounds compatible for bounding the given array. It also allows to set a \
        margin for the high and low values.

        The value of the high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            x: Numpy array used to initialize the bounds.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` instance.

        Examples:
            >>> import torch
            >>> x = torch.ones((3, 3))
            >>> x[1:-1, 1:-1] = -5
            >>> bounds = Bounds.from_array(x, scale=1.5)
            >>> print(bounds)
            TorchBounds shape torch.float32 dtype torch.Size([3]) \
low tensor([ 0.5000, -7.5000,  0.5000]) high tensor([1.5000, 1.5000, 1.5000])

        """
        return (
            NumpyBounds.from_array(x, scale=scale)
            if isinstance(x, numpy.ndarray)
            else TorchBounds.from_array(x, scale=scale)
        )


class TorchBounds:
    """The :class:`Bounds` implements the logic for defining and managing closed intervals, \
    and checking if a numpy array's values are inside a given interval.

    It is used on a numpy array of a target shape.
    """

    def __init__(
        self,
        high: torch.Tensor | Scalar = torch.inf,
        low: torch.Tensor | Scalar = -numpy.inf,
        shape: tuple | None = None,
        dtype: type | None = None,
        device: str | None = None,
    ):
        """Initialize a :class:`Bounds`.

        Args:
            high: Higher value for the bound interval. If it is an typing_.Scalar \
                  it will be applied to all the coordinates of a target vector. \
                  If it is a vector, the bounds will be checked coordinate-wise. \
                  It defines and closed interval.
            low: Lower value for the bound interval. If it is a typing_.Scalar it \
                 will be applied to all the coordinates of a target vector. \
                 If it is a vector, the bounds will be checked coordinate-wise. \
                 It defines and closed interval.
            shape: Shape of the array that will be bounded. Only needed if `high` and `low` are \
                   vectors, and it is used to define the dimensions that will be bounded.
            dtype:  Data type of the array that will be bounded. It can be inferred from `high` \
                    or `low` (the type of `high` takes priority).

            device: Device where the bounds will be stored. If None, it will be inferred from \
                    `high` or `low` (the device of `high` takes priority).

        Examples:
            Initializing :class:`Bounds` using  numpy arrays:

            >>> import torch
            >>> high, low = torch.ones(3, dtype=torch.float), -1 * torch.ones(3, dtype=torch.int)
            >>> bounds = Bounds(high=high, low=low)
            >>> print(bounds)
            TorchBounds shape torch.float32 dtype torch.Size([3]) \
low tensor([-1, -1, -1], dtype=torch.int32) high tensor([1., 1., 1.])

            Initializing :class:`Bounds` using  typing_.Scalars:

            >>> high, low, shape = 4, 2.1, (5,)
            >>> bounds = Bounds(high=high, low=low, shape=shape)
            >>> print(bounds)
            TorchBounds shape torch.float32 dtype torch.Size([5]) low \
tensor([2.1000, 2.1000, 2.1000, 2.1000, 2.1000]) high tensor([4., 4., 4., 4., 4.])

        """
        if dtype is not None:
            dtype = numpy_dtype_to_torch_dtype(dtype)
        # Infer shape if not specified
        if shape is None and hasattr(high, "shape"):
            shape = high.shape
        elif shape is None and hasattr(low, "shape"):
            shape = low.shape
        elif shape is None:
            msg = "If shape is None high or low need to have .shape attribute."
            raise TypeError(msg)
        # High and low will be arrays of target shape
        if not isinstance(high, torch.Tensor) or (
            isinstance(high, torch.Tensor) and high.ndim == 0
        ):
            high = (
                torch.tensor(high) if isinstance(high, _Iterable) else (torch.ones(shape) * high)
            )
        if not isinstance(low, torch.Tensor) or (isinstance(low, torch.Tensor) and low.ndim == 0):
            low = torch.tensor(low) if isinstance(low, _Iterable) else (torch.ones(shape) * low)
        self.high = high.to(dtype=dtype, device=device)
        self.low = low.to(dtype=dtype, device=device)
        self._bounds_dist = self.high - self.low
        if dtype is not None:
            self.dtype = dtype
        elif hasattr(high, "dtype"):
            self.dtype = high.dtype
        elif hasattr(low, "dtype"):
            self.dtype = low.dtype
        else:
            self.dtype = type(high) if high is not None else type(low)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} shape {self.dtype} dtype {self.shape}"
            f" low {self.low} high {self.high}"
        )

    def __len__(self) -> int:
        """Return the number of dimensions of the bounds."""
        return len(self.high)

    def __contains__(self, item):
        return self.contains(item)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the current bounds.

        Returns
            tuple containing the shape of `high` and `low`

        """
        return self.high.shape

    @classmethod
    def from_tuples(cls, bounds: Iterable[tuple]) -> "TorchBounds":
        """Instantiate a :class:`Bounds` from a collection of tuples containing \
        the higher and lower bounds for every dimension as a tuple.

        Args:
            bounds: Iterable that returns tuples containing the higher and lower \
                    bound for every dimension of the target bounds.

        Returns:
                :class:`Bounds` instance.

        Examples:
            >>> intervals = ((-1., 1.), (-2., 1.), (2., 3.))
            >>> bounds = Bounds.from_tuples(intervals)
            >>> print(bounds)
            TorchBounds shape torch.float32 dtype torch.Size([3]) \
low tensor([-1., -2.,  2.]) high tensor([1., 1., 3.])

        """
        low, high = [], []
        for lo, hi in bounds:
            low.append(lo)
            high.append(hi)
        low, high = torch.tensor(low, dtype=torch.float), torch.tensor(high, dtype=torch.float)
        return TorchBounds(low=low, high=high)

    @classmethod
    def from_space(cls, space: "gym.spaces.box.Box") -> "TorchBounds":  # noqa: F821
        """Initialize a :class:`Bounds` from a :class:`Box` gym action space."""
        return TorchBounds(low=space.low, high=space.high, dtype=space.dtype)

    @staticmethod
    def get_scaled_intervals(
        low: Tensor | (float | int),
        high: Tensor | (float | int),
        scale: float,
    ) -> tuple[Tensor | float, Tensor | float]:
        """Scale the high and low vectors by a scale factor.

        The value of the high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            high: Higher bound to be scaled.
            low: Lower bound to be scaled.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` instance.

        """
        pct = torch.tensor(scale - 1)
        big_scale = 1 + torch.abs(pct)
        small_scale = 1 - torch.abs(pct)
        zero_l = torch.zeros_like(low) if isinstance(low, Tensor) else 0.0
        zero_h = torch.zeros_like(high) if isinstance(high, Tensor) else 0.0
        if pct > 0:
            xmin_scaled = where(low < zero_l, low * big_scale, low * small_scale)
            xmax_scaled = where(high < zero_h, high * small_scale, high * big_scale)
        else:
            xmin_scaled = where(low < zero_l, low * small_scale, low * small_scale)
            xmax_scaled = where(high < zero_h, high * big_scale, high * small_scale)
        return xmin_scaled, xmax_scaled

    @classmethod
    def from_array(cls, x: Tensor, scale: float = 1.0) -> "Bounds":
        """Instantiate a bounds compatible for bounding the given array. It also allows to set a \
        margin for the high and low values.

        The value of the high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            x: Numpy array used to initialize the bounds.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` instance.

        Examples:
            >>> import torch
            >>> x = torch.ones((3, 3))
            >>> x[1:-1, 1:-1] = -5
            >>> bounds = Bounds.from_array(x, scale=1.5)
            >>> print(bounds)
            TorchBounds shape torch.float32 dtype torch.Size([3]) \
low tensor([ 0.5000, -7.5000,  0.5000]) high tensor([1.5000, 1.5000, 1.5000])

        """
        xmin, xmax = torch.min(x, dim=0).values, torch.max(x, dim=0).values
        xmin_scaled, xmax_scaled = cls.get_scaled_intervals(xmin, xmax, scale)
        return TorchBounds(low=xmin_scaled, high=xmax_scaled)

    def clip(self, x: Tensor) -> Tensor:
        """Clip the values of the target array to fall inside the bounds (closed interval).

        Args:
            x: Numpy array to be clipped.

        Returns:
            Clipped numpy array with all its values inside the defined bounds.

        """
        return torch.clamp(x, self.low.to(x), self.high.to(x))

    def pbc(self, x: Tensor) -> Tensor:
        """Calculate periodic boundary conditions of the target array to fall inside \
        the bounds (closed interval).

        Args:
            x: Tensor to apply the periodic boundary conditions.

        Returns:
            Periodic boundary condition so all the values are inside the defined bounds.

        """
        x = x.to(self.low)
        x = where(x < self.high, x, torch.fmod(x, self.high) + self.low)
        return where(x > self.low, x, self.high - torch.fmod(x, self.low))

    def pbc_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculate periodic boundary conditions of the target array to fall inside \
        the bounds (closed interval).

        Args:
            x: Tensor to apply the periodic boundary conditions.
            y: Tensor containing the frontier of the periodic boundary condition.

        Returns:
            Periodic boundary condition so all the values are inside the defined bounds.

        """
        x, y = x.to(self.low), y.to(self.low)
        delta = torch.abs(x - y)
        return where(x > 0.5 * self._bounds_dist, delta - self._bounds_dist, delta)

    def contains(self, x: Tensor) -> Tensor | bool:
        """Check if the rows of the target array have all their coordinates inside \
        specified bounds.

        If the array is one dimensional it will return a boolean, otherwise a vector of booleans.

        Args:
            x: Array to be checked against the bounds.

        Returns:
            Numpy array of booleans indicating if a row lies inside the bounds.

        """
        match = self.clip(x) == x
        return match.all(1).flatten() if len(match.shape) > 1 else match.all()

    def safe_margin(
        self,
        low: Tensor | Scalar = None,
        high: Tensor | Scalar | None = None,
        scale: float = 1.0,
    ) -> "Bounds":
        """Initialize a new :class:`Bounds` with its bounds increased o decreased \
        by an scale factor.

        This is done multiplying both high and low for a given factor. The value of the new \
        high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            high: Used to scale the `high` value of the current instance.
            low: Used to scale the `low` value of the current instance.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` with scaled high and low values.

        """
        xmax = self.high if high is None else high
        xmin = self.low if low is None else low
        xmin_scaled, xmax_scaled = self.get_scaled_intervals(xmin, xmax, scale)
        return Bounds(low=xmin_scaled, high=xmax_scaled)

    def to_tuples(self) -> tuple[tuple[Scalar, Scalar], ...]:
        """Return a tuple of tuples containing the lower and higher bound for each \
        coordinate of the :class:`Bounds` shape.

        Returns
            Tuple of the form ((x0_low, x0_high), (x1_low, x1_high), ...,\
              (xn_low, xn_high))

        Examples
            >>> import torch
            >>> array = torch.tensor([1, 2, 5])
            >>> bounds = Bounds(high=array, low=-array)
            >>> print(bounds.to_tuples())
            ((tensor(-1), tensor(1)), (tensor(-2), tensor(2)), (tensor(-5), tensor(5)))

        """
        return tuple(zip(self.low, self.high))

    def to_space(self) -> "gym.spaces.box.Box":  # noqa: F821
        """Return a :class:`Box` gym space with the same characteristics as the :class:`Bounds`."""
        from gym.spaces.box import Box  # noqa:PLC0415

        high = einops.asnumpy(self.high)
        return Box(low=einops.asnumpy(self.low), high=high, dtype=high.dtype)

    def points_in_bounds(self, x: Tensor) -> Tensor | bool:
        """Check if the rows of the target array have all their coordinates inside \
        specified bounds.

        If the array is one dimensional it will return a boolean, otherwise a vector of booleans.

        Args:
            x: Array to be checked against the bounds.

        Returns:
            Numpy array of booleans indicating if a row lies inside the bounds.

        """
        match = self.clip(x) == x.to(self.low)
        return match.all(1).flatten() if len(match.shape) > 1 else match.all()

    def sample(self, num_samples: int = 1) -> Tensor:
        """Sample a batch of random values within the bounds.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Tensor of shape (num_samples, *self.shape) containing random samples within the bounds.

        """
        # Determine the shape of the samples
        shape = (num_samples, *self.shape)
        dtype = self.dtype if self.dtype != torch.long else torch.float64
        rand = torch.rand(shape, dtype=dtype, device=self.high.device)
        return self.low + (self.high - self.low) * rand


class NumpyBounds:
    """The :class:`Bounds` implements the logic for defining and managing closed intervals, \
    and checking if a numpy array's values are inside a given interval.

    It is used on a numpy array of a target shape.
    """

    def __init__(
        self,
        high: numpy.ndarray | Scalar = numpy.inf,
        low: numpy.ndarray | Scalar = -numpy.inf,
        shape: tuple | None = None,
        dtype: type | None = None,
    ):
        """Initialize a :class:`Bounds`.

        Args:
            high: Higher value for the bound interval. If it is an typing_.Scalar \
                  it will be applied to all the coordinates of a target vector. \
                  If it is a vector, the bounds will be checked coordinate-wise. \
                  It defines and closed interval.
            low: Lower value for the bound interval. If it is a typing_.Scalar it \
                 will be applied to all the coordinates of a target vector. \
                 If it is a vector, the bounds will be checked coordinate-wise. \
                 It defines and closed interval.
            shape: Shape of the array that will be bounded. Only needed if `high` and `low` are \
                   vectors, and it is used to define the dimensions that will be bounded.
            dtype:  Data type of the array that will be bounded. It can be inferred from `high` \
                    or `low` (the type of `high` takes priority).

        Examples:
            Initializing :class:`Bounds` using  numpy arrays:

            >>> high, low = np.ones(3, dtype=np.float32), -1 * np.ones(3, dtype=np.int32)
            >>> bounds = Bounds(high=high, low=low)
            >>> print(bounds)
            NumpyBounds shape float32 dtype (3,) low [-1. -1. -1.] high [1. 1. 1.]

            Initializing :class:`Bounds` using  typing_.Scalars:

            >>> high, low, shape = 4, 2.1, (5,)
            >>> bounds = NumpyBounds(high=high, low=low, shape=shape)
            >>> print(bounds)
            NumpyBounds shape float64 dtype (5,) low [2.1 2.1 2.1 2.1 2.1] high [4. 4. 4. 4. 4.]

        """
        # Infer shape if not specified
        if shape is None and hasattr(high, "shape"):
            shape = high.shape
        elif shape is None and hasattr(low, "shape"):
            shape = low.shape
        elif shape is None:
            msg = "If shape is None high or low need to have .shape attribute."
            raise TypeError(msg)
        # High and low will be arrays of target shape
        if not isinstance(high, numpy.ndarray):
            high = numpy.array(high) if isinstance(high, _Iterable) else (numpy.ones(shape) * high)
        if not isinstance(low, numpy.ndarray):
            low = numpy.array(low) if isinstance(low, _Iterable) else (numpy.ones(shape) * low)
        self.high = high.astype(dtype)
        self.low = low.astype(dtype)
        self._bounds_dist = self.high - self.low
        if dtype is not None:
            self.dtype = dtype
        elif hasattr(high, "dtype"):
            self.dtype = high.dtype
        elif hasattr(low, "dtype"):
            self.dtype = low.dtype
        else:
            self.dtype = type(high) if high is not None else type(low)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} shape {self.dtype} dtype "
            f"{self.shape} low {self.low} high {self.high}"
        )

    def __len__(self) -> int:
        """Return the number of dimensions of the bounds."""
        return len(self.high)

    def __contains__(self, item):
        return self.contains(item)

    @property
    def shape(self) -> tuple:
        """Get the shape of the current bounds.

        Returns
            tuple containing the shape of `high` and `low`

        """
        return self.high.shape

    @classmethod
    def from_tuples(cls, bounds: Iterable[tuple], dtype=numpy.float32) -> "NumpyBounds":
        """Instantiate a :class:`Bounds` from a collection of tuples containing \
        the higher and lower bounds for every dimension as a tuple.

        Args:
            bounds: Iterable that returns tuples containing the higher and lower \
                    bound for every dimension of the target bounds.
            dtype: Data type of the array that will be bounded. Default is numpy.float32.

        Returns:
                :class:`Bounds` instance.

        Examples:
            >>> intervals = ((-1., 1.), (-2., 1.), (2, 3))
            >>> bounds = NumpyBounds.from_tuples(intervals)
            >>> print(bounds)
            NumpyBounds shape float32 dtype (3,) low [-1. -2.  2.] high [1. 1. 3.]

        """
        low, high = [], []
        for lo, hi in bounds:
            low.append(lo)
            high.append(hi)
        low = numpy.array(low, dtype=dtype)
        high = numpy.array(high, dtype=dtype)
        return Bounds(low=low, high=high)

    @classmethod
    def from_space(cls, space: "gym.spaces.box.Box") -> "NumpyBounds":  # noqa: F821
        """Initialize a :class:`Bounds` from a :class:`Box` gym action space."""
        return NumpyBounds(low=space.low, high=space.high, dtype=space.dtype)

    @staticmethod
    def get_scaled_intervals(
        low: numpy.ndarray | (float | int),
        high: numpy.ndarray | (float | int),
        scale: float,
    ) -> tuple[Tensor | float, Tensor | float]:
        """Scale the high and low vectors by a scale factor.

        The value of the high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            high: Higher bound to be scaled.
            low: Lower bound to be scaled.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` instance.

        """
        pct = numpy.array(scale - 1)
        big_scale = 1 + numpy.abs(pct)
        small_scale = 1 - numpy.abs(pct)
        zero = numpy.array(0.0).astype(low.dtype)
        if pct > 0:
            xmin_scaled = numpy.where(low < zero, low * big_scale, low * small_scale)
            xmax_scaled = numpy.where(high < zero, high * small_scale, high * big_scale)
        else:
            xmin_scaled = numpy.where(low < zero, low * small_scale, low * small_scale)
            xmax_scaled = numpy.where(high < zero, high * big_scale, high * small_scale)
        return xmin_scaled, xmax_scaled

    @classmethod
    def from_array(cls, x: Tensor | numpy.ndarray, scale: float = 1.0) -> "NumpyBounds":
        """Instantiate a bounds compatible for bounding the given array. It also allows to set a \
        margin for the high and low values.

        The value of the high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            x: Numpy array used to initialize the bounds.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` instance.

        Examples:
            >>> x = np.ones((3, 3))
            >>> x[1:-1, 1:-1] = -5
            >>> bounds = Bounds.from_array(x, scale=1.5)
            >>> print(bounds)
            NumpyBounds shape float64 dtype (3,) low [ 0.5 -7.5  0.5] high [1.5 1.5 1.5]

        """
        if isinstance(x, Tensor):
            x = x.numpy(force=True)
        xmin, xmax = numpy.min(x, axis=0), numpy.max(x, axis=0)
        xmin_scaled, xmax_scaled = cls.get_scaled_intervals(xmin, xmax, scale)
        return Bounds(low=xmin_scaled, high=xmax_scaled)

    def clip(self, x: numpy.ndarray) -> numpy.ndarray:
        """Clip the values of the target array to fall inside the bounds (closed interval).

        Args:
            x: Numpy array to be clipped.

        Returns:
            Clipped numpy array with all its values inside the defined bounds.

        """
        return numpy.clip(einops.asnumpy(x).astype(self.low.dtype), self.low, self.high)

    def pbc(self, x: numpy.ndarray) -> numpy.ndarray:
        """Calculate periodic boundary conditions of the target array to fall inside \
        the bounds (closed interval).

        Args:
            x: Tensor to apply the periodic boundary conditions.

        Returns:
            Periodic boundary condition so all the values are inside the defined bounds.

        """
        x = einops.asnumpy(x).astype(self.low.dtype)
        x = numpy.where(x < self.high, x, numpy.mod(x, self.high) + self.low)
        return numpy.where(x > self.low, x, self.high - numpy.mod(x, self.low))

    def pbc_distance(self, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        """Calculate periodic boundary conditions of the target array to fall inside \
        the bounds (closed interval).

        Args:
            x: Tensor to apply the periodic boundary conditions.
            y: Tensor containing the frontier of the periodic boundary condition.

        Returns:
            Periodic boundary condition so all the values are inside the defined bounds.

        """
        x, y = einops.asnumpy(x).astype(self.low), einops.asnumpy(y).astype(self.low)
        delta = numpy.abs(x - y)
        return numpy.where(x > 0.5 * self._bounds_dist, delta - self._bounds_dist, delta)

    def contains(self, x: numpy.ndarray) -> numpy.ndarray | bool:
        """Check if the rows of the target array have all their coordinates inside \
        specified bounds.

        If the array is one dimensional it will return a boolean, otherwise a vector of booleans.

        Args:
            x: Array to be checked against the bounds.

        Returns:
            Numpy array of booleans indicating if a row lies inside the bounds.

        """
        match = self.clip(x) == einops.asnumpy(x).astype(self.low.dtype)
        return match.all(1).flatten() if len(match.shape) > 1 else match.all()

    def safe_margin(
        self,
        low: numpy.ndarray | Scalar = None,
        high: numpy.ndarray | Scalar | None = None,
        scale: float = 1.0,
    ) -> "NumpyBounds":
        """Initialize a new :class:`Bounds` with its bounds increased o decreased \
        by a scale factor.

        This is done multiplying both high and low for a given factor. The value of the new \
        high and low will be proportional to the maximum and minimum values of \
        the array. Scale defines the proportion to make the bounds bigger and smaller. For \
        example, if scale is 1.1 the higher bound will be 10% higher, and the lower bounds 10% \
        smaller. If scale is 0.9 the higher bound will be 10% lower, and the lower bound 10% \
        higher. If scale is one, `high` and `low` will be equal to the maximum and minimum values \
        of the array.

        Args:
            high: Used to scale the `high` value of the current instance.
            low: Used to scale the `low` value of the current instance.
            scale: Value representing the tolerance in percentage from the current maximum and \
            minimum values of the array.

        Returns:
            :class:`Bounds` with scaled high and low values.

        """
        xmax = self.high if high is None else einops.asnumpy(high)
        xmin = self.low if low is None else einops.asnumpy(low)
        xmin_scaled, xmax_scaled = self.get_scaled_intervals(xmin, xmax, scale)
        return Bounds(low=xmin_scaled, high=xmax_scaled)

    def to_tuples(self) -> tuple[tuple[Scalar, Scalar], ...]:
        """Return a tuple of tuples containing the lower and higher bound for each \
        coordinate of the :class:`Bounds` shape.

        Returns
            Tuple of the form ((x0_low, x0_high), (x1_low, x1_high), ...,\
              (xn_low, xn_high))

        Examples
            >>> array = np.array([1, 2, 5])
            >>> bounds = Bounds(high=array, low=-array)
            >>> print(bounds.to_tuples())
            ((np.float64(-1.0), np.float64(1.0)), (np.float64(-2.0), np.float64(2.0)), \
(np.float64(-5.0), np.float64(5.0)))


        """
        return tuple(zip(self.low, self.high))

    def to_space(self) -> "gym.spaces.box.Box":  # noqa: F821
        """Return a :class:`Box` gym space with the same characteristics as the :class:`Bounds`."""
        from gym.spaces.box import Box  # noqa:PLC0415

        return Box(low=self.low, high=self.high, dtype=self.high.dtype)

    def points_in_bounds(self, x: numpy.ndarray) -> numpy.ndarray | bool:
        """Check if the rows of the target array have all their coordinates inside \
        specified bounds.

        If the array is one dimensional it will return a boolean, otherwise a vector of booleans.

        Args:
            x: Array to be checked against the bounds.

        Returns:
            Numpy array of booleans indicating if a row lies inside the bounds.

        """
        match = self.clip(x) == einops.asnumpy(x).astype(self.low.dtype)
        return match.all(1).flatten() if len(match.shape) > 1 else match.all()

    def sample(self, num_samples: int = 1) -> numpy.ndarray:
        """Sample a batch of random values within the bounds.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Numpy array of shape (num_samples, *self.shape) containing random samples
            within the bounds.

        """
        shape = (num_samples, *self.shape)
        rand = numpy.random.rand(*shape).astype(self.dtype)  # noqa: NPY002
        return self.low + (self.high - self.low) * rand
