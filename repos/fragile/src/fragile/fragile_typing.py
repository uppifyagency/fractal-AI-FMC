"""The :class:`Component` class is the base class for all components of the fragile framework."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import numpy
import torch


Scalar = Any
Device = str | torch.device
Dtype = Any
Shape = tuple[int, ...]
ValueType = str
Default = Any

# Values are those types that are passed as inputs to the component or returned by forward.
# The contain the population values of the state.
Value = torch.Tensor | numpy.ndarray | list
DictValues = dict[str, Value]
ValueWalker = torch.Tensor | numpy.ndarray | list  # Batch size 1 and represents a single walker

# Values passed and returned by step and reset
InputValues = dict[str, Value]  # Values passed to forward and warmup
OutputValues = dict[str, Value]  # Values returned by forward
ResetValues = dict[str, Value]  # Values returned by warmup
DictWalker = dict[str, Value]  # Represents data from a single walker
DictTensor = dict[str, torch.Tensor]
Walkers = list[DictWalker]

IndexVector = torch.Tensor
BooleanValue = torch.Tensor
NumpyIndex = numpy.ndarray


# Configs are those types that are used to describe the data that the component processes.
class ValueConfig(ABC):
    type: str = None
    shape: Shape
    dtype: Dtype
    default: Default

    @property
    @abstractmethod
    def example(self) -> Value:
        """Return an example value with batch size 1."""


ConfigValue = dict[str, Any] | ValueConfig
ValuesConfig = dict[str, ConfigValue]
InputsConfig = dict[str, dict[str, bool]]
OutputsConfig = tuple[str, ...]
ResetsConfig = tuple[str, ...]

NUMPY_TYPE = "numpy"
TORCH_TYPE = "torch"
LIST_TYPE = "list"
CONFIG_TYPES = {NUMPY_TYPE, TORCH_TYPE, LIST_TYPE}

HASH_DTYPE = numpy.uint64

NodeId = str | int
NodeData = tuple[dict, dict] | tuple[dict, dict, dict]
NodeDataGenerator = Generator[NodeData, None, None]
NamesData = tuple[str] | set[str] | list[str]
