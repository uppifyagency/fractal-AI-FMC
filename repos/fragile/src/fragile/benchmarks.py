from collections.abc import Callable
import math

import einops
from numba import jit
import numpy as np
import torch

from fragile.bounds import Bounds, NumpyBounds, TorchBounds


"""
This file includes several test functions for optimization described here:
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x**2, 1).flatten()


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    dims = x.shape[1]
    a = 10
    result = a * dims + torch.sum(x**2 - a * torch.cos(2 * math.pi * x), 1)
    return result.flatten()


def eggholder(x: torch.Tensor) -> torch.Tensor:
    x, y = x[:, 0], x[:, 1]
    first_root = torch.sqrt(torch.abs(x / 2.0 + (y + 47)))
    second_root = torch.sqrt(torch.abs(x - (y + 47)))
    return -1 * (y + 47) * torch.sin(first_root) - x * torch.sin(second_root)


def styblinski_tang(x) -> torch.Tensor:
    return torch.sum(x**4 - 16 * x**2 + 5 * x, 1) / 2.0


def rosenbrock(x) -> torch.Tensor:
    return 100 * torch.sum((x[:, :-2] ** 2 - x[:, 1:-1]) ** 2, 1) + torch.sum(
        (x[:, :-2] - 1) ** 2,
        1,
    )


def easom(x) -> torch.Tensor:
    exp_term = (x[:, 0] - np.pi) ** 2 + (x[:, 1] - np.pi) ** 2
    return -torch.cos(x[:, 0]) * torch.cos(x[:, 1]) * torch.exp(-exp_term)


def holder_table(x) -> torch.Tensor:
    x_, y = x[:, 0], x[:, 1]
    exp = torch.abs(1 - (torch.sqrt(x_ * x_ + y * y) / np.pi))
    return -torch.abs(torch.sin(x_) * torch.cos(y) * torch.exp(exp))


@jit(nopython=True)
def _lennard_fast(state):
    state = state.reshape(-1, 3)
    npart = len(state)
    epot = 0.0
    for i in range(npart):
        for j in range(npart):
            if i > j:
                r2 = np.sum((state[j, :] - state[i, :]) ** 2)
                r2i = 1.0 / r2
                r6i = r2i * r2i * r2i
                epot += r6i * (r6i - 1.0)
    return epot * 4


def lennard_jones(x: torch.Tensor) -> torch.Tensor:
    result = np.zeros(x.shape[0])
    x_ = einops.asnumpy(x)
    # assert isinstance(x, torch.Tensor)
    for i in range(x.shape[0]):
        try:
            result[i] = _lennard_fast(x_[i])
        except ZeroDivisionError:  # noqa: PERF203
            result[i] = np.inf
    return torch.from_numpy(result).to(x)


class OptimBenchmark:
    benchmark = None
    best_state = None

    def __init__(self, dims: int, function: Callable, **kwargs):  # noqa: ARG002
        self.dims = dims
        self.bounds = self.get_bounds(dims=dims)
        self.function = function

    def __call__(self, x):
        return self.function(x)

    def sample(self, n_samples):
        return self.bounds.sample(n_samples)

    def sample_action(self, n_samples=1):
        return self.sample(n_samples)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.bounds.shape

    @staticmethod
    def get_bounds(dims: int) -> Bounds | TorchBounds | NumpyBounds:
        raise NotImplementedError


class Sphere(OptimBenchmark):
    benchmark = torch.tensor(0.0)

    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=sphere, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-1000, 1000) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.zeros(self.shape)


class Rastrigin(OptimBenchmark):
    benchmark = torch.tensor(0.0)

    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=rastrigin, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-5.12, 5.12) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.zeros(self.shape)


class EggHolder(OptimBenchmark):
    benchmark = torch.tensor(-959.64066271)

    def __init__(self, dims: int | None = None, **kwargs):  # noqa: ARG002
        super().__init__(dims=2, function=eggholder, **kwargs)

    @staticmethod
    def get_bounds(dims=None):  # noqa: ARG004
        bounds = [(-512.0, 512.0), (-512.0, 512.0)]
        # bounds = [(1, 512.0), (1, 512.0)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.tensor([512.0, 404.2319])


class StyblinskiTang(OptimBenchmark):
    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=styblinski_tang, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-5.0, 5.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.ones(self.shape) * -2.903534

    @property
    def benchmark(self):
        return torch.tensor(-39.16617 * self.shape[0])


class Rosenbrock(OptimBenchmark):
    def __init__(self, dims: int, **kwargs):
        super().__init__(dims=dims, function=rosenbrock, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-10.0, 10.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.ones(self.shape)

    @property
    def benchmark(self):
        return torch.tensor(0.0)


class Easom(OptimBenchmark):
    def __init__(self, dims: int | None = None, **kwargs):  # noqa: ARG002
        super().__init__(dims=2, function=easom, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-100.0, 100.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.ones(self.shape) * np.pi

    @property
    def benchmark(self):
        return torch.tensor(-1)


class HolderTable(OptimBenchmark):
    def __init__(self, dims: int | None = None, *args, **kwargs):  # noqa: ARG002
        super().__init__(dims=2, function=holder_table, *args, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-10.0, 10.0) for _ in range(dims)]
        return Bounds.from_tuples(bounds)

    @property
    def best_state(self):
        return torch.tensor([8.05502, 9.66459])

    @property
    def benchmark(self):
        return torch.tensor(-19.2085)


class LennardJones(OptimBenchmark):
    # http://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html
    minima = {
        "2": -1,
        "3": -3,
        "4": -6,
        "5": -9.103852,
        "6": -12.712062,
        "7": -16.505384,
        "8": -19.821489,
        "9": -24.113360,
        "10": -28.422532,
        "11": -32.765970,
        "12": -37.967600,
        "13": -44.326801,
        "14": -47.845157,
        "15": -52.322627,
        "20": -77.177043,
        "25": -102.372663,
        "30": -128.286571,
        "38": -173.928427,
        "50": -244.549926,
        "100": -557.039820,
        "104": -582.038429,
    }

    benchmark = None

    def __init__(self, n_atoms: int = 10, dims=None, **kwargs):  # noqa: ARG002
        self.n_atoms = n_atoms
        self.dims = 3 * n_atoms
        self.benchmark = [torch.zeros(self.n_atoms * 3), self.minima.get(str(int(n_atoms)), 0)]
        super().__init__(dims=self.dims, function=lennard_jones, **kwargs)

    @staticmethod
    def get_bounds(dims):
        bounds = [(-15, 15) for _ in range(dims)]
        return Bounds.from_tuples(bounds)


ALL_BENCHMARKS = [Sphere, Rastrigin, EggHolder, StyblinskiTang, HolderTable, Easom]
