import numpy as np
import torch

from fragile.core import BaseDtSampler, BaseFractalTree, BasePolicy, FractalTree
from fragile.fractalai import clone_tensor


class UniformDtSampler(BaseDtSampler):
    def __init__(self, min_dt: int = 1, max_dt: int = 5, fractal: BaseFractalTree | None = None):
        super().__init__(fractal=fractal)
        self.max_dt = max_dt
        self.min_dt = min_dt

    def get_dt(self, n_walkers: int | None = None, fractal: BaseFractalTree | None = None):
        if n_walkers is None:
            n_walkers = fractal.n_walkers
        return np.random.randint(self.min_dt, self.max_dt, size=n_walkers)  # noqa: NPY002


class RandomGaussianPolicy(BasePolicy):
    def __init__(
        self,
        std: float = 1.0,
        min: float | None = None,
        max: float | None = None,
        fractal: BaseFractalTree | None = None,
    ):
        super().__init__(fractal=fractal)
        self.std = std
        self.min_ = min
        self.max_ = max

    def act(self, n_walkers: int | None = None, fractal: FractalTree | None = None):
        fractal = fractal if fractal is not None else self.fractal
        if n_walkers is None:
            n_walkers = fractal.n_walkers
        return (torch.randn((n_walkers, *fractal.action_shape)) * self.std).clamp(
            self.min_, self.max_
        )


class GaussianForce(RandomGaussianPolicy):
    def __init__(
        self,
        std: float = 1.0,
        min: float | None = None,
        max: float | None = None,
        fractal: FractalTree | None = None,
    ):
        super().__init__(std=std, fractal=fractal, min=min, max=max)
        action_shape = fractal.action_shape if fractal is not None else (1,)
        device = fractal.device if fractal is not None else "cpu"
        n_walkers = fractal.max_walkers if fractal is not None else 1
        self._velocity = torch.zeros((n_walkers, *action_shape), device=device)

    @property
    def velocity(self):
        return self._velocity[: self.fractal.n_walkers]

    def set_fractal(self, fractal: "FractalTree"):
        super().set_fractal(fractal)
        self._velocity = torch.zeros(
            (fractal.max_walkers, *fractal.action_shape), device=fractal.device
        )

    def act(self, n_walkers: int | None = None, fractal: FractalTree | None = None):
        action = super().act(n_walkers=n_walkers, fractal=fractal)
        wc = (
            self.fractal.will_clone
            if self.fractal.will_clone.sum() > 0
            else torch.ones_like(self.fractal.will_clone)
        )
        self.velocity[wc] += action
        return self.velocity[wc].clamp(self.min_, self.max_)

    def clone(self, will_clone: torch.Tensor, clone_ix: torch.Tensor):
        self.velocity[:] = clone_tensor(self.velocity, clone_ix, will_clone)

    def add_walkers(self, new_walkers):
        new_vel = torch.zeros((new_walkers, *self.velocity.shape[1:]), device=self.velocity.device)
        self.velocity[:] = torch.cat((self.velocity, new_vel), dim=0).contiguous()
