from collections.abc import Callable
import logging
import threading
import time

import einops
import numpy as np
import pandas as pd
import panel as pn
import param
import torch

from fragile.benchmarks import Rastrigin
from fragile.fractalai import (
    calculate_clone,
    clone_tensor,
    fai_iteration,
    l2_norm,
    random_alive_compas,
    relativize,
)
from fragile.utils import numpy_dtype_to_torch_dtype


logger = logging.getLogger(__name__)


def get_is_cloned(compas_ix, will_clone):
    target = torch.zeros_like(will_clone)
    cloned_to = compas_ix[will_clone].unique()
    target[cloned_to] = True
    return target


def get_is_leaf(parents):
    is_leaf = torch.ones_like(parents, dtype=torch.bool)
    is_leaf[parents] = False
    return is_leaf


def step(state, action, env):
    """Step the environment."""
    dt = np.random.randint(1, 4, size=state.shape[0])  # noqa: NPY002
    data = env.step(state=state, action=action, dts=dt)
    new_state, observ, reward, end, _truncated, info = data
    return new_state, observ, reward, end, info


def aggregate_visits(array, block_size=5, upsample=True):
    """
    Aggregates the input array over blocks in the last two dimensions.

    Parameters:
    - array (numpy.ndarray): Input array with shape (batch_size, width, height).
    - block_size (int): Size of the block over which to aggregate.

    Returns:
    - numpy.ndarray: Aggregated array with reduced dimensions.
    """
    batch_size, width, height = array.shape
    new_width = width // block_size
    new_height = height // block_size

    # Ensure that width and height are divisible by block_size
    if width % block_size != 0 or height % block_size != 0:
        msg = (
            "Width and height must be divisible by block_size. "
            "Got width: {}, height: {}, block_size: {}"
        )
        raise ValueError(msg.format(width, height, block_size))

    reshaped_array = array.reshape(batch_size, new_width, block_size, new_height, block_size)
    aggregated_array = reshaped_array.sum(axis=(2, 4))
    if not upsample:
        return aggregated_array
    return np.repeat(np.repeat(aggregated_array, block_size, axis=1), block_size, axis=2)


class BasePolicy(param.Parameterized):
    def __init__(self, fractal: "BaseFractalTree | None" = None, **kwargs):
        super().__init__(**kwargs)
        self._fractal = fractal

    def __call__(
        self, n_samples: int | None = None, fractal: "BaseFractalTree | None" = None
    ) -> torch.Tensor:
        return self.act(n_samples, fractal)

    @property
    def fractal(self):
        return self._fractal

    def set_fractal(self, fractal: "BaseFractalTree"):
        self._fractal = fractal

    def act(
        self, n_samples: int | None = None, fractal: "BaseFractalTree | None" = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def clone(self, will_clone: torch.Tensor, clone_ix: torch.Tensor):
        pass

    def add_walkers(self, new_walkers):
        pass


class BaseDtSampler(param.Parameterized):
    def __init__(self, fractal: "BaseFractalTree | None" = None, **kwargs):
        super().__init__(**kwargs)
        self._fractal = fractal

    def __call__(
        self, n_samples: int | None = None, fractal: "BaseFractalTree | None" = None
    ) -> np.ndarray:
        fractal = fractal if fractal is not None else self.fractal
        return self.get_dt(n_samples, fractal)

    @property
    def fractal(self):
        return self._fractal

    def set_fractal(self, fractal: "BaseFractalTree"):
        self._fractal = fractal

    def get_dt(
        self, n_samples: int | None = None, fractal: "BaseFractalTree | None" = None
    ) -> np.ndarray:
        raise NotImplementedError

    def clone(self, will_clone: torch.Tensor, clone_ix: torch.Tensor):
        pass

    def add_walkers(self, new_walkers):
        pass


class RandomPolicy(BasePolicy):
    def act(
        self, n_samples: int | None = None, fractal: "FractalTree | None" = None
    ) -> torch.Tensor:
        n_walkers = n_samples if n_samples is not None else fractal.n_walkers
        acts = [fractal.env.sample_action() for _ in range(n_walkers)]
        if isinstance(acts[0], torch.Tensor) and len(acts) > 1:
            return torch.concat(acts, dim=0)
        if isinstance(acts[0], torch.Tensor):
            return acts[0]
        return torch.tensor(acts, device=fractal.device, dtype=fractal.action_type)


class ConstantDt(BaseDtSampler):
    def __init__(self, dt: float = 1.0, fractal: "BaseFractalTree | None" = None):
        super().__init__(fractal)
        self.dt = dt

    def get_dt(
        self, n_samples: int | None = None, fractal: "BaseFractalTree | None" = None
    ) -> np.ndarray:
        n_walkers = n_samples if n_samples is not None else fractal.n_walkers
        return np.ones(n_walkers) * self.dt


class _BaseFractalTree:
    def __init__(
        self,
        max_walkers,
        env,
        device="cuda",
        start_walkers: int = 100,
        min_leafs: int = 100,
    ):
        self.max_walkers = max_walkers
        self.n_walkers = start_walkers
        self.start_walkers = start_walkers
        self.min_leafs = min_leafs
        self.env = env
        self.device = device

        self.total_steps = 0
        self.iteration = 0

        self.parent = torch.zeros(start_walkers, device=self.device, dtype=torch.long)
        self.is_leaf = torch.ones(start_walkers, device=self.device, dtype=torch.long)
        self.can_clone = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.is_cloned = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_distance = torch.ones(start_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_clone = torch.ones(start_walkers, device=self.device, dtype=torch.bool)
        self.is_dead = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)

        self.reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.cum_reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.oobs = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)

        self.virtual_reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.clone_prob = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.clone_ix = torch.zeros(start_walkers, device=self.device, dtype=torch.int64)
        self.distance_ix = torch.zeros(start_walkers, device=self.device, dtype=torch.int64)
        self.wants_clone = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.will_clone = torch.ones(start_walkers, device=self.device, dtype=torch.bool)
        self.distance = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.scaled_distance = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.scaled_reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)

    def reset_tensors(self):
        start_walkers = self.start_walkers
        self.parent = torch.zeros(start_walkers, device=self.device, dtype=torch.long)
        self.is_leaf = torch.ones(start_walkers, device=self.device, dtype=torch.long)
        self.can_clone = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.is_cloned = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_distance = torch.ones(start_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_clone = torch.ones(start_walkers, device=self.device, dtype=torch.bool)
        self.is_dead = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)

        self.reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.cum_reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.oobs = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)

        self.virtual_reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.clone_prob = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.clone_ix = torch.zeros(start_walkers, device=self.device, dtype=torch.int64)
        self.distance_ix = torch.zeros(start_walkers, device=self.device, dtype=torch.int64)
        self.wants_clone = torch.zeros(start_walkers, device=self.device, dtype=torch.bool)
        self.will_clone = torch.ones(start_walkers, device=self.device, dtype=torch.bool)
        self.distance = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.scaled_distance = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)
        self.scaled_reward = torch.zeros(start_walkers, device=self.device, dtype=torch.float32)

    def add_walkers(self, new_walkers):
        self.n_walkers += new_walkers

        parent = torch.zeros(new_walkers, device=self.device, dtype=torch.long)
        self.parent = torch.cat((self.parent, parent), dim=0).contiguous()

        is_leaf = torch.ones(new_walkers, device=self.device, dtype=torch.long)
        self.is_leaf = torch.cat((self.is_leaf, is_leaf), dim=0).contiguous()

        can_clone = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.can_clone = torch.cat((self.can_clone, can_clone), dim=0).contiguous()

        is_cloned = torch.zeros(new_walkers, device=self.device, dtype=torch.bool)
        self.is_cloned = torch.cat((self.is_cloned, is_cloned), dim=0).contiguous()

        is_compa_distance = torch.zeros(new_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_distance = torch.cat(
            (self.is_compa_distance, is_compa_distance), dim=0
        ).contiguous()

        is_compa_clone = torch.zeros(new_walkers, device=self.device, dtype=torch.bool)
        self.is_compa_clone = torch.cat((self.is_compa_clone, is_compa_clone), dim=0).contiguous()

        is_dead = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.is_dead = torch.cat((self.is_dead, is_dead), dim=0).contiguous()

        reward = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.reward = torch.cat((self.reward, reward), dim=0).contiguous()

        cum_reward = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.cum_reward = torch.cat((self.cum_reward, cum_reward), dim=0).contiguous()

        oobs = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.oobs = torch.cat((self.oobs, oobs), dim=0).contiguous()

        virtual_reward = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.virtual_reward = torch.cat((self.virtual_reward, virtual_reward), dim=0).contiguous()

        clone_prob = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.clone_prob = torch.cat((self.clone_prob, clone_prob), dim=0).contiguous()

        clone_ix = torch.zeros(new_walkers, device=self.device, dtype=torch.int64)
        self.clone_ix = torch.cat((self.clone_ix, clone_ix), dim=0).contiguous()

        distance_ix = torch.zeros(new_walkers, device=self.device, dtype=torch.int64)
        self.distance_ix = torch.cat((self.distance_ix, distance_ix), dim=0).contiguous()

        wants_clone = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.wants_clone = torch.cat((self.wants_clone, wants_clone), dim=0).contiguous()

        will_clone = torch.ones(new_walkers, device=self.device, dtype=torch.bool)
        self.will_clone = torch.cat((self.will_clone, will_clone), dim=0).contiguous()

        distance = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.distance = torch.cat((self.distance, distance), dim=0).contiguous()

        scaled_distance = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.scaled_distance = torch.cat(
            (self.scaled_distance, scaled_distance), dim=0
        ).contiguous()

        scaled_reward = torch.zeros(new_walkers, device=self.device, dtype=torch.float32)
        self.scaled_reward = torch.cat((self.scaled_reward, scaled_reward), dim=0).contiguous()


class BaseFractalTree(param.Parameterized):
    max_walkers = param.Integer(default=1000, bounds=(2, None), per_instance=True)
    start_walkers = param.Integer(default=100, bounds=(2, None), per_instance=True)
    min_leafs = param.Integer(default=100, bounds=(2, None), per_instance=True)
    n_walkers = param.Integer(default=100, bounds=(2, None), per_instance=True)

    def __init__(
        self,
        max_walkers,
        env,
        device="cuda",
        start_walkers: int = 100,
        min_leafs: int = 100,
        **kwargs,
    ):
        super().__init__(
            max_walkers=max_walkers,
            start_walkers=start_walkers,
            min_leafs=min_leafs,
            n_walkers=start_walkers,
            **kwargs,
        )
        self.env = env
        self.device = device

        self.total_steps = 0
        self.iteration = 0

        self._parent = torch.zeros(max_walkers, device=self.device, dtype=torch.long)
        self._is_leaf = torch.ones(max_walkers, device=self.device, dtype=torch.long)
        self._can_clone = torch.zeros(max_walkers, device=self.device, dtype=torch.bool)
        self._is_cloned = torch.zeros(max_walkers, device=self.device, dtype=torch.bool)
        self._is_compa_distance = torch.ones(max_walkers, device=self.device, dtype=torch.bool)
        self._is_compa_clone = torch.ones(max_walkers, device=self.device, dtype=torch.bool)
        self._is_dead = torch.zeros(max_walkers, device=self.device, dtype=torch.bool)

        self._reward = torch.zeros(max_walkers, device=self.device, dtype=torch.float32)
        self._cum_reward = torch.zeros(max_walkers, device=self.device, dtype=torch.float32)
        self._oobs = torch.zeros(max_walkers, device=self.device, dtype=torch.bool)

        self._virtual_reward = torch.zeros(max_walkers, device=self.device, dtype=torch.float32)
        self._clone_prob = torch.zeros(max_walkers, device=self.device, dtype=torch.float32)
        self._clone_ix = torch.zeros(max_walkers, device=self.device, dtype=torch.int64)
        self._distance_ix = torch.zeros(max_walkers, device=self.device, dtype=torch.int64)
        self._wants_clone = torch.zeros(max_walkers, device=self.device, dtype=torch.bool)
        self._will_clone = torch.ones(max_walkers, device=self.device, dtype=torch.bool)
        self._distance = torch.zeros(max_walkers, device=self.device, dtype=torch.float32)
        self._scaled_distance = torch.zeros(max_walkers, device=self.device, dtype=torch.float32)
        self._scaled_reward = torch.zeros(max_walkers, device=self.device, dtype=torch.float32)

    def widgets(self) -> dict[str, pn.widgets.Widget]:
        return {
            "max_walkers": pn.widgets.IntInput(
                name="max_walkers", value=self.max_walkers, width=100
            ),
            "start_walkers": pn.widgets.IntInput(
                name="start_walkers", value=self.start_walkers, width=100
            ),
            "min_leafs": pn.widgets.IntInput(name="min_leafs", value=self.min_leafs, width=100),
            "n_walkers": pn.widgets.IntInput(name="n_walkers", value=self.n_walkers, width=100),
        }

    def reset_tensors(self):
        self._parent = torch.zeros_like(self._parent)
        self._is_leaf = torch.ones_like(self._is_leaf)
        self._can_clone = torch.ones_like(self._can_clone)
        self._is_cloned = torch.zeros_like(self._is_cloned)
        self._is_compa_distance = torch.ones_like(self._is_compa_distance)
        self._is_compa_clone = torch.ones_like(self._is_compa_clone)
        self._is_dead = torch.ones_like(self._is_dead)

        self._reward = torch.zeros_like(self._reward)
        self._cum_reward = torch.zeros_like(self._cum_reward)
        self._oobs = torch.ones_like(self._oobs)

        self._virtual_reward = torch.zeros_like(self._virtual_reward)
        self._clone_prob = torch.zeros_like(self._clone_prob)
        self._clone_ix = torch.zeros_like(self._clone_ix)
        self._distance_ix = torch.zeros_like(self._distance_ix)
        self._wants_clone = torch.zeros_like(self._wants_clone)
        self._will_clone = torch.ones_like(self._will_clone)
        self._distance = torch.zeros_like(self._distance)
        self._scaled_distance = torch.zeros_like(self._scaled_distance)
        self._scaled_reward = torch.zeros_like(self._scaled_reward)

    @property
    def parent(self):
        return self._parent[: self.n_walkers]

    @property
    def is_leaf(self):
        return self._is_leaf[: self.n_walkers]

    @property
    def can_clone(self):
        return self._can_clone[: self.n_walkers]

    @property
    def is_cloned(self):
        return self._is_cloned[: self.n_walkers]

    @property
    def is_compa_distance(self):
        return self._is_compa_distance[: self.n_walkers]

    @property
    def is_compa_clone(self):
        return self._is_compa_clone[: self.n_walkers]

    @property
    def is_dead(self):
        return self._is_dead[: self.n_walkers]

    @property
    def reward(self):
        return self._reward[: self.n_walkers]

    @property
    def cum_reward(self):
        return self._cum_reward[: self.n_walkers]

    @property
    def oobs(self):
        return self._oobs[: self.n_walkers]

    @property
    def virtual_reward(self):
        return self._virtual_reward[: self.n_walkers]

    @property
    def clone_prob(self):
        return self._clone_prob[: self.n_walkers]

    @property
    def clone_ix(self):
        return self._clone_ix[: self.n_walkers]

    @property
    def distance_ix(self):
        return self._distance_ix[: self.n_walkers]

    @property
    def wants_clone(self):
        return self._wants_clone[: self.n_walkers]

    @property
    def will_clone(self):
        return self._will_clone[: self.n_walkers]

    @property
    def distance(self):
        return self._distance[: self.n_walkers]

    @property
    def scaled_distance(self):
        return self._scaled_distance[: self.n_walkers]

    @property
    def scaled_reward(self):
        return self._scaled_reward[: self.n_walkers]

    def add_walkers(self, new_walkers):
        self.n_walkers += new_walkers


class FractalTree(BaseFractalTree):
    dist_coef = param.Number(default=1.0, bounds=(0.0, None), per_instance=True)
    reward_coef = param.Number(default=1.0, bounds=(0.0, None), per_instance=True)

    def __init__(
        self,
        max_walkers,
        env,
        policy: BasePolicy | None = None,
        dt_sampler: BaseDtSampler | None = None,
        minimize: bool = False,
        device="cuda",
        start_walkers=100,
        min_leafs=100,
        obs_shape: tuple[int, ...] | None = None,
        obs_dtype: torch.dtype | None = None,
        action_shape: tuple[int, ...] | None = None,
        action_dtype: torch.dtype | None = None,
        state_shape: tuple[int, ...] | None = (),
        state_dtype: torch.dtype | type = object,
        img_shape: tuple[int, ...] | None = None,
        img_dtype: torch.dtype | None = None,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = l2_norm,
        dist_coef: float = 1.0,
        reward_coef: float = 1.0,
        accumulate_reward: bool = True,
    ):
        super().__init__(
            max_walkers=max_walkers,
            env=env,
            device=device,
            start_walkers=start_walkers,
            min_leafs=min_leafs,
            dist_coef=dist_coef,
            reward_coef=reward_coef,
        )
        self.accumulate_reward = accumulate_reward
        self.distance_function = distance_function
        self.minimize = minimize
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self._observ = self.reset_observ()

        self.action_shape = action_shape
        self.action_type = action_dtype
        self._action = self.reset_action()

        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self._state = self.reset_state()

        self.img_shape = img_shape
        self.img_dtype = img_dtype
        self._img = self.reset_img()

        self.state_step = None
        self.action_step = None
        self.dt = None
        self.dt_sampler = dt_sampler if dt_sampler is not None else ConstantDt(fractal=self)
        self.policy = policy if policy is not None else RandomPolicy(self)
        self.policy.set_fractal(self)
        self.dt_sampler.set_fractal(self)

    @property
    def best_reward(self) -> float:
        return (
            self.cum_reward.min().cpu().item()
            if self.minimize
            else self.cum_reward.max().cpu().item()
        )

    @property
    def best_ix(self) -> int:
        return (
            self.cum_reward.argmin().cpu().item()
            if self.minimize
            else self.cum_reward.argmax().cpu().item()
        )

    @property
    def observ(self):
        return self._observ[: self.n_walkers]

    @property
    def action(self):
        return self._action[: self.n_walkers]

    @property
    def state(self):
        return self._state[: self.n_walkers]

    @property
    def img(self):
        return self._img[: self.n_walkers]

    def widgets(self) -> dict[str, pn.widgets.Widget]:
        widgets = super().widgets()
        curr_widgets = {
            "dist_coef": pn.widgets.FloatInput(name="dist_coef", value=self.dist_coef, width=100),
            "reward_coef": pn.widgets.FloatInput(
                name="reward_coef", value=self.reward_coef, width=100
            ),
        }
        return {**widgets, **curr_widgets}

    def __panel__(self):
        top_row_names = ["max_walkers", "start_walkers", "min_leafs", "n_walkers"]
        top_widgets = {k: v for k, v in self.widgets().items() if k in top_row_names}
        bot_row_names = ["dist_coef", "reward_coef"]
        bot_widgets = {k: v for k, v in self.widgets().items() if k in bot_row_names}
        return pn.Column(
            pn.pane.Markdown("## Fractal Tree"),
            pn.Row(
                pn.Param(
                    self,
                    parameters=top_row_names,
                    widgets=top_widgets,
                    show_name=False,
                    default_layout=pn.Row,
                ),
                pn.Param(
                    self,
                    parameters=bot_row_names,
                    widgets=bot_widgets,
                    show_name=False,
                    default_layout=pn.Row,
                ),
            ),
        )

    def reset_img(self):
        self.img_shape = self.img_shape if self.img_shape is not None else self.env.img_shape
        if self.img_shape is not None and self.img_dtype is None:
            self.img_dtype = self.env.get_image().dtype
        elif self.img_shape is None:
            self.img_dtype = np.uint8
            self.img_shape = (10, 10, 3)
        return np.zeros((self.max_walkers, *self.img_shape), dtype=self.img_dtype)

    def reset_observ(self):
        self.obs_shape = (
            self.obs_shape if self.obs_shape is not None else self.env.observation_space.shape
        )
        self.obs_dtype = (
            self.obs_dtype
            if self.obs_dtype is not None
            else numpy_dtype_to_torch_dtype(self.env.observation_space.dtype)
        )
        return torch.zeros(
            (self.max_walkers, *self.obs_shape), device=self.device, dtype=self.obs_dtype
        )

    def reset_action(self):
        self.action_shape = (
            self.action_shape if self.action_shape is not None else self.env.action_space.shape
        )
        self.action_type = (
            self.action_type
            if self.action_type is not None
            else numpy_dtype_to_torch_dtype(self.env.action_space.dtype)
        )
        return torch.zeros(
            (self.max_walkers, *self.action_shape), device=self.device, dtype=self.action_type
        )

    def reset_state(self):
        return np.empty((self.max_walkers, *self.state_shape), dtype=self.state_dtype)

    def reset_tensors(self):
        self._action = self.reset_action()
        self._observ = self.reset_observ()
        self._state = self.reset_state()
        self._is_leaf = get_is_leaf(self._parent)
        super().reset_tensors()

    def to_dict(self):
        observ = self.observ.cpu().numpy()
        return {
            "parent": self.parent.cpu().numpy(),
            "can_clone": self.can_clone.cpu().numpy(),
            "is_cloned": self.is_cloned.cpu().numpy(),
            "is_leaf": self.is_leaf.cpu().numpy(),
            "is_compa_distance": self.is_compa_distance.cpu().numpy(),
            "is_compa_clone": self.is_compa_clone.cpu().numpy(),
            "is_dead": self.is_dead.cpu().numpy(),
            "x": observ[:, 0],
            "y": observ[:, 1],
            "reward": self.reward.cpu().numpy(),
            "oobs": self.oobs.to(torch.float32).cpu().numpy(),
            "virtual_reward": self.virtual_reward.cpu().numpy(),
            "clone_prob": self.clone_prob.cpu().numpy(),
            "clone_ix": self.clone_ix.cpu().numpy(),
            "distance_ix": self.distance_ix.cpu().numpy(),
            "wants_clone": self.wants_clone.cpu().numpy(),
            "will_clone": self.will_clone.cpu().numpy(),
            "distance": self.distance.cpu().numpy(),
            "scaled_distance": self.scaled_distance.cpu().numpy(),
            "scaled_reward": self.scaled_reward.cpu().numpy(),
            # "img": self.img,
        }

    def summary(self):
        return {
            "iteration": self.iteration,
            "total_steps": self.total_steps,
            "best_reward": self.best_reward,
            "leaf_nodes": self.is_leaf.sum().cpu().item(),
            "oobs": self.oobs.sum().cpu().item(),
            "will_clone": self.will_clone.sum().cpu().item(),
            "n_walkers": self.n_walkers,
        }

    def clone_data(self):
        best = self.best_ix
        self.will_clone[best] = False
        try:
            self._observ[: self.n_walkers] = clone_tensor(
                self.observ, self.clone_ix, self.will_clone
            )
            self._reward[: self.n_walkers] = clone_tensor(
                self.reward, self.clone_ix, self.will_clone
            )
            self._cum_reward[: self.n_walkers] = clone_tensor(
                self.cum_reward, self.clone_ix, self.will_clone
            )
            self._state[: self.n_walkers] = clone_tensor(
                self.state, self.clone_ix, self.will_clone
            )
            self._img[: self.n_walkers] = clone_tensor(self.img, self.clone_ix, self.will_clone)
            # This line is what updates the parents to create the tree structure
            # comment this, and you get the swarm wave
            self._parent[: self.n_walkers][self.will_clone] = self.clone_ix[self.will_clone]

            self.policy.clone(self.will_clone, self.clone_ix)
            self.dt_sampler.clone(self.will_clone, self.clone_ix)

        except Exception as e:
            logger.error(
                "Error in clone data %s %s %s",
                self.observ.shape,
                self.will_clone.shape,
                self.clone_ix.shape,
            )
            raise e

    def sample_dt(self, n_walkers: int | None = None):
        if n_walkers is None:
            n_walkers = self.n_walkers
        return self.dt_sampler(n_walkers, self)

    def sample_actions(self, max_walkers: int | None = None):
        if max_walkers is None:
            max_walkers = self.n_walkers
        return self.policy(max_walkers, self)

    def step_env(self):
        self.dt = self.sample_dt(self.action_step.shape[0])
        action = einops.asnumpy(self.action_step)
        data = self.env.step_batch(states=self.state_step, actions=action, dt=self.dt)
        new_states, observ, reward, oobs, truncateds, infos = data
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
        obs_tensor = torch.tensor(observ, dtype=torch.float32, device=self.device)
        oobs_tensor = torch.tensor(oobs, dtype=torch.bool, device=self.device)
        return new_states, obs_tensor, reward_tensor, oobs_tensor, truncateds, infos

    def step_walkers(self):
        wc_np = self.will_clone.cpu().numpy()

        self.state_step = self.state[wc_np]
        if len(self.state_step) == 0:
            return None, None, None, None, None, None
        self.action_step = self.sample_actions(int(wc_np.sum()))
        try:
            logger.info("In step_walkers: %s, %s", self.observ.shape, self.action_step.shape)
            new_states, observ, reward, oobs, truncateds, infos = self.step_env()
        except Exception as e:
            logger.error(
                "Error in step_walkers %s %s %s %s",
                self.state_step.shape,
                self.action_step,
                self.dt.shape,
                wc_np.sum(),
            )
            raise e
        self.observ[self.will_clone] = observ.to(self.observ.dtype)
        self.reward[self.will_clone] = reward.to(self.reward.dtype)
        if self.accumulate_reward:
            self.cum_reward[self.will_clone] += reward.to(self.cum_reward.dtype)
        else:
            self.cum_reward[self.will_clone] = reward.to(self.cum_reward.dtype)
        self.oobs[self.will_clone] = oobs.to(self.oobs.dtype)
        self.state[wc_np] = new_states
        self.img[wc_np] = np.array([info["rgb"] for info in infos])
        self.action[self.will_clone] = self.action_step.to(self.action.dtype)
        return new_states, observ, reward, oobs, truncateds, infos

    def calculate_other_reward(self):
        return 1.0

    def calculate_virtual_reward(self):
        cum_reward = -1.0 * self.cum_reward if self.minimize else self.cum_reward
        compas = random_alive_compas(self.oobs, self.observ)
        flattened_observs = self.observ.reshape(compas.shape[0], -1).to(torch.float32)
        o_rew = self.calculate_other_reward()
        other_reward = o_rew.flatten() if isinstance(o_rew, torch.Tensor) else o_rew
        distance = self.distance_function(flattened_observs, flattened_observs[compas])
        distance_norm = relativize(distance.flatten())
        rew_leafs = cum_reward[self.is_leaf]
        rewards_norm = relativize(cum_reward.flatten(), mean=rew_leafs.mean(), std=rew_leafs.std())
        virtual_reward = (
            distance_norm**self.dist_coef * rewards_norm**self.reward_coef * other_reward
        )
        self.virtual_reward[:] = virtual_reward
        self.distance[:] = distance
        self.scaled_distance[:] = distance_norm
        self.scaled_reward[:] = rewards_norm
        self.distance_ix[:] = compas

    def step_tree(self):
        self.calculate_virtual_reward()
        self.is_leaf[:] = get_is_leaf(self.parent)

        clone_ix, wants_clone, clone_prob = calculate_clone(self.virtual_reward, self.oobs)
        self.clone_ix[:] = clone_ix
        self.wants_clone[:] = wants_clone
        self.clone_prob[:] = clone_prob
        self.is_cloned[:] = get_is_cloned(self.clone_ix, self.wants_clone)
        self.wants_clone[self.oobs] = True
        self.will_clone[:] = self.wants_clone & self.is_leaf & ~self.is_cloned
        if self.will_clone.sum().cpu().item() == 0:
            self.iteration += 1
            return

        self.clone_data()

        self.step_walkers()

        leafs = self.is_leaf.sum().cpu().item()
        new_walkers = self.min_leafs - leafs
        if new_walkers > 0:
            self.add_walkers(new_walkers)

        self.total_steps += self.will_clone.sum().cpu().item()
        self.iteration += 1

    def reset_env(self):
        state, obs, info = self.env.reset()
        return state, obs, info

    def reset(self):
        logger.info("reset start: %s %s", self.observ.shape, self._observ.shape)
        self.n_walkers = self.start_walkers
        self.total_steps = 0
        self.iteration = 0
        self.reset_tensors()
        logger.info("after reset tensor: %s %s", self.observ.shape, self._observ.shape)
        start_action = self.sample_actions(self.start_walkers)
        self._action[: self.n_walkers] = start_action

        state, obs, info = self.reset_env()
        states = np.stack([state.copy() for _ in range(self.start_walkers)], axis=0)
        self._state[: self.n_walkers] = states

        self.state_step = self.state
        self.action_step = self.action

        new_states, observ, reward, oobs, truncateds, infos = self.step_env()
        # here we set the first state to the one after reset. This state will act as the root
        # of the tree
        logger.info("INIT: %s %s", self.observ.shape, observ.shape)
        self._observ[0, :] = observ[0, :]
        # We fill up the remaining states with the ones returned  after stepping the environment.
        # All these states have the root as their parent.
        self._observ[1 : self.n_walkers, :] = observ[1:, :]
        self._reward[1 : self.n_walkers] = reward[1:]
        self._cum_reward[1 : self.n_walkers] = reward[1:]
        self._oobs[1 : self.n_walkers] = oobs[1 : self.n_walkers]
        self._state[1 : self.n_walkers] = new_states[1:]

        self._img = np.zeros_like(self._img)
        try:
            self.env.set_state(state)
            img = self.env.get_image()
            if img is not None:
                self.img[0, :] = img
                self.img[1:, :] = np.array([info["rgb"] for info in infos[1:]])
        except Exception:
            logger.warning("cannot start image")
            # raise e

        return (state, obs, info), new_states, observ, reward, oobs, truncateds, infos


class FaiRunner(param.Parameterized):
    is_running = param.Boolean(default=False)

    def __init__(self, fai, n_steps, plot=None, report_interval=100):
        super().__init__()
        self.reset_btn = pn.widgets.Button(icon="restore", button_type="primary")
        self.play_btn = pn.widgets.Button(icon="player-play", button_type="primary")
        self.pause_btn = pn.widgets.Button(icon="player-pause", button_type="primary")
        self.step_btn = pn.widgets.Button(name="Step", button_type="primary")
        self.progress = pn.indicators.Progress(
            name="Progress", value=0, width=600, max=n_steps, bar_color="primary"
        )
        self.sleep_val = pn.widgets.FloatInput(value=0.0, width=60)
        self.report_interval = pn.widgets.IntInput(value=report_interval, width=70)
        self.table = pn.widgets.Tabulator(show_index=False)
        self.fai = fai
        self.n_steps = n_steps
        self.curr_step = 0
        self.plot = plot
        self.thread = None
        self.erase_coef_val = pn.widgets.FloatInput(value=0.05, width=75)

    @param.depends("erase_coef_val.value")
    def update_erase_coef(self):
        self.fai.erase_coef = self.erase_coef_val.value

    @param.depends("reset_btn.value")
    def on_reset_click(self):
        self.fai.reset()
        self.curr_step = 0
        self.progress.value = 1
        self.curr_step = 0
        self.play_btn.disabled = False
        self.pause_btn.disabled = True
        self.step_btn.disabled = False
        self.is_running = False
        self.progress.bar_color = "primary"
        summary = pd.DataFrame(self.fai.summary(), index=[0])
        self.table.value = summary
        if self.plot is not None:
            self.plot.reset(self.fai)
            self.plot.send(self.fai)

    @param.depends("play_btn.value")
    def on_play_click(self):
        self.play_btn.disabled = True
        self.pause_btn.disabled = False
        self.is_running = True
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run)
            self.thread.start()

    @param.depends("pause_btn.clicks")
    def on_pause_click(self):
        self.is_running = False
        self.play_btn.disabled = False
        self.pause_btn.disabled = True
        if self.thread is not None:
            self.thread.join()

    @param.depends("step_btn.value")
    def on_step_click(self):
        self.take_single_step()

    def take_single_step(self):
        self.fai.step_tree()
        self.curr_step += 1
        self.progress.value = self.curr_step
        if self.curr_step >= self.n_steps:
            self.is_running = False
            self.progress.bar_color = "success"
            self.step_btn.disabled = True
            self.play_btn.disabled = True
            self.pause_btn.disabled = True

        if self.fai.oobs.sum().cpu().item() == self.fai.n_walkers - 1:
            self.is_running = False
            self.progress.bar_color = "danger"

        if self.fai.iteration % self.report_interval.value == 0:
            summary = pd.DataFrame(self.fai.summary(), index=[0])
            self.table.value = summary
            if self.plot is not None:
                self.plot.send(self.fai)

    def run(self):
        while self.is_running:
            self.take_single_step()
            time.sleep(self.sleep_val.value)

    def __panel__(self):
        # pn.state.add_periodic_callback(self.run, period=20)
        buttons_row = pn.Row(
            self.play_btn,
            self.pause_btn,
            self.reset_btn,
            self.step_btn,
            pn.pane.Markdown("**Sleep**"),
            self.sleep_val,
            pn.pane.Markdown("**Report Interval**"),
            self.report_interval,
            pn.pane.Markdown("**Erase coef**"),
            self.erase_coef_val,
        )
        return pn.Column(
            self.fai,
            self.table,
            self.progress,
            buttons_row,
            self.on_play_click,
            self.on_pause_click,
            self.on_reset_click,
            self.on_step_click,
            self.update_erase_coef,
            # self.run,
        )


def sample_actions(x):
    """Sample actions from the environment."""
    return torch.randn_like(x)


def _step(x, actions, benchmark):
    """Step the environment."""
    new_x = x + actions * 0.1
    rewards = benchmark(new_x)
    oobs = benchmark.bounds.contains(new_x)
    return new_x, rewards, oobs


def causal_cone(state, env, policy, n_steps, init_action):
    """Compute the causal cone of a state."""
    env.set_state(state)
    action = init_action
    for i in range(n_steps):
        data = env.step(state=state, action=action)
        state, observ, reward, end, _truncated, _info = data
        compas_ix, will_clone, *_rest_data = fai_iteration(observ, reward, end)
        observ = clone_tensor(observ, compas_ix, will_clone)
        state = clone_tensor(state, compas_ix, will_clone)
        action = policy(observ)


def run_swarm(n_walkers, benchmark, n_steps):
    x = benchmark.sample(n_walkers)
    x[:] = x[0, :]

    actions = sample_actions(x)
    x, rewards, oobs = _step(x, actions, benchmark)

    for i in range(n_steps):
        print(rewards.numpy(force=True).min())
        compas_ix, will_clone, *_rest_data = fai_iteration(x, -rewards, oobs)
        x = clone_tensor(x, compas_ix, will_clone)
        # rewards = clone_tensor(rewards, compas_ix, will_clone)

        actions = sample_actions(x)
        x, rewards, oobs = _step(x, actions, benchmark)


if __name__ == "__main__":
    benchmark = Rastrigin(5)
    run_swarm(500, benchmark=benchmark, n_steps=1000)
