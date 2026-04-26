from typing import Callable

import numpy as np
import torch

from fragile.core import FractalTree
from fragile.fractalai import clone_tensor, l2_norm, relativize


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


class VideoGameTree(FractalTree):
    def __init__(
        self,
        max_walkers,
        env,
        policy=None,
        device="cuda",
        start_walkers=100,
        min_leafs=100,
        obs_shape: tuple[int, ...] | None = None,
        obs_dtype: torch.dtype | None = None,
        action_shape: tuple[int, ...] | None = None,
        action_dtype: torch.dtype | None = None,
        state_shape: tuple[int, ...] | None = (),
        state_dtype: torch.dtype | type = object,
        rgb_shape: tuple[int, ...] = (210, 160, 3),
        agg_block_size: int = 5,
    ):
        super().__init__(
            max_walkers=max_walkers,
            env=env,
            policy=policy,
            device=device,
            start_walkers=start_walkers,
            min_leafs=min_leafs,
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            state_shape=state_shape,
            state_dtype=state_dtype,
        )
        self.agg_block_size = agg_block_size
        self.rgb_shape = rgb_shape
        self.obs_shape = self.env.observation_space.shape
        self.rgb = np.zeros((self.start_walkers, *self.rgb_shape), dtype=np.uint8)

    def add_walkers(self, new_walkers):
        super().add_walkers(new_walkers)
        rgb = np.zeros((new_walkers, *self.rgb_shape), dtype=np.uint8)
        self.rgb = np.concatenate((self.rgb, rgb), axis=0)

    def sample_dt(self, n_walkers: int | None = None):
        if n_walkers is None:
            n_walkers = self.n_walkers
        return np.random.randint(1, 5, size=len(n_walkers))  # noqa: NPY002

    def to_dict(self):
        data = super().to_dict()
        data["rgb"] = self.rgb
        return data

    def clone_data(self):
        super().clone_data()
        try:
            self.rgb = clone_tensor(self.rgb, self.clone_ix, self.will_clone)
        except Exception as e:
            print("CACA", self.observ.shape, self.will_clone.shape, self.clone_ix.shape)
            raise e

    def step_walkers(self):
        wc_np = self.will_clone.cpu().numpy()
        new_states, observ, reward, oobs, truncateds, infos = super().step_walkers()
        if new_states is None:
            return new_states, observ, reward, oobs, truncateds, infos
        self.rgb[wc_np] = np.array([info["rgb"] for info in infos])
        return new_states, observ, reward, oobs, truncateds, infos

    def reset(self):
        (state, obs, info), new_states, observ, reward, oobs, truncateds, infos = super().reset()
        self.rgb = np.zeros((self.start_walkers, *self.rgb_shape), dtype=np.uint8)
        self.env.set_state(state)
        self.rgb[0, :] = self.env.get_image()
        self.rgb[1:, :] = np.array([info["rgb"] for info in infos[1:]])
        return (state, obs, info), new_states, observ, reward, oobs, truncateds, infos


class MontezumaTree(FractalTree):
    def __init__(
        self,
        max_walkers,
        env,
        policy=None,
        dt_sampler=None,
        device="cuda",
        start_walkers=100,
        min_leafs=100,
        obs_shape: tuple[int, ...] | None = None,
        obs_dtype: torch.dtype | None = None,
        action_shape: tuple[int, ...] | None = None,
        action_dtype: torch.dtype | None = None,
        state_shape: tuple[int, ...] | None = (),
        state_dtype: torch.dtype | type = object,
        img_shape: tuple[int, ...] = (210, 160, 3),
        count_visits: bool = True,
        erase_coef=0.05,
        agg_block_size: int = 5,
        distance_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = l2_norm,
        dist_coef: float = 1.0,
        reward_coef: float = 1.0,
    ):
        super().__init__(
            max_walkers=max_walkers,
            env=env,
            policy=policy,
            dt_sampler=dt_sampler,
            device=device,
            start_walkers=start_walkers,
            min_leafs=min_leafs,
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            state_shape=state_shape,
            state_dtype=state_dtype,
            img_shape=img_shape,
            distance_function=distance_function,
            dist_coef=dist_coef,
            reward_coef=reward_coef,
        )
        self.agg_block_size = agg_block_size
        self.count_visits = count_visits
        self.erase_coef = erase_coef
        self.visits = np.zeros((24, 160, 160), dtype=np.int64)

    def to_dict(self):
        data = super().to_dict()
        data["visits"] = self.visits
        return data

    def step_walkers(self):
        new_states, observ, reward, oobs, truncateds, infos = super().step_walkers()
        if new_states is None:
            return new_states, observ, reward, oobs, truncateds, infos
        if self.count_visits:
            self.update_visits(np.array(observ, dtype=np.int64))
        return new_states, observ, reward, oobs, truncateds, infos

    def reset(self):
        state, new_states, observ, reward, oobs, truncateds, infos = super().reset()
        self.visits = np.zeros((24, 160, 160), dtype=np.float32)
        if self.count_visits:
            self.update_visits(np.array(observ, dtype=np.int64))
        return state, new_states, observ, reward, oobs, truncateds, infos

    def update_visits(self, observ):
        observ = observ.astype(np.float64)
        observ[:, 0] /= int(self.env.gym_env._x_repeat)
        observ = observ.astype(np.int64)

        self.visits[observ[:, 2], observ[:, 1], observ[:, 0]] = np.where(
            np.isnan(self.visits[observ[:, 2], observ[:, 1], observ[:, 0]]),
            1,
            self.visits[observ[:, 2], observ[:, 1], observ[:, 0]] + 1,
        )
        self.visits = np.clip(self.visits - self.erase_coef, 0, 1000)

    def calculate_other_reward(self):
        visits = aggregate_visits(self.visits, block_size=self.agg_block_size, upsample=True)
        obs = self.observ.numpy(force=True).astype(np.int64)
        x, y, room_ix = obs[:, 0], obs[:, 1], obs[:, 2]
        visits_val = -torch.tensor(visits[room_ix, y, x], device=self.device, dtype=torch.float32)
        leaf_visits = visits_val[self.is_leaf]
        return relativize(visits_val, mean=leaf_visits.mean(), std=leaf_visits.std())
