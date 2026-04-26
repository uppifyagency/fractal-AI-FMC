import logging

import numpy
import numpy as np
import torch

from fragile.core import BaseDtSampler, BasePolicy, FractalTree
from fragile.utils import numpy_dtype_to_torch_dtype


logger = logging.getLogger(__name__)


class FunctionTree(FractalTree):
    def __init__(
        self,
        max_walkers,
        env,
        policy: BasePolicy | None = None,
        dt_sampler: BaseDtSampler | None = None,
        minimize: bool = True,
        device="cuda",
        start_walkers=100,
        min_leafs=100,
        obs_shape: tuple[int, ...] | None = None,
        obs_dtype: torch.dtype | None = torch.float32,
        action_shape: tuple[int, ...] | None = None,
        action_dtype: torch.dtype | None = torch.float32,
        state_shape: tuple[int, ...] | None = (),
        state_dtype: torch.dtype | type = np.float32,
        img_shape: tuple[int, ...] | None = (10, 10, 3),
        img_dtype: torch.dtype | numpy.dtype | None = np.uint8,
    ):
        obs_shape = obs_shape or env.bounds.shape
        action_shape = action_shape or obs_shape
        super().__init__(
            max_walkers=max_walkers,
            env=env,
            policy=policy,
            dt_sampler=dt_sampler,
            minimize=minimize,
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
            img_dtype=img_dtype,
        )

    def reset_img(self):
        return np.zeros((self.max_walkers, *self.img_shape), dtype=self.img_dtype)

    def reset_observ(self):
        self.obs_shape = self.obs_shape if self.obs_shape is not None else self.env.bounds.shape
        self.obs_dtype = (
            self.obs_dtype
            if self.obs_dtype is not None
            else numpy_dtype_to_torch_dtype(self.env.bounds.dtype)
        )
        return torch.zeros(
            (self.max_walkers, *self.obs_shape), device=self.device, dtype=self.obs_dtype
        )

    def reset_action(self):
        self.action_shape = (
            self.action_shape if self.action_shape is not None else self.env.bounds.shape
        )
        self.action_type = (
            self.action_type
            if self.action_type is not None
            else numpy_dtype_to_torch_dtype(self.env.bounds.dtype)
        )
        return torch.zeros(
            (self.max_walkers, *self.action_shape), device=self.device, dtype=self.action_type
        )

    def step_env(self):
        self.dt = self.sample_dt(self.action_step.shape[0])
        n_wakers = self.will_clone.sum().item()
        if n_wakers != self.action_step.shape[0]:
            obs = self.observ
            n_wakers = self.action_step.shape[0]
        else:
            logger.info(
                "In step env: %s %s %s %s",
                n_wakers,
                self.action_step.shape[0],
                self.observ.shape[0],
                self.will_clone.shape[0],
            )
            obs = self.observ[self.will_clone]

        new_obs = self.action_step + obs
        reward_tensor = self.env(new_obs)

        new_states = new_obs.numpy(force=True)
        oobs_tensor = (
            torch.logical_not(self.env.bounds.contains(new_obs)).flatten().to(new_obs.device)
        )
        truncateds = oobs_tensor.detach().clone()
        infos = [{"rgb": np.zeros(self.img_shape)} for _ in range(n_wakers)]
        return new_states, new_obs, reward_tensor, oobs_tensor, truncateds, infos

    def reset_env(self):
        obs = self.env.bounds.sample(1)[0]
        self.observ[:] = obs
        return obs.numpy(force=True), obs, {"rgb": np.zeros(self.img_shape)}
