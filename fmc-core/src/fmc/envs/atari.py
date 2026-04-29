"""plangym Atari -> fmc.envs.base.Environment adapter.

Bridges plangym's Atari backend (which exposes get_state/set_state for
true planning) into fmc-core's Environment protocol. Used by:

  - work/09_fmc_vs_mcts_replication/  (P0 — FMC vs MCTS-UCT)
  - work/10_atari_replication/         (P1a — multi-seed Atari)
  - work/11_ram_vs_img_ablation/       (P3 — RAM vs IMG sweep)

Design
------
plangym's `set_state` restores the simulator but does NOT carry the
cumulative episode reward. Since fmc.core.plan calls env.reward(s)
expecting per-walker accumulated reward at the end of the rollout, we
package (snapshot, cum_reward) together as the state token. step()
sets the simulator from the snapshot, applies the action, captures
the per-step reward, and returns a fresh (snapshot', cum_reward')
tuple.

This keeps plangym pure and FMC pure — the adapter owns reward
accumulation, which is the part that doesn't fit either side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AtariState:
    """Bundle of plangym snapshot + cumulative episode reward + done flag."""
    snapshot: np.ndarray   # plangym state token, opaque to us
    obs: np.ndarray        # last observation (RAM bytes or RGB array)
    cum_reward: float
    done: bool


class AtariEnv:
    """Implements fmc.envs.base.Environment over plangym Atari.

    Parameters
    ----------
    name : str
        Plangym environment name, e.g. "ALE/Boxing-v5".
    obs_type : {"ram", "rgb"}
        RAM => 128-byte vector observation. RGB => (210, 160, 3) image.
    frame_skip : int, default 4
        Standard Atari benchmark frame-skip.
    sticky_actions : bool, default False
        Off for deterministic planning (matches paper §5.1.3.3).
    """

    def __init__(
        self,
        name: str = "ALE/Boxing-v5",
        obs_type: str = "ram",
        frame_skip: int = 4,
        sticky_actions: bool = False,
    ):
        # Imported lazily so the rest of fmc-core stays plangym-free.
        import plangym
        self._env = plangym.make(
            name,
            obs_type=obs_type,
            frameskip=frame_skip,
        )
        self._n_actions = int(self._env.action_space.n)
        self._obs_type = obs_type
        self._name = name
        self._frame_skip = frame_skip

    # --- fmc.envs.base.Environment protocol ---

    def reset(self, seed: int | None = None) -> AtariState:
        # plangym.reset(return_state=True) -> (snapshot, obs, info)
        snapshot, obs, _info = self._env.reset(return_state=True)
        return AtariState(
            snapshot=np.asarray(snapshot).copy(),
            obs=np.asarray(obs).copy(),
            cum_reward=0.0,
            done=False,
        )

    def actions(self):
        return tuple(range(self._n_actions))

    def clone_state(self, state: AtariState) -> AtariState:
        return AtariState(
            snapshot=state.snapshot.copy(),
            obs=state.obs.copy(),
            cum_reward=state.cum_reward,
            done=state.done,
        )

    def step(self, state: AtariState, action) -> AtariState:
        if state.done:
            # Absorbing: keep the snapshot, no reward gain.
            return self.clone_state(state)
        self._env.set_state(state.snapshot)
        # plangym.step -> (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, _info = self._env.step(action)
        new_snapshot = self._env.get_state()
        return AtariState(
            snapshot=np.asarray(new_snapshot).copy(),
            obs=np.asarray(obs).copy(),
            cum_reward=state.cum_reward + float(reward),
            done=bool(terminated or truncated),
        )

    def observe(self, state: AtariState) -> np.ndarray:
        # FMC distance metric is L2 in the obs space. RAM is naturally
        # well-scaled (uint8 [0, 255]); RGB is too but the image is huge.
        return state.obs.astype(np.float64).ravel()

    def reward(self, state: AtariState) -> float:
        return state.cum_reward

    def sample_action(self, state: AtariState, rng: np.random.Generator):
        return int(rng.integers(0, self._n_actions))

    # Helpers ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_actions(self) -> int:
        return self._n_actions
