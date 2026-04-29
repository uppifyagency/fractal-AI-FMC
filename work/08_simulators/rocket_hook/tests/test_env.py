"""Pytest suite for RocketHookEnv.

Validates the PlanEnv contract (`get_state`/`set_state` round-trip,
`apply_action` returns correct tuple, `step_batch` vectorizes), plus
physics invariants (gravity pulls down, reward is positive, episode
terminates on crash, success flag works) and the rendering path.
"""

from __future__ import annotations

import numpy as np
import pytest

from env import (  # type: ignore[import-not-found]
    ACTION_TABLE,
    ARENA_X,
    ARENA_Y,
    DT_PHYS,
    G,
    HOOK_GRAB_RADIUS,
    STATE_DIM,
    TARGET_RADIUS,
    RocketHookEnv,
    _pack,
    _unpack,
)


# ----------------------------------------------------------------- fixtures
@pytest.fixture
def env() -> RocketHookEnv:
    return RocketHookEnv(seed=42)


# ----------------------------------------------------------------- contract
class TestPlanEnvContract:
    """Tests that the env honors the plangym PlanEnv contract."""

    def test_setup_initializes_spaces(self, env: RocketHookEnv) -> None:
        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.observation_space.shape == (STATE_DIM,)
        assert env.action_space.n == len(ACTION_TABLE)

    def test_reset_returns_obs_and_info(self, env: RocketHookEnv) -> None:
        state, obs, info = env.reset(return_state=True)
        assert isinstance(state, np.ndarray)
        assert obs.shape == (STATE_DIM,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_get_set_state_roundtrip(self, env: RocketHookEnv) -> None:
        env.reset()
        state_a = env.get_state()
        # Step a few times — env should now diverge from state_a.
        for _ in range(10):
            env.step(env.sample_action())
        state_b = env.get_state()
        assert not np.allclose(state_a, state_b)
        # Restore and verify
        env.set_state(state_a)
        state_a2 = env.get_state()
        assert np.allclose(state_a, state_a2)

    def test_set_state_makes_step_deterministic(self, env: RocketHookEnv) -> None:
        env.reset()
        env_b = RocketHookEnv(seed=99)
        env_b.reset()
        anchor = env.get_state()
        env_b.set_state(anchor)
        # Same action from same state must produce same next state.
        out_a = env.step(action=3, state=anchor, return_state=True)
        out_b = env_b.step(action=3, state=anchor, return_state=True)
        np.testing.assert_allclose(out_a[0], out_b[0], atol=1e-12)
        # Reward and obs equal too.
        np.testing.assert_allclose(out_a[1], out_b[1], atol=1e-12)
        assert out_a[2] == out_b[2]

    def test_step_batch_signature(self, env: RocketHookEnv) -> None:
        env.reset()
        anchor = env.get_state()
        actions = [env.sample_action() for _ in range(8)]
        states = [anchor.copy() for _ in actions]
        out = env.step_batch(actions=actions, states=states, dt=1, return_state=True)
        # PlanEnv.step_batch returns tuple-of-lists: (states, obs, rewards, terminals, truncs, infos)
        assert len(out) == 6
        new_states, observs, rewards, terminals, truncs, infos = out
        assert len(new_states) == len(actions)
        assert len(observs) == len(actions)
        assert observs[0].shape == (STATE_DIM,)
        assert all(isinstance(r, float) for r in rewards)


# ----------------------------------------------------------------- physics
class TestPhysics:
    def test_gravity_pulls_stone_down_in_phase_0(self, env: RocketHookEnv) -> None:
        env.reset()
        d = _unpack(env.get_state())
        # Force a clean phase-0 state: stone in mid-air with zero velocity.
        d["sx"], d["sy"], d["svx"], d["svy"] = 5.0, 14.0, 0.0, 0.0
        d["hx"], d["hy"] = -5.0, 5.0  # hook far from stone so no auto-grab
        d["stone_phase"] = 0.0
        env.set_state(np.concatenate([_pack(d), [0.0]]))
        for _ in range(5):
            env.step(0)  # noop
        post = _unpack(env.get_state())
        assert post["sy"] < 14.0, "Stone should fall under gravity"
        assert post["svy"] < 0.0, "Stone vertical velocity should be negative"

    def test_thrust_lifts_rocket(self, env: RocketHookEnv) -> None:
        env.reset()
        d = _unpack(env.get_state())
        d["x"], d["y"] = 0.0, 5.0
        d["vx"], d["vy"], d["theta"], d["omega"] = 0.0, 0.0, 0.0, 0.0
        # Push stone+hook far so hook physics doesn't dominate
        d["sx"], d["sy"] = 0.0, 50.0
        d["hx"], d["hy"] = 0.0, 5.0 - 1.5
        d["vhx"], d["vhy"] = 0.0, 0.0
        env.set_state(np.concatenate([_pack(d), [0.0]]))
        # Many thrust ticks — should produce non-trivial upward acceleration.
        for _ in range(20):
            env.step(3)  # pure thrust
        post = _unpack(env.get_state())
        assert post["vy"] > 0.0, f"Thrust must add upward velocity, got vy={post['vy']:.3f}"

    def test_auto_grab_when_hook_near_stone(self, env: RocketHookEnv) -> None:
        env.reset()
        d = _unpack(env.get_state())
        # Pin rocket directly above hook at rope rest length so spring is neutral.
        d["x"], d["y"] = 5.0, 11.5
        d["vx"], d["vy"] = 0.0, 0.0
        d["theta"], d["omega"] = 0.0, 0.0
        d["hx"], d["hy"] = 5.0, 10.0
        d["vhx"], d["vhy"] = 0.0, 0.0
        # Stone right next to hook (well within HOOK_GRAB_RADIUS=0.5)
        d["sx"], d["sy"] = 5.0 + 0.2 * HOOK_GRAB_RADIUS, 10.0
        d["svx"], d["svy"] = 0.0, 0.0
        d["stone_phase"] = 0.0
        env.set_state(np.concatenate([_pack(d), [0.0]]))
        env.step(0)
        post = _unpack(env.get_state())
        assert post["stone_phase"] >= 1.0, "Stone should auto-grab when hook is within radius"

    def test_auto_deposit_when_stone_at_target(self) -> None:
        # Use autoreset=False so we can inspect state after terminal.
        env = RocketHookEnv(seed=42, autoreset=False)
        env.reset()
        target = env.target_xy
        d = _unpack(env.get_state())
        # Pin rocket above target at rope rest length — hook stays at target,
        # stone follows hook, deposit triggers.
        d["x"], d["y"] = float(target[0]), float(target[1]) + 1.5
        d["vx"], d["vy"] = 0.0, 0.0
        d["theta"], d["omega"] = 0.0, 0.0
        d["hx"], d["hy"] = float(target[0]), float(target[1])
        d["vhx"], d["vhy"] = 0.0, 0.0
        d["sx"], d["sy"] = float(target[0]), float(target[1])
        d["stone_phase"] = 1.0
        env.set_state(np.concatenate([_pack(d), [0.0]]))
        obs, reward, term, trunc, info = env.step(0)
        assert env.get_state()[14] >= 2.0, f"Stone should be delivered, got phase={env.get_state()[14]}"
        assert term, "Episode should terminate on delivery"
        # Reward at the deposit step is computed from phase-1 branch
        # (the deposit happens during this very step), so we only verify
        # the next step would yield 5.0 — but here we just check terminal
        # and post-delivery state.
        assert reward > 1.0

    def test_crash_terminates_episode(self, env: RocketHookEnv) -> None:
        env.reset()
        d = _unpack(env.get_state())
        d["x"], d["y"] = 0.0, 0.05
        d["vy"] = -1.0
        env.set_state(np.concatenate([_pack(d), [0.0]]))
        obs, reward, term, trunc, info = env.step(0)
        assert term, "Rocket below ground must terminate"

    def test_truncation_at_max_steps(self) -> None:
        env = RocketHookEnv(max_steps=5, seed=7)
        env.reset()
        # Make sure we don't terminate naturally: pin rocket high in arena
        d = _unpack(env.get_state())
        d["x"], d["y"] = 0.0, 10.0
        d["vx"], d["vy"] = 0.0, 0.0
        d["sx"], d["sy"] = -10.0, 16.0  # stone far from hook
        env.set_state(np.concatenate([_pack(d), [0.0]]))
        last = None
        for _ in range(6):
            last = env.step(0)
            if last[2] or last[3]:
                break
        assert last is not None
        assert last[3] is True or last[2] is True

    def test_reward_is_positive_in_phase_0(self, env: RocketHookEnv) -> None:
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert reward > 0.0


# ----------------------------------------------------------------- rendering
class TestRendering:
    def test_image_shape_and_dtype(self, env: RocketHookEnv) -> None:
        env.reset()
        img = env.get_image()
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8

    def test_img_shape_property(self, env: RocketHookEnv) -> None:
        # cached_property triggers the rasterizer once.
        env.reset()
        assert env.img_shape == (64, 64, 3)


# ----------------------------------------------------------------- packing
class TestPacking:
    def test_pack_unpack_roundtrip(self) -> None:
        d = {
            "x": 1.0, "y": 2.0, "vx": 0.1, "vy": -0.2, "theta": 0.3, "omega": 0.05,
            "hx": 0.5, "hy": 0.5, "vhx": 0.0, "vhy": -0.4,
            "sx": 4.0, "sy": 12.0, "svx": 0.0, "svy": -3.0,
            "stone_phase": 1.0,
        }
        packed = _pack(d)
        assert packed.shape == (STATE_DIM,)
        d2 = _unpack(packed)
        for k in d:
            assert d[k] == pytest.approx(d2[k])
