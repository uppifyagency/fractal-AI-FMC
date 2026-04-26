from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fragile.random_state import random_state


def random_choice(x, size=None, replace=True):
    """Randomly sample from a tensor."""
    size = size if size is not None else 1
    if replace:
        size = size if isinstance(size, tuple) else (size,)
        indices = random_state.randint(0, x.shape[0], size=size).to(x.device)
    else:
        indices = random_state.randperm(x.shape[0]).to(x.device)[:size]
    return x[indices]


def l2_norm(x: Tensor, y: Tensor) -> Tensor:
    """Euclidean distance between two batches of points stacked across the first dimension."""
    return torch.norm(x - y, dim=1)


def relativize(
    x: Tensor, std: float | Tensor | None = None, mean: float | Tensor | None = None
) -> Tensor:
    """Normalize the data using a custom smoothing technique."""
    std = x.std() if std is None else std
    if std == 0 or torch.isnan(std) or torch.isinf(std):
        return torch.ones_like(x)
    mean = x.mean() if mean is None else mean
    standard = (x - mean) / std
    return torch.where(standard > 0.0, torch.log(1.0 + standard) + 1.0, torch.exp(standard))


def get_alive_indexes(oobs: Tensor):
    """Get indexes representing random alive walkers given a vector of death conditions."""
    size = oobs.size(0)
    ix_range = torch.arange(size, device=oobs.device)
    if torch.all(oobs):
        return ix_range
    ix = torch.logical_not(oobs).flatten()
    return random_choice(ix_range[ix], size=size, replace=ix.sum() < size).to(oobs.device)


def random_alive_compas(
    oobs: Tensor | None = None,
    ref_tensor: Tensor | None = None,
):
    """Get random alive compas."""
    if oobs is None and ref_tensor is None:
        msg = "Must provide either oobs or ref_tensor"
        raise ValueError(msg)
    if oobs is None and ref_tensor is not None:
        compas = torch.arange(ref_tensor.shape[0], device=ref_tensor.device)
    else:
        compas = get_alive_indexes(oobs)
    return compas[torch.randperm(compas.size(0), device=compas.device)]


def calculate_distance(
    observs: Tensor,
    distance_function: Callable = l2_norm,
    return_compas: bool = False,
    oobs: Tensor = None,
    compas: Tensor = None,
):
    """Calculate a distance metric for each walker with respect to a random companion."""
    if compas is None:
        compas = random_alive_compas(oobs, observs)
    flattened_observs = observs.view(observs.shape[0], -1)
    distance = distance_function(flattened_observs, flattened_observs[compas])
    distance_norm = relativize(distance.flatten())
    return distance_norm if not return_compas else (distance_norm, compas)


def cross_distance(
    observ: Tensor,
    dst_observ: Tensor,
    oobs: Tensor = None,
    distance_function: Callable = l2_norm,
    return_compas: bool = False,
    return_distance: bool = False,
    compas: Tensor = None,
):
    """Calculate a distance metric for each walker with respect to a random companion."""
    if compas is None:
        compas = random_alive_compas(oobs, dst_observ)[: observ.shape[0]]
    flattened_observs = observ.view(observ.shape[0], -1)
    flattened_dst_observs = dst_observ.view(dst_observ.shape[0], -1)
    distance = distance_function(flattened_observs, flattened_dst_observs[compas])
    distance_norm = relativize(distance.flatten())
    data = (distance_norm,)
    if return_compas:
        data = (*data, compas)
    if return_distance:
        data = (*data, distance)
    return data[0] if len(data) == 1 else tuple(data)


def calculate_virtual_reward(
    observs: Tensor,
    rewards: Tensor,
    oobs: Tensor = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    other_reward: Tensor | float = 1.0,
    return_compas: bool = False,
    return_distance: bool = False,
    distance_function: Callable[[Tensor, Tensor], Tensor] = l2_norm,
) -> Tensor | (tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]):
    """Calculate the virtual rewards given the required data."""
    compas = random_alive_compas(oobs, observs)
    flattened_observs = observs.reshape(len(compas), -1).to(torch.float32)
    other_reward = other_reward.flatten() if isinstance(other_reward, Tensor) else other_reward
    distance = distance_function(flattened_observs, flattened_observs[compas])
    distance_norm = relativize(distance.flatten())
    rewards_norm = relativize(rewards.flatten())
    virtual_reward = distance_norm**dist_coef * rewards_norm**reward_coef * other_reward
    return_data = (virtual_reward,)
    if return_compas:
        return_data = (*return_data, compas)
    if return_distance:
        return_data = (*return_data, distance)
    return return_data[0] if len(return_data) == 1 else tuple(return_data)


def cross_virtual_reward(
    observs: Tensor,
    rewards: Tensor,
    dst_observs: Tensor,
    oobs: Tensor = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    other_reward: Tensor = 1.0,
    return_compas: bool = False,
    return_distance: bool = False,
    distance_function: Callable = l2_norm,
) -> Tensor | (tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]):
    """Calculate the virtual rewards given the required data."""
    distance_norm, compas, distance = cross_distance(
        observs,
        dst_observs,
        oobs,
        distance_function,
        return_compas=True,
        return_distance=True,
    )
    rewards_norm = relativize(rewards.flatten())
    virtual_reward = distance_norm**dist_coef * rewards_norm**reward_coef * other_reward
    return_data = (virtual_reward,)
    if return_compas:
        return_data = (*return_data, compas)
    if return_distance:
        return_data = (*return_data, distance)
    return return_data[0] if len(return_data) == 1 else tuple(return_data)


def calculate_clone(
    virtual_rewards: Tensor, oobs: Tensor | None = None, eps=1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Calculate the clone indexes and masks from the virtual rewards."""
    compas_ix = random_alive_compas(oobs, virtual_rewards)
    vir_rew = virtual_rewards.flatten()
    clone_probs = (vir_rew[compas_ix] - vir_rew) / torch.where(
        vir_rew > eps,
        vir_rew,
        torch.tensor(eps, device=vir_rew.device),
    )
    will_clone = clone_probs.flatten() > torch.rand(len(clone_probs), device=clone_probs.device)
    return compas_ix, will_clone, clone_probs


def cross_clone(
    virtual_rewards: Tensor, dst_virtual_reward: Tensor, oobs: Tensor | None = None, eps=1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Calculate the clone indexes and masks from the virtual rewards."""
    compas_ix = random_alive_compas(oobs, dst_virtual_reward)[: virtual_rewards.shape[0]]
    vir_rew = virtual_rewards.flatten()
    dst_vir_rew = dst_virtual_reward.flatten()
    clone_probs = (dst_vir_rew[compas_ix] - vir_rew) / torch.where(
        vir_rew > eps,
        vir_rew,
        torch.tensor(eps, device=vir_rew.device),
    )
    will_clone = clone_probs.flatten() > torch.randperm(
        len(clone_probs), device=clone_probs.device
    )
    return compas_ix, will_clone, clone_probs


def fai_iteration(
    observs: Tensor,
    rewards: Tensor,
    oobs: Tensor | None = None,
    dist_coef: float = 1.0,
    reward_coef: float = 1.0,
    eps=1e-8,
    other_reward: Tensor = 1.0,
    return_clone_probs: bool = False,
    return_compas_dist: bool = False,
    return_distance: bool = False,
    distance_function: Callable = l2_norm,
) -> tuple[Tensor, ...]:
    """Perform a FAI iteration."""
    oobs = (
        oobs
        if oobs is not None
        else torch.zeros(rewards.shape, dtype=torch.bool, device=rewards.device)
    )
    virtual_reward = calculate_virtual_reward(
        observs,
        rewards,
        oobs,
        dist_coef=dist_coef,
        reward_coef=reward_coef,
        other_reward=other_reward,
        return_distance=return_distance,
        return_compas=return_compas_dist,
        distance_function=distance_function,
    )
    rest_data: Any = ()
    if isinstance(virtual_reward, tuple):
        virtual_reward, *rest_data = virtual_reward
    compas_ix, will_clone, clone_probs = calculate_clone(
        virtual_rewards=virtual_reward, oobs=oobs, eps=eps
    )
    if return_clone_probs:
        rest_data = (*tuple(rest_data), clone_probs)
    return compas_ix, will_clone, *rest_data


def clone_tensor(
    x: Tensor | np.ndarray, compas_ix: Tensor, will_clone: Tensor
) -> Tensor | np.ndarray:
    """Clone the data from the compas indexes."""
    if not will_clone.any():
        return x
    if isinstance(x, torch.Tensor):
        x[will_clone] = x[compas_ix][will_clone]
    elif isinstance(x, np.ndarray):
        compas_ix, will_clone = compas_ix.cpu().numpy(), will_clone.cpu().numpy()
        x[will_clone] = x[compas_ix][will_clone]
    else:
        raise ValueError(f"Unsupported type {type(x)}")
    return x
