"""SuperMimic's per-environment material and per-episode calibration draws.

These functions deliberately avoid importing Isaac Lab so the actual event
implementations can be checked with small articulation/action-manager fixtures.
"""

from __future__ import annotations

import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.utils.wam_context import invalidate_physical_context


def _environment_ids(env, env_ids, device):
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=device)
    if isinstance(env_ids, slice):
        return torch.arange(env.scene.num_envs, device=device)[env_ids]
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _context(env, asset):
    shape = (int(env.scene.num_envs), 33)
    xi = getattr(env, "_wam_xi", None)
    target = asset.data.joint_pos
    if xi is None or tuple(xi.shape) != shape:
        xi = target.new_zeros(shape)
        xi[:, 0] = 1.0
        env._wam_xi = xi
    return xi


def randomize_supermimic_material(env, env_ids, asset_cfg, friction_range=(0.3, 1.2)):
    """One continuous coefficient per environment, shared by all robot shapes.

    PhysX has separate static/dynamic coefficients; both receive the same MuJoCo
    coefficient. The matched ground uses max-combination and coefficient 0.3.
    Restitution is fixed at zero rather than sampled.
    """
    asset = env.scene[asset_cfg.name]
    view = getattr(asset, "root_physx_view", None)
    if view is None:
        view = asset.root_view
    materials = view.get_material_properties().clone()
    ids = _environment_ids(env, env_ids, materials.device)
    low, high = map(float, friction_range)
    if not 0 <= low <= high:
        raise ValueError("SuperMimic friction requires 0 <= low <= high")
    mu = materials.new_empty(len(ids)).uniform_(low, high)
    materials[ids, :, 0] = mu[:, None]
    materials[ids, :, 1] = mu[:, None]
    materials[ids, :, 2] = 0.0
    view.set_material_properties(materials, ids.to(device="cpu", dtype=torch.int32))
    xi = _context(env, asset)
    xi[ids.to(xi.device), 0] = mu.to(xi)
    invalidate_physical_context(asset)


def reset_supermimic_joint_offsets(env, env_ids, asset_cfg, offset_range=(-0.01, 0.01)):
    """Resample the 29 body-joint offsets from immutable nominal defaults.

    Extra dexterous-hand joints are not part of SuperMimic's calibration vector.
    Action offsets are updated by joint name, independently of either ordering.
    """
    asset = env.scene[asset_cfg.name]
    defaults = asset.data.default_joint_pos
    nominal = getattr(asset.data, "default_joint_pos_nominal", None)
    if nominal is None:
        nominal = defaults[0].detach().clone()
        asset.data.default_joint_pos_nominal = nominal
    if nominal.ndim == 1:
        nominal = nominal.unsqueeze(0)
    if nominal.shape[-1] != defaults.shape[-1] or nominal.shape[0] not in (1, defaults.shape[0]):
        raise ValueError("SuperMimic nominal joint defaults do not match the robot")
    ids = _environment_ids(env, env_ids, defaults.device)
    names = tuple(joint_utils.get_robot_joint_names(asset, order="mujoco"))
    if len(names) != 29:
        raise ValueError("SuperMimic calibration requires exactly 29 body joints")
    body_ids = torch.as_tensor([asset.joint_names.index(name) for name in names], device=defaults.device)
    low, high = map(float, offset_range)
    if low > high:
        raise ValueError("SuperMimic joint offset range must be ordered")
    offsets = defaults.new_empty(len(ids), 29).uniform_(low, high)
    nominal_rows = nominal.expand_as(defaults)[ids]
    defaults[ids[:, None], body_ids[None, :]] = nominal_rows[:, body_ids] + offsets

    action = env.action_manager.get_term("joint_pos")
    selected = [(i, asset.joint_names.index(name)) for i, name in enumerate(action._joint_names) if name in names]
    if selected:
        action_ids = torch.as_tensor([pair[0] for pair in selected], device=defaults.device)
        asset_ids = torch.as_tensor([pair[1] for pair in selected], device=defaults.device)
        action._offset[ids[:, None], action_ids[None, :]] = defaults[ids[:, None], asset_ids[None, :]]
    xi = _context(env, asset)
    xi[ids.to(xi.device), 1:30] = offsets.to(xi)
