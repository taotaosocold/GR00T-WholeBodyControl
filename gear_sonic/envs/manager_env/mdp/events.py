"""Event functions for domain randomization and environment resets in RL training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import randomize_rigid_body_material as _randomize_rigid_body_material
from isaaclab.envs.mdp import randomize_rigid_body_mass as _randomize_rigid_body_mass
from isaaclab.envs.mdp import push_by_setting_velocity as _push_by_setting_velocity
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
import torch

from gear_sonic.utils.wam_context import invalidate_physical_context, record_push

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = None
    add_joint_default_pos = None
    add_hand_joint_default_pos = None
    base_com = None

    # interval - balance training
    push_robot = None

    randomize_rigid_body_mass = None


def _wam_xi_buffer(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the per-environment SuperMimic physical-context buffer.

    The buffer deliberately lives on the environment rather than in the WAM
    trainer.  Domain-randomization events are the source of truth for these
    values, so replay and imagined rollouts see the same ``xi`` that was used
    by the simulator.
    """
    xi = getattr(env, "_wam_xi", None)
    num_envs = int(env.scene.num_envs)
    device = env.device
    if xi is None or tuple(xi.shape) != (num_envs, 33):
        xi = torch.zeros(num_envs, 33, device=device)
        xi[:, 0] = 1.0
        env._wam_xi = xi
    return xi


def _record_material_context(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Copy the effective static friction into the per-environment ξ buffer."""
    xi = _wam_xi_buffer(env)
    asset = env.scene[asset_cfg.name]
    invalidate_physical_context(asset)
    try:
        view = getattr(asset, "root_physx_view", None)
        if view is None:
            view = getattr(asset, "root_view", None)
        properties = view.get_material_properties()
        friction = properties[..., 0].reshape(properties.shape[0], -1).mean(-1)
        if env_ids is None:
            ids = torch.arange(env.scene.num_envs, device=xi.device)
        elif isinstance(env_ids, slice):
            ids = torch.arange(env.scene.num_envs, device=xi.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=xi.device)
        xi[ids, 0] = friction[ids].to(device=xi.device, dtype=xi.dtype)
    except (AttributeError, RuntimeError, TypeError, IndexError):
        # The exact material getter is version-dependent. The simulator has
        # still been randomized; leave ξ at its nominal value when unavailable.
        pass


if isinstance(_randomize_rigid_body_material, type):

    class randomize_rigid_body_material(_randomize_rigid_body_material):
        """Isaac Lab class term that also records its friction in ξ."""

        def __call__(
            self,
            env: ManagerBasedEnv,
            env_ids: torch.Tensor | slice | None,
            static_friction_range: tuple[float, float],
            dynamic_friction_range: tuple[float, float],
            restitution_range: tuple[float, float],
            num_buckets: int,
            asset_cfg: SceneEntityCfg,
            make_consistent: bool = False,
        ) -> None:
            super().__call__(
                env,
                env_ids,
                static_friction_range,
                dynamic_friction_range,
                restitution_range,
                num_buckets,
                asset_cfg,
                make_consistent,
            )
            _record_material_context(env, env_ids, asset_cfg)

else:

    def randomize_rigid_body_material(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
    ) -> None:
        """Compatibility wrapper for Isaac Lab releases using a function term."""
        _randomize_rigid_body_material(
            env,
            env_ids,
            static_friction_range=static_friction_range,
            dynamic_friction_range=dynamic_friction_range,
            restitution_range=restitution_range,
            num_buckets=num_buckets,
            asset_cfg=asset_cfg,
        )
        _record_material_context(env, env_ids, asset_cfg)


if isinstance(_randomize_rigid_body_mass, type):

    class randomize_rigid_body_mass(_randomize_rigid_body_mass):
        """Preserve IsaacLab mass/inertia randomization and refresh WAM context."""

        def __call__(
            self, env, env_ids, asset_cfg, mass_distribution_params, operation,
            distribution="uniform", recompute_inertia=True,
        ):
            super().__call__(
                env, env_ids, asset_cfg, mass_distribution_params, operation,
                distribution=distribution, recompute_inertia=recompute_inertia,
            )
            invalidate_physical_context(env.scene[asset_cfg.name])

else:

    def randomize_rigid_body_mass(
        env, env_ids, asset_cfg, mass_distribution_params, operation,
        distribution="uniform", recompute_inertia=True,
    ):
        """Compatibility wrapper for IsaacLab releases with a function term."""
        _randomize_rigid_body_mass(
            env, env_ids, asset_cfg, mass_distribution_params, operation,
            distribution=distribution, recompute_inertia=recompute_inertia,
        )
        invalidate_physical_context(env.scene[asset_cfg.name])


def push_by_setting_velocity(
    env, env_ids, velocity_range, asset_cfg=SceneEntityCfg("robot"),
):
    """Apply the original impulse and mark its transition as externally disturbed."""
    _push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg=asset_cfg)
    record_push(env, env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize joint default positions to simulate calibration errors.

    Applies random offsets to the default joint positions of the robot, modeling
    real-world joint encoder calibration inaccuracies. Also updates the action
    manager offset to keep action space aligned with the new defaults.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If None, randomizes all.
        asset_cfg: Scene entity config with joint IDs to randomize.
        pos_distribution_params: Min/max range for the position offset distribution.
        operation: How to combine the random value with the original ("add", "scale", "abs").
        distribution: Sampling distribution type.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])
    _wam_xi_buffer(env)

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos,
            pos_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically

        action_joint_names = env.action_manager.get_term("joint_pos")._joint_names
        asset_joint_names = asset.joint_names
        shared_joint_names = list(set(action_joint_names).intersection(set(asset_joint_names)))
        shared_joint_indices_action = [
            action_joint_names.index(name) for name in shared_joint_names
        ]
        shared_joint_indices_asset = [asset_joint_names.index(name) for name in shared_joint_names]

        shared_offset = asset.data.default_joint_pos[env_ids, shared_joint_indices_asset]
        env.action_manager.get_term("joint_pos")._offset[
            env_ids, shared_joint_indices_action
        ] = shared_offset


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu"
    ).unsqueeze(1)

    xi = _wam_xi_buffer(env)
    xi_ids = torch.as_tensor(env_ids, device=xi.device)
    xi[xi_ids, 30:33] = rand_samples[:, 0].to(device=xi.device, dtype=xi.dtype)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()
    if getattr(asset, "_wam_nominal_coms", None) is None:
        asset._wam_nominal_coms = coms.clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
    invalidate_physical_context(asset)
