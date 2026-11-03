"""Physical context snapshots for the deterministic WAM dynamics model.

The first 33 coordinates retain Sonic's calibration interface. Complete mode
also records every PhysX collision material and every body's mass, inertia and
local CoM pose. Ordered metadata is kept separately so equal tensor dimensions
cannot silently identify different robots. This module does not import IsaacLab.
"""

from __future__ import annotations

from copy import deepcopy

import torch

from gear_sonic.envs.env_utils.joint_utils import get_robot_joint_names, get_robot_torso_name


CONTEXT_PREFIX_DIM = 33
CONTEXT_SCHEMA_VERSION = 1


def invalidate_physical_context(asset):
    """Invalidate cached static properties after a simulator randomization event."""
    asset._wam_physical_context = None


def _view(asset):
    view = getattr(asset, "root_physx_view", None)
    if view is None:
        view = getattr(asset, "root_view", None)
    if view is None:
        raise ValueError("Complete WAM context requires the robot's PhysX articulation view")
    return view


def _shape_counts(asset, view, shape_count):
    counts = getattr(asset, "_wam_shapes_per_body", None)
    if counts is None:
        try:
            # IsaacLab uses the same workaround to map flattened articulation
            # materials to individual bodies in randomize_rigid_body_material.
            paths = view.link_paths[0]
            counts = tuple(
                int(asset._physics_sim_view.create_rigid_body_view(path).max_shapes)
                for path in paths
            )
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("Complete WAM context cannot resolve collision-shape body ordering") from exc
        asset._wam_shapes_per_body = counts
    counts = tuple(int(value) for value in counts)
    if len(counts) != len(asset.body_names) or min(counts, default=0) < 0 or sum(counts) != shape_count:
        raise ValueError("Complete WAM context collision-shape layout does not match the articulation")
    return counts


def _property(view, name, shape):
    try:
        value = getattr(view, name)()
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Complete WAM context requires PhysX {name}()") from exc
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"Complete WAM context {name} must have shape {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"Complete WAM context {name} contains non-finite properties")
    return value


def _snapshot(asset):
    cached = getattr(asset, "_wam_physical_context", None)
    if cached is not None:
        return cached
    view = _view(asset)
    n = int(asset.data.joint_pos.shape[0])
    bodies = tuple(asset.body_names)
    joints = tuple(asset.joint_names)
    if not bodies or len(set(bodies)) != len(bodies) or len(set(joints)) != len(joints):
        raise ValueError("Complete WAM context requires unique ordered body and joint names")
    try:
        shapes = int(view.max_shapes)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Complete WAM context requires the PhysX collision-shape count") from exc
    if shapes < 1:
        raise ValueError("Complete WAM context requires at least one robot collision shape")
    counts = _shape_counts(asset, view, shapes)
    materials = _property(view, "get_material_properties", (n, shapes, 3))
    masses = _property(view, "get_masses", (n, len(bodies)))
    inertias = _property(view, "get_inertias", (n, len(bodies), 9))
    coms = _property(view, "get_coms", (n, len(bodies), 7))
    # PhysX stores local CoM quaternions as xyzw; use the repository's wxyz
    # convention in context, and persist that convention in its schema.
    coms = torch.cat((coms[..., :3], coms[..., 6:7], coms[..., 3:6]), -1)
    fields = []
    cursor = CONTEXT_PREFIX_DIM
    for name, shape in (("shape_materials", (shapes, 3)), ("body_mass", (len(bodies),)),
                        ("body_inertia", (len(bodies), 9)), ("body_com_pose", (len(bodies), 7)),
                        ("joint_default_offsets", (len(joints),))):
        size = 1
        for width in shape:
            size *= width
        fields.append({"name": name, "start": cursor, "end": cursor + size, "shape": list(shape)})
        cursor += size
    schema = {
        "version": CONTEXT_SCHEMA_VERSION,
        "dim": cursor,
        "prefix_dim": CONTEXT_PREFIX_DIM,
        "prefix_joint_names": list(get_robot_joint_names(asset, order="mujoco")),
        "body_names": list(bodies),
        "joint_names": list(joints),
        "shapes_per_body": list(counts),
        "shape_order": [[body, index] for body, count in zip(bodies, counts) for index in range(count)],
        "material_order": ["static_friction", "dynamic_friction", "restitution"],
        "com_quaternion_order": "wxyz",
        "inertia_order": "row_major_3x3",
        "fields": fields,
    }
    target = asset.data.joint_pos
    extra = torch.cat((materials.flatten(1), masses, inertias.flatten(1), coms.flatten(1)), -1)
    cached = {
        "values": extra.to(device=target.device, dtype=target.dtype).detach().clone(),
        "friction": materials[..., 0].mean(-1).to(device=target.device, dtype=target.dtype).detach().clone(),
        "coms": coms.to(device=target.device, dtype=target.dtype).detach().clone(),
        "schema": schema,
    }
    asset._wam_physical_context = cached
    return cached


def physical_context_schema(asset):
    """Return a serializable description of the complete context feature order."""
    return deepcopy(_snapshot(asset)["schema"])


def physical_context(asset, legacy_context=None):
    """Return complete static physics/calibration context, ``[N, 33+3S+17B+J]``.

    The legacy prefix is [mean static friction, 29 nominal joint offsets,
    torso CoM offset]. Detailed appended properties are exact snapshots, never
    material averages. Joint offsets are read afresh because reset events can
    update calibration without changing the cached rigid-body properties.
    """
    cached = _snapshot(asset)
    values = cached["values"]
    n = values.shape[0]
    if legacy_context is None:
        prefix = values.new_zeros(n, CONTEXT_PREFIX_DIM)
    else:
        if tuple(legacy_context.shape) != (n, CONTEXT_PREFIX_DIM):
            raise ValueError("Complete WAM context legacy calibration prefix must have shape [N,33]")
        prefix = legacy_context.to(device=values.device, dtype=values.dtype).clone()
    prefix[:, 0] = cached["friction"]
    try:
        indices = torch.as_tensor(
            [asset.joint_names.index(name) for name in get_robot_joint_names(asset, order="mujoco")], device=values.device,
        )
    except ValueError as exc:
        raise ValueError("Complete physical WAM context requires all 29 named robot joints") from exc
    current = asset.data.default_joint_pos
    nominal = getattr(asset.data, "default_joint_pos_nominal", None)
    if nominal is None:
        if not torch.equal(current, current[:1].expand_as(current)):
            raise ValueError("WAM calibration offsets require nominal joint positions before randomization")
        nominal = asset.data.default_joint_pos[0]
    if nominal.ndim == 1:
        nominal = nominal.unsqueeze(0)
    if nominal.shape[0] not in (1, n) or nominal.shape[-1] != len(asset.joint_names):
        raise ValueError("WAM nominal joint positions do not match the articulation layout")
    offsets = current - nominal
    prefix[:, 1:30] = offsets[:, indices]
    nominal_coms = getattr(asset, "_wam_nominal_coms", None)
    torso_name = get_robot_torso_name(asset)
    if nominal_coms is not None and torso_name in asset.body_names:
        torso = asset.body_names.index(torso_name)
        prefix[:, 30:33] = cached["coms"][:, torso, :3] - nominal_coms[:, torso, :3].to(values)
    return torch.cat((prefix, values, offsets), -1)


def get_push_versions(env, num_envs=None):
    """Snapshot per-environment impulse counters without exposing a mutable view."""
    n = int(env.scene.num_envs if num_envs is None else num_envs)
    versions = getattr(env, "_wam_push_version", None)
    if versions is None:
        versions = torch.zeros(n, dtype=torch.long, device=getattr(env, "device", "cpu"))
        env._wam_push_version = versions
    if tuple(versions.shape) != (n,) or versions.dtype != torch.long:
        raise ValueError("WAM push counters must be one integer per environment")
    return versions.clone()


def record_push(env, env_ids):
    """Record an already applied external impulse for dynamics-supervision masks."""
    get_push_versions(env)
    ids = slice(None) if env_ids is None else env_ids
    env._wam_push_version[ids] += 1
