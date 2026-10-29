"""Differentiable G1 link poses and velocities from the repository's MJCF.

Only the body tree, hinge joints and limits are read from XML. Training does
not need MuJoCo, meshes, or a running simulator to evaluate this kinematics.
Root linear velocity is interpreted at the pelvis link origin, not its CoM.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch import nn

from gear_sonic.trl.modules.wam import physical_state_parts


G1_MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint",
    "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint",
    "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

TRACKED_BODIES = (
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)


def _vector(element, name, default, length):
    values = tuple(float(value) for value in element.get(name, default).split())
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid MJCF {element.tag} {name}: {values}")
    return values


def _quat_matrix(quaternion):
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(quaternion.shape[:-1] + (3, 3))


class G1Kinematics(nn.Module):
    """Compute tracked link origins, rotations, and world-frame velocities.

    The input's last dimension is the canonical 71-D physical WAM state:
    ``p, quat(wxyz), v, w, q(29), qd(29)``. All leading dimensions are retained.
    Geometry and per-joint limits are buffers and follow ``.to(device, dtype)``.
    """

    def __init__(self, mjcf_path=None, tracked_bodies=None):
        super().__init__()
        repository = Path(__file__).resolve().parents[2]
        path = Path(mjcf_path) if mjcf_path is not None else Path(
            "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
        )
        if not path.is_absolute():
            path = repository / path
        self.mjcf_path = str(path.resolve())
        model = ET.parse(path).getroot()
        if model.find("include") is not None:
            raise ValueError("WAM kinematics requires a self-contained G1 MJCF")
        compiler = model.find("compiler")
        if compiler is not None and compiler.get("angle", "degree") != "radian":
            raise ValueError("WAM G1 MJCF must use radian joint limits")
        roots = [body for world in model.findall("worldbody") for body in world.findall("body")]
        roots = [body for body in roots if body.find("freejoint") is not None or any(
            joint.get("type", "hinge") == "free" for joint in body.findall("joint")
        )]
        if len(roots) != 1 or roots[0].get("name") != "pelvis":
            raise ValueError("WAM kinematics requires one floating pelvis root")

        self.joint_names = G1_MUJOCO_JOINT_NAMES
        joint_index = {name: index for index, name in enumerate(self.joint_names)}
        body_names, parents, body_joint_indices = [], [], []
        offsets, rotations, axes, pivots = [], [], [], []
        limits = {}

        def visit(body, parent):
            name = body.get("name")
            if name is None or name in body_names:
                raise ValueError("WAM MJCF bodies need unique names")
            if any(body.get(key) is not None for key in ("euler", "axisangle", "xyaxes", "zaxis")):
                raise ValueError(f"WAM MJCF body {name} must express orientation as a quaternion")
            joints = body.findall("joint") + body.findall("freejoint")
            if len(joints) > 1:
                raise ValueError(f"WAM MJCF body {name} has multiple joints")
            current = len(body_names)
            body_names.append(name)
            parents.append(parent)
            offsets.append(_vector(body, "pos", "0 0 0", 3))
            rotations.append(_vector(body, "quat", "1 0 0 0", 4))
            index, axis, pivot = -1, (0., 0., 1.), (0., 0., 0.)
            if joints:
                joint = joints[0]
                joint_type = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
                if parent == -1:
                    if joint_type != "free":
                        raise ValueError("The pelvis root must have a free joint")
                else:
                    if joint_type != "hinge" or joint.get("name") not in joint_index:
                        raise ValueError(f"Unsupported WAM joint on {name}: {joint.attrib}")
                    joint_name = joint.get("name")
                    if joint_name in limits:
                        raise ValueError(f"Duplicate WAM joint {joint_name}")
                    index = joint_index[joint_name]
                    axis = _vector(joint, "axis", "0 0 1", 3)
                    pivot = _vector(joint, "pos", "0 0 0", 3)
                    norm = math.sqrt(sum(value * value for value in axis))
                    if norm < 1e-12:
                        raise ValueError(f"Zero joint axis on {name}")
                    axis = tuple(value / norm for value in axis)
                    if "range" not in joint.attrib:
                        raise ValueError(f"Missing joint range on {name}")
                    lower, upper = _vector(joint, "range", "", 2)
                    if lower >= upper:
                        raise ValueError(f"Invalid joint range on {name}")
                    limits[joint_name] = (lower, upper)
            body_joint_indices.append(index)
            axes.append(axis)
            pivots.append(pivot)
            for child in body.findall("body"):
                visit(child, current)

        visit(roots[0], -1)
        if set(limits) != set(self.joint_names):
            raise ValueError(f"WAM MJCF is missing joints: {set(self.joint_names) - set(limits)}")
        self.body_names = tuple(body_names)
        self.parent = tuple(parents)
        self.joint_indices = tuple(body_joint_indices)
        self.tracked_bodies = tuple(TRACKED_BODIES if tracked_bodies is None else tracked_bodies)
        if not self.tracked_bodies or len(set(self.tracked_bodies)) != len(self.tracked_bodies):
            raise ValueError("WAM tracked bodies must be nonempty and unique")
        missing = set(self.tracked_bodies) - set(self.body_names)
        if missing:
            raise ValueError(f"Unknown WAM tracked bodies: {missing}")
        self.tracked_indices = tuple(self.body_names.index(name) for name in self.tracked_bodies)
        self.register_buffer("off_p", torch.tensor(offsets, dtype=torch.float32))
        self.register_buffer("off_R", _quat_matrix(torch.tensor(rotations, dtype=torch.float32)))
        self.register_buffer("axes", torch.tensor(axes, dtype=torch.float32))
        self.register_buffer("pivots", torch.tensor(pivots, dtype=torch.float32))
        self.register_buffer("q_lo", torch.tensor([limits[name][0] for name in self.joint_names]))
        self.register_buffer("q_hi", torch.tensor([limits[name][1] for name in self.joint_names]))

    @staticmethod
    def _joint_rotation(axis, angle):
        x, y, z = axis.unbind(-1)
        zero = x.new_zeros(())
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
        outer = axis[:, None] * axis[None, :]
        eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
        cosine, sine = angle.cos()[..., None, None], angle.sin()[..., None, None]
        return cosine * eye + (1 - cosine) * outer + sine * skew

    def forward(self, state):
        p, R, v, w, q, qd = physical_state_parts(state)
        if self.off_p.device != state.device or self.off_p.dtype != state.dtype:
            raise ValueError("Move G1Kinematics to the state device and dtype before evaluation")
        positions, rotations, velocities, angular_velocities = [], [], [], []
        for index, parent in enumerate(self.parent):
            if parent < 0:
                positions.append(p)
                rotations.append(R)
                velocities.append(v)
                angular_velocities.append(w)
                continue
            Rp = rotations[parent]
            pre_rotation = Rp @ self.off_R[index]
            initial_origin = positions[parent] + (Rp @ self.off_p[index, :, None]).squeeze(-1)
            joint = self.joint_indices[index]
            if joint >= 0:
                body_rotation = pre_rotation @ self._joint_rotation(self.axes[index], q[..., joint])
                pivot_before = (pre_rotation @ self.pivots[index, :, None]).squeeze(-1)
                pivot_after = (body_rotation @ self.pivots[index, :, None]).squeeze(-1)
                anchor = initial_origin + pivot_before
                body_position = anchor - pivot_after
                axis_world = (pre_rotation @ self.axes[index, :, None]).squeeze(-1)
                body_angular = angular_velocities[parent] + axis_world * qd[..., joint:joint + 1]
                anchor_velocity = velocities[parent] + torch.cross(
                    angular_velocities[parent], anchor - positions[parent], dim=-1,
                )
                body_velocity = anchor_velocity - torch.cross(body_angular, pivot_after, dim=-1)
            else:
                body_rotation, body_position = pre_rotation, initial_origin
                body_angular = angular_velocities[parent]
                body_velocity = velocities[parent] + torch.cross(
                    body_angular, body_position - positions[parent], dim=-1,
                )
            positions.append(body_position)
            rotations.append(body_rotation)
            velocities.append(body_velocity)
            angular_velocities.append(body_angular)
        return (
            torch.stack([positions[index] for index in self.tracked_indices], dim=-2),
            torch.stack([rotations[index] for index in self.tracked_indices], dim=-3),
            torch.stack([velocities[index] for index in self.tracked_indices], dim=-2),
            torch.stack([angular_velocities[index] for index in self.tracked_indices], dim=-2),
        )
