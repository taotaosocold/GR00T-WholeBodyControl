"""
IsaacLab <-> MuJoCo ordering conversion utilities.

This module provides utilities for converting both DOF and body ordering
between IsaacLab and MuJoCo conventions for humanoid robots.

qpos format: [root_trans(3), root_quat(4), dof_angles(N)]
"""

from abc import ABC

import numpy as np
import torch


class IsaacLabMuJoCoConverter(ABC):
    """Abstract base class for DOF/body order conversion between IsaacLab and MuJoCo.

    Subclasses must define DOF_MAPPINGS for their specific robot.
    """

    ROOT_QPOS_OFFSET = 7  # root_trans(3) + root_quat(4)
    DOF_MAPPINGS: dict = {}  # Must be overridden by subclasses
    VALID_DOF_ORDERS = ("mujoco", "isaaclab")

    def convert(self, data: torch.Tensor, from_order: str, to_order: str) -> torch.Tensor:
        """Convert DOF order between conventions.

        Auto-detects input format:
            - qpos:           [..., 7 + num_dof]           -> reorder DOF portion, keep root
            - dof angles:     [..., num_dof]                -> reorder directly
            - body transforms [..., num_dof + 1, D]         -> keep root body, reorder DOF bodies
            - body transforms [..., num_dof, D]             -> reorder directly
            - rotation mats   [..., num_dof (+ 1), 3, 3]   -> same as above on dim -3

        Full body transforms use BODY_MAPPINGS when available, including fixed
        bodies such as N3's toes. Actuated-link subsets retain the historical
        DOF-based conversion, with an optional leading root body.
        """
        if from_order == to_order:
            return data

        mapping = self.DOF_MAPPINGS[(from_order, to_order)]
        last_dim = data.shape[-1]

        if last_dim == self.ROOT_QPOS_OFFSET + self.num_dof:
            # qpos: [..., 7 + num_dof]
            return torch.cat(
                [
                    data[..., : self.ROOT_QPOS_OFFSET],
                    data[..., self.ROOT_QPOS_OFFSET :][..., mapping],
                ],
                dim=-1,
            )
        elif last_dim == self.num_dof:
            # Raw DOF angles: [..., num_dof]
            return data[..., mapping]
        else:
            # Per-body transforms: body dim is -3 for [..., J, 3, 3], else -2 for [..., J, D]
            body_dim = (
                -3 if (data.ndim >= 3 and data.shape[-1] == 3 and data.shape[-2] == 3) else -2
            )
            num_bodies = data.shape[body_dim]

            full_body_mapping = getattr(self, "BODY_MAPPINGS", {}).get((from_order, to_order))
            if full_body_mapping is not None and num_bodies == len(full_body_mapping):
                body_order = full_body_mapping
            elif num_bodies == self.num_dof + 1:
                # Includes root at index 0 — keep root, reorder DOF bodies
                body_order = [0] + [m + 1 for m in mapping]
            elif num_bodies == self.num_dof:
                body_order = mapping
            else:
                raise ValueError(
                    f"Cannot detect format: last_dim={last_dim}, body_dim size={num_bodies}, "
                    f"expected num_dof+1={self.num_dof + 1} or num_dof={self.num_dof}"
                )

            if body_dim == -2:
                return data[..., body_order, :]
            else:  # body_dim == -3
                return data[..., body_order, :, :]

    def to_mujoco(self, data: torch.Tensor) -> torch.Tensor:
        """Convert from IsaacLab to MuJoCo DOF order."""
        return self.convert(data, from_order="isaaclab", to_order="mujoco")

    def to_isaaclab(self, data: torch.Tensor) -> torch.Tensor:
        """Convert from MuJoCo to IsaacLab DOF order."""
        return self.convert(data, from_order="mujoco", to_order="isaaclab")

    @property
    def num_dof(self) -> int:
        """Number of actuated DOFs (excluding root)."""
        return len(self.DOF_MAPPINGS[(self.VALID_DOF_ORDERS[0], self.VALID_DOF_ORDERS[1])])

    @property
    def isaaclab_to_mujoco_dof(self):
        return self.DOF_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_dof(self):
        return self.DOF_MAPPINGS[("mujoco", "isaaclab")]

    @property
    def isaaclab_to_mujoco_body(self):
        return self.BODY_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_body(self):
        return self.BODY_MAPPINGS[("mujoco", "isaaclab")]


class G1Converter(IsaacLabMuJoCoConverter):
    """G1 ordering converter.

    Imports G1 body/DOF/joint mappings from gear_sonic.envs.manager_env.robots.g1.
    """

    def __init__(self):
        # Lazy import to avoid circular dependency:
        # order_converter -> g1 -> mdp/__init__ -> commands -> order_converter
        from gear_sonic.envs.manager_env.robots.g1 import (
            G1_ISAACLAB_JOINTS,
            G1_ISAACLAB_TO_MUJOCO_BODY,
            G1_ISAACLAB_TO_MUJOCO_DOF,
            G1_MUJOCO_TO_ISAACLAB_BODY,
            G1_MUJOCO_TO_ISAACLAB_DOF,
        )

        self.JOINT_NAMES = G1_ISAACLAB_JOINTS
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): G1_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): G1_MUJOCO_TO_ISAACLAB_DOF,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): G1_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): G1_MUJOCO_TO_ISAACLAB_BODY,
        }

    # Body subset names for MPJPE metrics (used by reconstruction_trainer)
    VR_3POINTS_BODY_NAMES = ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

    @property
    def vr_3points_mujoco_indices(self):
        """VR 3-point body indices in full (30-body) MuJoCo body order.

        These index into the full body array after isaaclab_to_mujoco_body
        reordering, NOT the 14-body motion.yaml body_names subset.
        """
        mj_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mj_names.index(n) for n in self.VR_3POINTS_BODY_NAMES]

    @property
    def foot_mujoco_indices(self):
        """Foot body indices in full (30-body) MuJoCo body order.

        These index into the full body array after isaaclab_to_mujoco_body
        reordering, NOT the 14-body motion.yaml body_names subset.
        """
        mj_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mj_names.index(n) for n in self.FOOT_BODY_NAMES]

    @property
    def isaaclab_to_mujoco_dof(self):
        """DOF reorder indices: IsaacLab -> MuJoCo."""
        return self.DOF_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_dof(self):
        """DOF reorder indices: MuJoCo -> IsaacLab."""
        return self.DOF_MAPPINGS[("mujoco", "isaaclab")]

    @property
    def isaaclab_to_mujoco_body(self):
        """Body reorder indices: IsaacLab -> MuJoCo."""
        return self.BODY_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_body(self):
        """Body reorder indices: MuJoCo -> IsaacLab."""
        return self.BODY_MAPPINGS[("mujoco", "isaaclab")]

    def get_isaaclab_to_mujoco_mapping(self):
        """Return the full mapping dict for body/DOF reordering."""
        return {
            "isaaclab_joints": self.JOINT_NAMES,
            "isaaclab_to_mujoco_dof": self.isaaclab_to_mujoco_dof,
            "mujoco_to_isaaclab_dof": self.mujoco_to_isaaclab_dof,
            "isaaclab_to_mujoco_body": self.isaaclab_to_mujoco_body,
            "mujoco_to_isaaclab_body": self.mujoco_to_isaaclab_body,
        }


class H2Converter(IsaacLabMuJoCoConverter):
    """H2 robot joint/body order converter between IsaacLab and MuJoCo conventions."""

    def __init__(self):
        from gear_sonic.envs.manager_env.robots.h2 import (
            H2_ISAACLAB_JOINTS,
            H2_ISAACLAB_TO_MUJOCO_BODY,
            H2_ISAACLAB_TO_MUJOCO_DOF,
            H2_MUJOCO_TO_ISAACLAB_BODY,
            H2_MUJOCO_TO_ISAACLAB_DOF,
        )

        self.JOINT_NAMES = H2_ISAACLAB_JOINTS
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): H2_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): H2_MUJOCO_TO_ISAACLAB_DOF,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): H2_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): H2_MUJOCO_TO_ISAACLAB_BODY,
        }

    VR_3POINTS_BODY_NAMES = ["torso_link", "left_wrist_pitch_link", "right_wrist_pitch_link"]
    FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


class N3Converter(IsaacLabMuJoCoConverter):
    """N3 ordering with independent joint/body maps, including fixed toes.

    Runtime articulation names override the expected IsaacLab traversal order.
    Validation is by name and rejects missing/extra joints or bodies before
    motion data can be silently assigned to the wrong coordinate.
    """

    VR_3POINTS_BODY_NAMES = ["l_arm_hand_pitch_link", "r_arm_hand_pitch_link", "head_yaw_link"]
    FOOT_BODY_NAMES = ["l_leg_ankle_roll_link", "r_leg_ankle_roll_link"]

    def __init__(self, *, joint_names=None, body_names=None):
        from gear_sonic.envs.env_utils.n3_utils import (
            N3_ISAACLAB_BODY_NAMES, N3_ISAACLAB_DOF_NAMES,
            N3_LOWER_BODY_JOINT_NAMES, N3_MUJOCO_BODY_NAMES,
            N3_MUJOCO_JOINT_NAMES, N3_WRIST_JOINT_NAMES,
        )

        self.isaaclab_joint_names = tuple(N3_ISAACLAB_DOF_NAMES if joint_names is None else joint_names)
        self.isaaclab_body_names = tuple(N3_ISAACLAB_BODY_NAMES if body_names is None else body_names)
        self.mujoco_joint_names = tuple(N3_MUJOCO_JOINT_NAMES)
        self.mujoco_body_names = tuple(N3_MUJOCO_BODY_NAMES)
        for label, supplied, expected in (
            ("joints", self.isaaclab_joint_names, self.mujoco_joint_names),
            ("bodies", self.isaaclab_body_names, self.mujoco_body_names),
        ):
            if len(set(supplied)) != len(supplied) or set(supplied) != set(expected):
                raise ValueError(f"N3 {label} mismatch: missing={sorted(set(expected) - set(supplied))}, "
                                 f"extra={sorted(set(supplied) - set(expected))}")
        self.JOINT_NAMES = self.isaaclab_body_names  # legacy key denotes body names
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): [self.isaaclab_joint_names.index(name) for name in self.mujoco_joint_names],
            ("mujoco", "isaaclab"): [self.mujoco_joint_names.index(name) for name in self.isaaclab_joint_names],
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): [self.isaaclab_body_names.index(name) for name in self.mujoco_body_names],
            ("mujoco", "isaaclab"): [self.mujoco_body_names.index(name) for name in self.isaaclab_body_names],
        }
        self.motion_metadata = {
            "robot_type": "n3",
            "body_joint_names_mujoco": list(self.mujoco_joint_names),
            "body_names_mujoco": list(self.mujoco_body_names),
            "lower_joint_indices_mujoco": [self.mujoco_joint_names.index(name) for name in N3_LOWER_BODY_JOINT_NAMES],
            "wrist_mujoco_dof_indices": [self.mujoco_joint_names.index(name) for name in N3_WRIST_JOINT_NAMES],
            "left_foot_body_idx": [self.mujoco_body_names.index(self.FOOT_BODY_NAMES[0])],
            "right_foot_body_idx": [self.mujoco_body_names.index(self.FOOT_BODY_NAMES[1])],
        }

    def get_isaaclab_to_mujoco_mapping(self):
        return {
            "isaaclab_joints": list(self.isaaclab_body_names),
            "isaaclab_dof_names": list(self.isaaclab_joint_names),
            "isaaclab_to_mujoco_dof": self.DOF_MAPPINGS[("isaaclab", "mujoco")],
            "mujoco_to_isaaclab_dof": self.DOF_MAPPINGS[("mujoco", "isaaclab")],
            "isaaclab_to_mujoco_body": self.BODY_MAPPINGS[("isaaclab", "mujoco")],
            "mujoco_to_isaaclab_body": self.BODY_MAPPINGS[("mujoco", "isaaclab")],
        }


def load_qpos_from_csv(csv_path: str) -> torch.Tensor:
    """Load qpos [T, D] from CSV."""
    import pandas as pd

    return torch.from_numpy(pd.read_csv(csv_path).values.astype(np.float32))


def save_qpos_to_csv(qpos: torch.Tensor, csv_path: str):
    """Save qpos to CSV."""
    import pandas as pd

    data = qpos[0].cpu().numpy() if qpos.dim() == 3 else qpos.cpu().numpy()
    pd.DataFrame(data).to_csv(csv_path, index=False)


if __name__ == "__main__":
    from isaacsim import SimulationApp

    _sim_app = SimulationApp({"headless": True})

    # Test DOF conversion round-trip
    converter = G1Converter()

    # Create random qpos (G1 has 29 DOFs)
    T, num_dof = 30, 29
    qpos = torch.cat(
        [
            torch.tensor([[0.0, 0.0, 1.0]]).expand(T, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(T, 4),
            torch.randn(T, num_dof) * 0.3,
        ],
        dim=-1,
    )

    # Test round-trip conversion
    qpos_mujoco = converter.to_mujoco(qpos)
    qpos_back = converter.to_isaaclab(qpos_mujoco)
    print(f"Round-trip error: {(qpos - qpos_back).abs().max():.2e}")

    # Test explicit convert API
    qpos_mujoco2 = converter.convert(qpos, from_order="isaaclab", to_order="mujoco")
    print(f"to_mujoco vs convert match: {torch.allclose(qpos_mujoco, qpos_mujoco2)}")

    _sim_app.close()
