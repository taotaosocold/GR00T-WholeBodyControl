"""SMPL five-point geometry without Isaac, robot FK or SMPL model assets."""

import copy
import math
import unittest

import torch

from gear_sonic.isaac_utils import rotations
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    quaternion_to_rotation_matrix,
)
from gear_sonic.utils.smpl_vr import (
    SMPL_VR5_JOINT_INDICES,
    smpl_vr5_local_orientations,
    smpl_vr5_local_positions,
    smpl_vr5_targets,
    validate_smpl_vr5_data,
)


def _rotation(axis_angle):
    """Independent Rodrigues reference through a matrix exponential."""
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    return torch.matrix_exp(skew.reshape(*axis_angle.shape[:-1], 3, 3))


def _root(pose, y_up=False):
    root = angle_axis_to_quaternion(pose[:, 0])
    if y_up:
        root = rotations.smpl_root_ytoz_up(root)
    return rotations.remove_smpl_base_rot(root, w_last=False)


def _payload(frames=3):
    joints = torch.zeros(frames, 24, 3, dtype=torch.float64)
    joints[:, 20] = torch.tensor((0.5, 0.3, 0.6))
    joints[:, 21] = torch.tensor((0.5, -0.3, 0.6))
    joints[:, 15, 2] = 0.9
    joints[:, 7] = torch.tensor((0., 0.1, -0.7))
    joints[:, 8] = torch.tensor((0., -0.1, -0.7))
    return dict(fps=50., pose_aa=torch.zeros(frames, 72, dtype=torch.float64), smpl_joints=joints)


class SmplVR5Tests(unittest.TestCase):
    def test_positions_select_named_points_and_remove_pelvis_translation(self):
        data = _payload(2)
        original = data["smpl_joints"].clone()
        root = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).repeat(2, 1)
        actual = smpl_vr5_local_positions(original, root)
        expected = original[:, [20, 21, 15, 7, 8]].reshape(2, 15)
        torch.testing.assert_close(actual, expected)
        translated = original + torch.tensor([[[12., -7., 4.]]], dtype=torch.float64)
        torch.testing.assert_close(smpl_vr5_local_positions(translated, root), expected)
        torch.testing.assert_close(data["smpl_joints"], original)

    def test_fk_accumulates_spine_shoulder_elbow_wrist_and_leg_chains(self):
        pose = torch.zeros(2, 24, 3, dtype=torch.float64)
        pose[:, 3, 0] = 0.2
        pose[:, 9, 2] = -0.3
        pose[:, 13, 1] = 0.4
        pose[:, 16, 0] = 0.5
        pose[:, 18, 2] = -0.6
        pose[:, 20, 1] = 0.7
        pose[:, 1, 2] = 0.3
        pose[:, 4, 0] = -0.4
        pose[:, 7, 1] = 0.5
        pose[:, 0, 2] = torch.tensor((0.1, -0.8))
        raw_rot = _rotation(pose)
        # Independent, explicit paths prevent accidentally testing the same tree.
        paths = ([3, 6, 9, 13, 16, 18, 20], [3, 6, 9, 14, 17, 19, 21],
                 [3, 6, 9, 12, 15], [1, 4, 7], [2, 5, 8])
        base = torch.tensor(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)), dtype=torch.float64)
        expected = []
        for path in paths:
            rotation = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
            for index in path:
                rotation = rotation @ raw_rot[:, index]
            expected.append(base @ rotation @ base.T)
        actual = smpl_vr5_local_orientations(pose.reshape(2, 72), _root(pose))
        actual_rot = quaternion_to_rotation_matrix(actual.reshape(2, 5, 4))
        torch.testing.assert_close(actual_rot, torch.stack(expected, dim=1), atol=2e-7, rtol=2e-7)
        # The selected wrist contains every ancestor, not just its own local AA.
        self.assertGreater(torch.linalg.vector_norm(actual_rot[:, 0] - raw_rot[:, 20]).item(), 0.1)

    def test_y_up_pose_and_z_up_pose_produce_identical_targets(self):
        pose_y = torch.zeros(2, 24, 3, dtype=torch.float64)
        pose_y[:, 18, 2] = 0.4
        pose_z = pose_y.clone()
        pose_z[:, 0, 0] = math.pi / 2
        joints_z = _payload(2)["smpl_joints"]
        y_outputs = smpl_vr5_targets(joints_z, pose_y, _root(pose_y, True), smpl_y_up=True)
        z_outputs = smpl_vr5_targets(joints_z, pose_z, _root(pose_z), smpl_y_up=False)
        for actual, expected in zip(y_outputs, z_outputs):
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)

    def test_joint_targets_are_invariant_to_common_world_yaw(self):
        data = _payload(1)
        pose = data["pose_aa"].reshape(1, 24, 3)
        pose[:, 18, 1] = 0.4
        positions, orientations = smpl_vr5_targets(data["smpl_joints"], pose, _root(pose))
        turned_pose = pose.clone()
        turned_pose[:, 0, 2] = 1.2
        yaw_matrix = _rotation(turned_pose[:, 0])
        turned_joints = (yaw_matrix[:, None] @ data["smpl_joints"][..., None]).squeeze(-1)
        turned = smpl_vr5_targets(turned_joints, turned_pose, _root(turned_pose))
        torch.testing.assert_close(turned[0], positions, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(turned[1], orientations, atol=2e-7, rtol=2e-7)

    def test_terminal_hand_pose_cannot_change_wrist_head_or_ankle_targets(self):
        pose = torch.zeros(1, 24, 3, dtype=torch.float64)
        initial = smpl_vr5_local_orientations(pose, _root(pose))
        pose[:, 22:, :] = 1.7
        torch.testing.assert_close(smpl_vr5_local_orientations(pose, _root(pose)), initial)
        torch.testing.assert_close(initial.reshape(1, 5, 4)[..., 0], torch.ones(1, 5, dtype=torch.float64))

    def test_source_validation_is_read_only_and_allows_one_zero_frame(self):
        data = _payload()
        data["smpl_joints"][0] = 0
        original = copy.deepcopy(data)
        self.assertIsNone(validate_smpl_vr5_data(data))
        for field in ("pose_aa", "smpl_joints"):
            torch.testing.assert_close(data[field], original[field], rtol=0, atol=0)
        data["pose_aa"] = data["pose_aa"].reshape(-1, 24, 3)
        self.assertIsNone(validate_smpl_vr5_data(data))

    def test_source_validation_rejects_missing_mismatched_nonfinite_and_degenerate_data(self):
        invalid = []
        for field in ("fps", "pose_aa", "smpl_joints"):
            data = _payload()
            del data[field]
            invalid.append(data)
        for field in ("pose_aa", "smpl_joints"):
            data = _payload()
            data[field] = data[field][:-1]
            invalid.append(data)
            for value in (float("nan"), float("inf")):
                data = _payload()
                data[field].view(-1)[0] = value
                invalid.append(data)
        for fps in (0, -1, float("inf"), float("nan"), True, None):
            data = _payload()
            data["fps"] = fps
            invalid.append(data)
        data = _payload()
        data["pose_aa"] = data["pose_aa"][:, :69]
        invalid.append(data)
        data = _payload()
        data["smpl_joints"] = data["smpl_joints"][:, :23]
        invalid.append(data)
        data = _payload(0)
        invalid.append(data)
        data = _payload()
        data["smpl_joints"][:] = 3.5  # Translation alone cannot disguise a collapsed skeleton.
        invalid.append(data)
        for index, data in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_smpl_vr5_data(data)

    def test_runtime_shape_errors_are_explicit(self):
        joints = _payload(2)["smpl_joints"]
        root = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).repeat(2, 1)
        with self.assertRaisesRegex(ValueError, "smpl_joints"):
            smpl_vr5_local_positions(joints[:, :23], root)
        with self.assertRaisesRegex(ValueError, "batch shape"):
            smpl_vr5_local_positions(joints, root[:1])
        with self.assertRaisesRegex(ValueError, "dtype"):
            smpl_vr5_local_positions(joints, root.float())
        with self.assertRaisesRegex(ValueError, "pose_aa"):
            smpl_vr5_local_orientations(torch.zeros(2, 69), root.float())

    def test_quaternion_sign_and_extra_batch_dimensions_are_supported(self):
        pose = torch.zeros(2, 24, 3, dtype=torch.float64)
        pose[:, 18, 0] = 0.7
        root = _root(pose)
        torch.testing.assert_close(smpl_vr5_local_orientations(pose, root),
                                   smpl_vr5_local_orientations(pose, -root))
        joints = _payload(2)["smpl_joints"]
        actual = smpl_vr5_targets(joints.unsqueeze(0), pose.unsqueeze(0), root.unsqueeze(0))
        expected = smpl_vr5_targets(joints, pose, root)
        for value, original in zip(actual, expected):
            torch.testing.assert_close(value, original.unsqueeze(0))
        self.assertEqual(SMPL_VR5_JOINT_INDICES, (20, 21, 15, 7, 8))


if __name__ == "__main__":
    unittest.main()
