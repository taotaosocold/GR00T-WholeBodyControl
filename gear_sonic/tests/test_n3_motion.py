"""N3 motion ordering/FK regressions that do not require an Isaac simulator."""

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from easydict import EasyDict
import torch

from gear_sonic.envs.env_utils.n3_utils import (
    N3_ISAACLAB_BODY_NAMES, N3_ISAACLAB_DOF_NAMES, N3_LOWER_BODY_JOINT_NAMES,
    N3_MJCF_PATH, N3_MUJOCO_BODY_NAMES, N3_MUJOCO_JOINT_NAMES, N3_WRIST_JOINT_NAMES,
)
from gear_sonic.trl.utils.order_converter import IsaacLabMuJoCoConverter, N3Converter
from gear_sonic.trl.utils import order_converter
from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from gear_sonic.utils.motion_lib.motion_lib_base import FixHeightMode, MotionLibBase


class N3MotionTests(unittest.TestCase):
    def test_offline_command_uses_n3_mapping_and_native_leg_groups(self):
        # Load the real offline factory without importing simulator-only base
        # classes; the motion-loader boundary is the only substituted object.
        source = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/commands.py"
        tree = ast.parse(source.read_text())
        tracking = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrackingCommand")
        factory = next(node for node in tracking.body if isinstance(node, ast.FunctionDef) and node.name == "create_offline")
        initializer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_init_variable_frames")
        declaration = ast.ClassDef(name="OfflineCommand", bases=[], keywords=[], body=[factory], decorator_list=[])

        class MotionLoader:
            def __init__(self, config, num_envs, device):
                self.config = config
            def load_motions_for_training(self, **kwargs):
                pass

        namespace = {"torch": torch, "easydict": SimpleNamespace(EasyDict=EasyDict),
                     "order_converter": order_converter,
                     "motion_lib_robot": SimpleNamespace(MotionLibRobot=MotionLoader)}
        module = ast.fix_missing_locations(ast.Module(body=[initializer, declaration], type_ignores=[]))
        exec(compile(module, str(source), "exec"), namespace)
        command = namespace["OfflineCommand"].create_offline(EasyDict(target_fps=50), "cpu", robot_type="n3")
        self.assertEqual(command.robot_type, "n3")
        self.assertEqual(command.lower_joint_indices_mujoco, list(range(17, 29)))
        self.assertEqual(command.motion_lib.config["wrist_mujoco_dof_indices"], [5, 6, 7, 12, 13, 14])
        self.assertEqual(len(command.motion_lib.config["mujoco_to_isaaclab_body"]), 32)

    def test_independent_full_body_mapping_includes_fixed_toes(self):
        converter = N3Converter()
        dof = torch.arange(29).float()
        qpos = torch.cat((torch.arange(7).float() + 100, dof))
        body = torch.arange(32).float()[:, None].expand(-1, 3)
        for value in (dof, qpos, body, torch.eye(3).expand(32, 3, 3).clone()):
            torch.testing.assert_close(converter.to_isaaclab(converter.to_mujoco(value)), value)
        expected = torch.tensor([N3_ISAACLAB_BODY_NAMES.index(name) for name in N3_MUJOCO_BODY_NAMES]).float()
        torch.testing.assert_close(converter.to_mujoco(body)[:, 0], expected)
        self.assertEqual(len(converter.BODY_MAPPINGS[("isaaclab", "mujoco")]), 32)
        self.assertEqual(converter.motion_metadata["lower_joint_indices_mujoco"], list(range(17, 29)))

    def test_actual_articulation_orders_override_bfs_without_losing_names(self):
        joint_names = tuple(reversed(N3_ISAACLAB_DOF_NAMES))
        body_names = tuple(reversed(N3_ISAACLAB_BODY_NAMES))
        converter = N3Converter(joint_names=joint_names, body_names=body_names)
        ordered = converter.to_mujoco(torch.arange(29).float())
        expected = torch.tensor([joint_names.index(name) for name in N3_MUJOCO_JOINT_NAMES]).float()
        torch.testing.assert_close(ordered, expected)
        for args in ({"joint_names": joint_names[:-1]}, {"body_names": body_names[:-1]},
                     {"joint_names": joint_names[:-1] + (joint_names[0],)}):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "N3 .* mismatch"):
                N3Converter(**args)

    def test_motion_groups_are_names_in_native_order(self):
        metadata = N3Converter().motion_metadata
        self.assertEqual([N3_MUJOCO_JOINT_NAMES[i] for i in metadata["lower_joint_indices_mujoco"]],
                         list(N3_LOWER_BODY_JOINT_NAMES))
        self.assertEqual([N3_MUJOCO_JOINT_NAMES[i] for i in metadata["wrist_mujoco_dof_indices"]],
                         list(N3_WRIST_JOINT_NAMES))
        self.assertEqual(N3_MUJOCO_BODY_NAMES[metadata["left_foot_body_idx"][0]], "l_leg_ankle_roll_link")
        self.assertEqual(N3_MUJOCO_BODY_NAMES[metadata["right_foot_body_idx"][0]], "r_leg_ankle_roll_link")
        self.assertNotIn("head_pitch_joint", N3_WRIST_JOINT_NAMES)

    def test_legacy_dof_body_conversion_still_works_without_full_body_map(self):
        class LegacyConverter(IsaacLabMuJoCoConverter):
            DOF_MAPPINGS = {("isaaclab", "mujoco"): [1, 0], ("mujoco", "isaaclab"): [1, 0]}
        converter = LegacyConverter()
        value = torch.arange(9).float().reshape(3, 3)
        torch.testing.assert_close(converter.to_mujoco(value), value[[0, 2, 1]])

    def test_actuator_free_fk_keeps_fixed_body_inside_tree(self):
        xml = """<mujoco><worldbody><body name="base"><joint name="root" type="free"/>
        <body name="first" pos="0 0 1"><joint name="j1" axis="0 0 1" range="-3 3"/>
        <body name="toe" pos="1 0 0"/></body>
        <body name="second" pos="0 2 0"><joint name="j2" axis="0 -1 0" range="-3 3"/></body>
        </body></worldbody></mujoco>"""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "test.xml").write_text(xml)
            cfg = EasyDict(robot_type="n3", asset=EasyDict(assetRoot=directory, assetFileName="test.xml"), extend_config=[])
            # Rendering meshes are independent of the training FK equations.
            with patch.object(Humanoid_Batch, "load_mesh"):
                fk = Humanoid_Batch(cfg)
            self.assertEqual(fk.num_dof, 2)
            self.assertEqual(fk.actuated_joints_idx.tolist(), [1, 3])
            qpos = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0.4, -0.3]])
            positions, rotations = fk.qpos_to_global_transforms(qpos)
            torch.testing.assert_close(positions[0, 2], torch.tensor([torch.cos(torch.tensor(.4)), torch.sin(torch.tensor(.4)), 1.]))
            torch.testing.assert_close(positions[0, 3], torch.tensor([0., 2., 0.]))
            torch.testing.assert_close(fk.global_transforms_to_qpos(rotations, positions), qpos, atol=1e-6, rtol=1e-6)
            # root+hinge pose input and explicit body-pose input must agree.
            pose = torch.zeros(1, 4, 3, 3)
            pose[..., 1, 2] = .4
            pose[..., 2, 1] = .3  # negative hinge axis times -0.3 radians
            result = fk.fk_batch(pose, torch.zeros(1, 4, 3), return_full=True)
            torch.testing.assert_close(result.dof_pos, torch.tensor([.4, -.3]).expand(1, 4, 2), atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(result.global_translation[:, 0], positions)

    def test_authoritative_n3_fk_qpos_roundtrip_has_all_toe_bodies(self):
        path = Path(__file__).resolve().parents[2] / N3_MJCF_PATH
        cfg = EasyDict(robot_type="n3", asset=EasyDict(assetRoot=str(path.parent), assetFileName=path.name), extend_config=[])
        with patch.object(Humanoid_Batch, "load_mesh"):
            fk = Humanoid_Batch(cfg)
        self.assertEqual(tuple(fk.body_names), N3_MUJOCO_BODY_NAMES)
        self.assertEqual((fk.num_dof, fk.num_bodies), (29, 32))
        generator = torch.Generator().manual_seed(58)
        qpos = torch.cat((torch.randn(5, 3, generator=generator), torch.tensor([1., 0., 0., 0.]).expand(5, 4),
                          .2 * torch.randn(5, 29, generator=generator)), dim=-1)
        positions, rotations = fk.qpos_to_global_transforms(qpos)
        self.assertEqual(positions.shape, (5, 32, 3))
        torch.testing.assert_close(fk.global_transforms_to_qpos(rotations, positions), qpos, atol=2e-6, rtol=2e-6)

    def test_motion_loader_wrist_augmentation_does_not_touch_head_or_legs(self):
        metadata = N3Converter().motion_metadata
        path = Path(__file__).resolve().parents[2] / N3_MJCF_PATH
        cfg = EasyDict(metadata, asset=EasyDict(assetRoot=str(path.parent), assetFileName=path.name), extend_config=[])
        with patch.object(Humanoid_Batch, "load_mesh"):
            fk = Humanoid_Batch(cfg)
        # Exercise the real loader/FK augmentation path without loading user
        # datasets or starting multiprocessing workers.
        loader = object.__new__(MotionLibBase)
        attributes = dict(m_cfg=EasyDict(metadata), randomize_upper_body_poses=False,
                          randomize_wrist_poses=True, randomize_wrist_prob=1., randomize_wrist_std=.1,
                          wrist_mujoco_dof_indices=metadata["wrist_mujoco_dof_indices"], mesh_parsers=fk,
                          target_fps=50, use_parallel_fk=False, smpl_data=None, object_data=None,
                          has_action=False, vid_smpl_pose=None)
        for key, value in attributes.items():
            setattr(loader, key, value)
        motion = {"root_trans_offset": torch.zeros(4, 3), "pose_aa": torch.zeros(4, 32, 3), "fps": 50}
        torch.manual_seed(16)
        result = loader.load_motion_with_skeleton([0], [motion], None, None, None,
                                                 FixHeightMode.no_fix, None, -1, False, None, 1)[0][1]
        wrist = metadata["wrist_mujoco_dof_indices"]
        others = [i for i in range(29) if i not in wrist]
        self.assertTrue((result.dof_pos[:, wrist].abs().sum(0) > 0).all())
        torch.testing.assert_close(result.dof_pos[:, others], torch.zeros(4, len(others)), atol=0, rtol=0)
        positions = torch.ones(4, 32, 3)
        positions[:, metadata["left_foot_body_idx"], 2] = 0
        left, right = loader.foot_detect(positions, .0005, .05)
        self.assertTrue((left == 1).all())
        self.assertTrue((right == 0).all())

    def test_authoritative_fk_matches_mujoco_for_random_root_rotations(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo is unavailable")
        path = Path(__file__).resolve().parents[2] / N3_MJCF_PATH
        cfg = EasyDict(robot_type="n3", asset=EasyDict(assetRoot=str(path.parent), assetFileName=path.name), extend_config=[])
        with patch.object(Humanoid_Batch, "load_mesh"):
            fk = Humanoid_Batch(cfg)
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        generator = torch.Generator().manual_seed(903)
        for _ in range(5):
            qpos = .2 * torch.randn(36, generator=generator)
            qpos[3:7] /= qpos[3:7].norm()
            positions, rotations = fk.qpos_to_global_transforms(qpos[None])
            data.qpos[:] = qpos.numpy()
            mujoco.mj_forward(model, data)
            torch.testing.assert_close(positions[0], torch.from_numpy(data.xpos[1:]).float(), atol=4e-7, rtol=2e-6)
            torch.testing.assert_close(rotations[0], torch.from_numpy(data.xmat[1:].reshape(-1, 3, 3)).float(),
                                       atol=4e-7, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
