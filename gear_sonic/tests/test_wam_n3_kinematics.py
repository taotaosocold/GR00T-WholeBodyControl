"""N3 FK, shared rewards/critic and immutable references without Isaac Sim."""

import importlib.util
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.envs.env_utils.n3_utils import (
    N3_ISAACLAB_DOF_NAMES, N3_MUJOCO_JOINT_NAMES, N3_TRACKED_BODIES,
)
from gear_sonic.utils.wam_critic import WAMCriticFeatures, future_reference_states
from gear_sonic.utils.wam_kinematics import G1Kinematics, RobotKinematics, make_robot_kinematics
from gear_sonic.utils.wam_reference import WAMReferenceBank
from gear_sonic.utils.wam_reward import WAMTrackingReward, reference_state, tracking_reward_term


def state(batch=2):
    result = torch.zeros(batch, 71, dtype=torch.float64)
    result[:, 2] = 0.724
    result[:, 3] = 1
    return result


class N3ReferenceLibrary:
    def get_time_step_total(self, ids):
        return torch.full_like(ids, 6)

    def get_body_pos_w(self, ids, frames):
        result = torch.zeros(len(ids), 1, 3, dtype=torch.float64)
        result[:, 0, 0], result[:, 0, 2] = frames, 0.724
        return result

    def get_body_quat_w(self, ids, frames):
        result = torch.zeros(len(ids), 1, 4, dtype=torch.float64)
        result[..., 0] = 1
        return result

    def get_body_lin_vel_w(self, ids, frames):
        return torch.zeros(len(ids), 1, 3, dtype=torch.float64)

    get_body_ang_vel_w = get_body_lin_vel_w

    def get_dof_pos(self, ids, frames):
        return frames[:, None].double() * 100 + torch.arange(29)[None]

    def get_dof_vel(self, ids, frames):
        return frames[:, None].double() * 10 + torch.arange(29)[None]


class N3Command:
    def __init__(self):
        self._env = SimpleNamespace(num_envs=2, device="cpu")
        self.robot = SimpleNamespace(joint_names=tuple(reversed(N3_ISAACLAB_DOF_NAMES)))
        self.motion_lib = N3ReferenceLibrary()
        self.motion_ids = torch.zeros(2, dtype=torch.long)
        self.time_steps = torch.tensor([0, 4])
        self.motion_start_time_steps = torch.zeros(2, dtype=torch.long)
        self.motion_num_steps = torch.full((2,), 6)
        self.motion_anchor_body_index = self.robot_anchor_body_index = 0
        self.num_future_frames = self.smpl_num_future_frames = 2
        self.future_time_steps_init = self.smpl_future_time_steps_init = torch.tensor([[0, 2], [0, 2]])
        self.vr_3point_body_offsets = torch.zeros(2, 3, 3)
        self.encoder_index = torch.tensor([[1., 0., 0.], [0., 1., 0.]])

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device


class N3KinematicsTests(unittest.TestCase):
    def test_n3_geometry_uses_native_joint_limits_base_and_fixed_toes(self):
        kinematics = make_robot_kinematics("n3")
        self.assertIsInstance(kinematics, RobotKinematics)
        self.assertEqual(kinematics.root_body, "base_link")
        self.assertEqual(kinematics.joint_names, tuple(N3_MUJOCO_JOINT_NAMES))
        self.assertEqual(kinematics.tracked_bodies, tuple(N3_TRACKED_BODIES))
        self.assertIn("head_yaw_link", kinematics.tracked_bodies)
        xml = ET.parse(kinematics.mjcf_path).getroot()
        limits = {j.get("name"): [float(x) for x in j.get("range").split()]
                  for j in xml.findall(".//worldbody//joint") if j.get("type", "hinge") == "hinge"}
        expected = torch.tensor([limits[name] for name in N3_MUJOCO_JOINT_NAMES])
        torch.testing.assert_close(kinematics.q_lo, expected[:, 0])
        torch.testing.assert_close(kinematics.q_hi, expected[:, 1])
        for name in ("l_toe_link", "r_toe_link"):
            self.assertEqual(kinematics.joint_indices[kinematics.body_names.index(name)], -1)
        with self.assertRaisesRegex(ValueError, "floating pelvis"):
            RobotKinematics(kinematics.mjcf_path, N3_TRACKED_BODIES, N3_MUJOCO_JOINT_NAMES, "pelvis")

    def test_g1_wrapper_keeps_buffers_and_results_identical_to_generic_implementation(self):
        legacy = G1Kinematics().double()
        generic = RobotKinematics(
            legacy.mjcf_path, legacy.tracked_bodies, legacy.joint_names, "pelvis",
        ).double()
        self.assertEqual(set(legacy.state_dict()), set(generic.state_dict()))
        for name, value in legacy.state_dict().items():
            torch.testing.assert_close(generic.state_dict()[name], value)
        for expected, actual in zip(legacy(state()), generic(state())):
            torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo validates FK independently")
    def test_n3_body_poses_velocities_and_fixed_toes_match_mujoco_jacobians(self):
        import mujoco

        reference = make_robot_kinematics("n3")
        # Visual meshes are irrelevant to FK. Remove them in memory so this
        # check depends only on the committed body tree, joints and inertials.
        xml = ET.parse(reference.mjcf_path).getroot()
        for element in list(xml):
            if element.tag in {"asset", "actuator", "sensor"}:
                xml.remove(element)
        for body in xml.findall(".//worldbody//body"):
            for geom in body.findall("geom"):
                body.remove(geom)
        mj_model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        data = mujoco.MjData(mj_model)
        tracked = (*N3_TRACKED_BODIES, "l_toe_link", "r_toe_link")
        kinematics = RobotKinematics(reference.mjcf_path, tracked, N3_MUJOCO_JOINT_NAMES, "base_link").double()
        ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in N3_MUJOCO_JOINT_NAMES]
        qpos_ids, dof_ids = mj_model.jnt_qposadr[ids], mj_model.jnt_dofadr[ids]
        body_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name) for name in tracked]
        root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        generator = np.random.default_rng(318)
        jac_p, jac_r = np.empty((3, mj_model.nv)), np.empty((3, mj_model.nv))
        for _ in range(3):
            data.qpos[:3] = generator.normal(size=3)
            quaternion = generator.normal(size=4)
            data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
            data.qpos[qpos_ids] = generator.uniform(kinematics.q_lo.numpy(), kinematics.q_hi.numpy())
            data.qvel[:] = generator.normal(size=mj_model.nv)
            mujoco.mj_forward(mj_model, data)
            mujoco.mj_jacBody(mj_model, data, jac_p, jac_r, root_id)
            robot = torch.tensor(np.concatenate((
                data.xpos[root_id], data.xquat[root_id], jac_p @ data.qvel,
                jac_r @ data.qvel, data.qpos[qpos_ids], data.qvel[dof_ids],
            )), dtype=torch.float64)
            velocities, angular = [], []
            for body_id in body_ids:
                mujoco.mj_jacBody(mj_model, data, jac_p, jac_r, body_id)
                velocities.append((jac_p @ data.qvel).copy())
                angular.append((jac_r @ data.qvel).copy())
            expected = (data.xpos[body_ids], data.xmat[body_ids].reshape(-1, 3, 3), velocities, angular)
            for actual, value in zip(kinematics(robot), expected):
                torch.testing.assert_close(actual, torch.tensor(np.asarray(value)), atol=2e-6, rtol=2e-6)

    def test_n3_reward_and_shared_critic_support_temporal_batches_and_gradients(self):
        reward = WAMTrackingReward(robot_type="n3").double()
        robot = state()[:, None].expand(-1, 3, -1).clone()
        robot[..., 13:42] = (reward.soft_lo + reward.soft_hi) / 2
        target = robot.clone()
        robot[..., 0] += 0.2
        robot[..., 13:42] += 0.03
        robot[..., 42:71] = 0.1
        robot.requires_grad_()
        action = torch.full((2, 3, 29), 0.1, dtype=torch.float64, requires_grad=True)
        values = reward(robot, target, action, torch.zeros_like(action))
        critic = WAMCriticFeatures(robot_type="n3", future_frames=2, context_dim=0).double()
        future = target.unsqueeze(-2).expand(-1, -1, 2, -1)
        features = critic(robot, target, None, future)
        self.assertEqual(values.shape, (2, 3))
        self.assertEqual(features.shape, (2, 3, critic.input_dim))
        (values.sum() + 0.001 * features.square().sum()).backward()
        for gradient in (robot.grad, action.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)


class N3ReferenceTests(unittest.TestCase):
    def test_real_reward_callbacks_match_imagined_n3_reward(self):
        command = N3Command()
        reward = WAMTrackingReward(robot_type="n3").double()
        robot = state()
        robot[:, 13:42] = (reward.soft_lo + reward.soft_hi) / 2
        robot[:, 42:71] = 0.2
        source = tuple(command.robot.joint_names)
        to_source = [N3_MUJOCO_JOINT_NAMES.index(name) for name in source]
        command.robot.data = SimpleNamespace(
            joint_pos=robot[:, 13:42][:, to_source], joint_vel=robot[:, 42:71][:, to_source],
            body_link_lin_vel_w=robot[:, None, 7:10],
        )
        origins = torch.tensor([[10., 20., 0.], [-5., -3., 0.]], dtype=torch.float64)
        command.robot_anchor_pos_w = robot[:, :3] + origins
        command.robot_anchor_quat_w = robot[:, 3:7]
        command.robot_anchor_ang_vel_w = robot[:, 10:13]
        action = torch.linspace(-0.2, 0.3, 58, dtype=torch.float64).reshape(2, 29)
        previous = torch.zeros_like(action)
        env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda _: command),
            scene=SimpleNamespace(env_origins=origins), common_step_counter=1,
            action_manager=SimpleNamespace(action=action[:, to_source], prev_action=previous),
        )
        expected = reward.components(robot, reference_state(command), action, previous)
        for name, value in expected.items():
            actual = tracking_reward_term(env, name, reward_config={"robot_type": "n3"})
            torch.testing.assert_close(actual, value)

    def test_current_next_and_future_references_follow_actual_robot_order(self):
        command = N3Command()
        source = tuple(command.robot.joint_names)
        order = torch.tensor([source.index(name) for name in N3_MUJOCO_JOINT_NAMES])
        future = future_reference_states(command, num_frames=3, frame_stride=2)
        frames = torch.tensor([[0., 2., 4.], [4., 5., 5.]], dtype=torch.float64)
        torch.testing.assert_close(future[..., 13:42], frames[..., None] * 100 + order)
        torch.testing.assert_close(future[..., 42:71], frames[..., None] * 10 + order)
        expected_next = future_reference_states(command, num_frames=1, next_step=True).squeeze(1)
        torch.testing.assert_close(reference_state(command), expected_next)
        self.assertFalse(torch.equal(order, torch.arange(29)))

    def test_reference_bank_proxy_preserves_n3_identity_and_joint_order(self):
        command = N3Command()
        specs = (("encoder_index", 0, 3, (3,)), ("command_z", 3, 4, (1,)))
        manager = SimpleNamespace(
            _group_obs_term_names={"tokenizer": ["encoder_index", "command_z"]},
            _group_obs_term_cfgs={"tokenizer": [
                SimpleNamespace(func=lambda env: env.command_manager.get_term("motion").encoder_index, params={}),
                SimpleNamespace(func=lambda env: env.command_manager.get_term("motion").time_steps[:, None].float(), params={}),
            ]},
        )
        bank = WAMReferenceBank(specs)
        indices = bank.capture(command, manager)
        goals = torch.cat((command.encoder_index, command.time_steps[:, None]), -1)
        trajectory = bank.trajectory(indices, goals, 3, 2, 1, "cpu")
        expected = future_reference_states(command, num_frames=4)
        torch.testing.assert_close(trajectory["references"], expected.float())
        proxy, _ = bank._command_view(command, 0, torch.tensor([0, 1, 2]), 6)
        self.assertEqual(joint_utils.get_robot_type(proxy.robot), "n3")
        self.assertEqual(tuple(proxy.robot.joint_names), tuple(command.robot.joint_names))


if __name__ == "__main__":
    unittest.main()
