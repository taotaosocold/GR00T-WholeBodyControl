"""Numerical parity of physical WAM rewards, without starting Isaac Lab."""

import math
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.utils.config_utils import configure_wam_rewards
from gear_sonic.utils.wam_kinematics import G1_MUJOCO_JOINT_NAMES
from gear_sonic.utils.wam_reward import WAMTrackingReward, tracking_reward_term


TERM_NAMES = (
    "tracking_relative_body_pos", "tracking_relative_body_ori", "tracking_body_linvel",
    "tracking_body_angvel", "tracking_anchor_pos", "tracking_anchor_ori",
    "action_rate_l2", "joint_limit",
)


def make_state(model, batch=2):
    state = torch.zeros(batch, 71, dtype=model.soft_lo.dtype, device=model.soft_lo.device)
    state[:, 2] = 0.8
    state[:, 3] = 1
    state[:, 13:42] = (model.soft_lo + model.soft_hi) / 2
    return state


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.model = WAMTrackingReward().double()
        self.state = make_state(self.model)
        self.action = torch.zeros(2, 29, dtype=torch.float64)

    def test_matching_state_has_maximum_tracking_reward(self):
        values = self.model.components(self.state, self.state, self.action, self.action)
        self.assertEqual(set(values), set(TERM_NAMES))
        for name in TERM_NAMES[:6]:
            torch.testing.assert_close(values[name], torch.ones(2, dtype=torch.float64), atol=2e-5, rtol=0)
        self.assertFalse(values["action_rate_l2"].any())
        self.assertFalse(values["joint_limit"].any())
        torch.testing.assert_close(
            self.model(self.state, self.state, self.action, self.action),
            torch.full((2,), 0.1, dtype=torch.float64), atol=2e-6, rtol=0,
        )

    def test_action_rate_is_sum_and_dt_is_applied_once(self):
        action = self.action + 0.5
        values = self.model.components(self.state, self.state, action, self.action)
        torch.testing.assert_close(values["action_rate_l2"], torch.full((2,), 29 * 0.25, dtype=torch.float64))
        without_rate = self.model(self.state, self.state, self.action, self.action, dt=0.037)
        with_rate = self.model(self.state, self.state, action, self.action, dt=0.037)
        torch.testing.assert_close(without_rate - with_rate, torch.full((2,), 0.1 * 29 * 0.25 * 0.037, dtype=torch.float64))
        torch.testing.assert_close(
            with_rate / self.model(self.state, self.state, action, self.action, dt=0.02),
            torch.full((2,), 0.037 / 0.02, dtype=torch.float64),
        )

    def test_limits_use_individual_joint_ranges_and_sum(self):
        state = self.state.clone()
        state[:, 13] = self.model.soft_hi[0] + 0.2
        state[:, 14] = self.model.soft_lo[1] - 0.3
        values = self.model.components(state, self.state, self.action, self.action)
        torch.testing.assert_close(values["joint_limit"], torch.full((2,), 0.5, dtype=torch.float64))
        self.assertGreater((self.model.soft_hi - self.model.soft_lo).std().item(), 0.1)

    def test_body_targets_are_reanchored_by_robot_xy_and_heading(self):
        state = self.state.clone()
        state[:, 0] += 3
        state[:, 1] -= 4
        angle = 0.7
        state[:, 3] = math.cos(angle / 2)
        state[:, 6] = math.sin(angle / 2)
        expected = self.model.components(self.state, self.state, self.action, self.action)
        values = self.model.components(state, self.state, self.action, self.action)
        for name in ("tracking_relative_body_pos", "tracking_relative_body_ori"):
            torch.testing.assert_close(values[name], expected[name], atol=2e-6, rtol=0)
        self.assertTrue((values["tracking_anchor_pos"] < 1e-5).all())
        self.assertTrue((values["tracking_anchor_ori"] < 0.1).all())

    def test_custom_widths_and_weights_are_shared_by_total(self):
        weights = dict.fromkeys(TERM_NAMES, 0.0)
        weights["tracking_anchor_pos"] = 2.0
        model = WAMTrackingReward(stds={"root_pos": 1.0}, weights=weights).double()
        state = self.state.clone()
        state[:, 0] += 1
        values = model.components(state, self.state, self.action, self.action)
        torch.testing.assert_close(values["tracking_anchor_pos"], torch.full((2,), math.exp(-1), dtype=torch.float64))
        torch.testing.assert_close(
            model(state, self.state, self.action, self.action, dt=0.05),
            torch.full((2,), 2 * math.exp(-1) * 0.05, dtype=torch.float64),
        )

    def test_rewards_supply_finite_joint_velocity_and_action_gradients(self):
        state = self.state.clone()
        state[:, 13:42] += torch.linspace(-0.15, 0.1, 29, dtype=torch.float64)
        state[:, 42:71] = torch.linspace(-0.5, 0.4, 29, dtype=torch.float64)
        state[:, 7:13] = torch.tensor([0.1, -0.2, 0.3, 0.2, -0.3, 0.1])
        state.requires_grad_()
        action = (self.action + 0.1).requires_grad_()
        result = self.model(state, self.state, action, self.action)
        result.sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(state.grad[:, 13:42].abs().sum().item(), 0)
        self.assertGreater(state.grad[:, 42:71].abs().sum().item(), 0)
        self.assertGreater(action.grad.abs().sum().item(), 0)

    def test_arbitrary_leading_batch_dimensions(self):
        state = self.state[:, None].expand(-1, 3, -1)
        action = self.action[:, None].expand(-1, 3, -1)
        output = self.model(state, state, action, action, dt=0.03)
        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.isfinite(output).all())


class MotionLibrary:
    def __init__(self, reference):
        self.reference = reference
        self.calls = []

    def get_time_step_total(self, ids):
        return torch.full_like(ids, 5)

    def _reference(self, ids, frames):
        self.calls.append(frames.clone())
        result = self.reference[ids].clone()
        # The current command is frame zero. Only frame one has the supplied
        # reference; reading frame zero introduces a visible tracking error.
        result[:, 2] += (frames - 1).to(result.dtype) * 0.5
        return result

    def get_body_pos_w(self, ids, frames):
        return self._reference(ids, frames)[:, None, :3]

    def get_body_quat_w(self, ids, frames):
        return self._reference(ids, frames)[:, None, 3:7]

    def get_body_lin_vel_w(self, ids, frames):
        return self._reference(ids, frames)[:, None, 7:10]

    def get_body_ang_vel_w(self, ids, frames):
        return self._reference(ids, frames)[:, None, 10:13]

    def get_dof_pos(self, ids, frames):
        order = [G1_MUJOCO_JOINT_NAMES.index(name) for name in joint_utils.G1_ISAACLab_ORDER]
        return self._reference(ids, frames)[:, 13:42][:, order]

    def get_dof_vel(self, ids, frames):
        order = [G1_MUJOCO_JOINT_NAMES.index(name) for name in joint_utils.G1_ISAACLab_ORDER]
        return self._reference(ids, frames)[:, 42:71][:, order]


def make_env(state, reference, action, previous_action, with_hands=False):
    body_order = [G1_MUJOCO_JOINT_NAMES.index(name) for name in joint_utils.G1_ISAACLab_ORDER]
    names = list(joint_utils.G1_ISAACLab_ORDER)
    q, qd = state[:, 13:42][:, body_order], state[:, 42:71][:, body_order]
    a, ap = action[:, body_order], previous_action[:, body_order]
    if with_hands:
        names += joint_utils.G1_HAND_JOINTS
        q = torch.cat((q, q.new_full((len(state), 14), 30)), dim=-1)
        qd = torch.cat((qd, qd.new_full((len(state), 14), 40)), dim=-1)
        a = torch.cat((a, a.new_full((len(state), 14), 20)), dim=-1)
        ap = torch.cat((ap, ap.new_full((len(state), 14), -20)), dim=-1)
    robot = SimpleNamespace(
        device=state.device, joint_names=names,
        data=SimpleNamespace(joint_pos=q, joint_vel=qd, body_link_lin_vel_w=state[:, None, 7:10]),
    )
    origins = state.new_tensor([[8., 3., 2.], [-4., 6., -3.]])
    library = MotionLibrary(reference)
    command = SimpleNamespace(
        robot=robot, motion_lib=library, motion_ids=torch.arange(len(state)),
        motion_start_time_steps=torch.zeros(len(state), dtype=torch.long),
        time_steps=torch.zeros(len(state), dtype=torch.long), motion_anchor_body_index=0,
        robot_anchor_body_index=0,
        robot_anchor_pos_w=state[:, :3] + origins,
        robot_anchor_quat_w=state[:, 3:7], robot_anchor_lin_vel_w=state[:, 7:10],
        robot_anchor_ang_vel_w=state[:, 10:13],
    )

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.env_origins = origins
    env = SimpleNamespace(
        scene=scene, device=state.device, num_envs=len(state), common_step_counter=1,
        step_dt=0.037, command_manager=SimpleNamespace(get_term=lambda name: command),
        action_manager=SimpleNamespace(action=a, prev_action=ap),
    )
    return env, command


class RewardAdapterTests(unittest.TestCase):
    def test_simulator_components_match_imagined_and_manager_scales_dt_once(self):
        model = WAMTrackingReward().double()
        reference = make_state(model)
        state = reference.clone()
        state[:, 13:42] += torch.linspace(-0.1, 0.15, 29, dtype=torch.float64)
        state[:, 42:71] += 0.2
        state[:, 2] += 0.1
        action = torch.linspace(-0.3, 0.2, 29, dtype=torch.float64).expand(2, -1)
        previous = action * 0.5
        expected = model.components(state, reference, action, previous)
        for with_hands in (False, True):
            with self.subTest(with_hands=with_hands):
                env, command = make_env(state, reference, action, previous, with_hands)
                values = {name: tracking_reward_term(env, name) for name in TERM_NAMES}
                for name in TERM_NAMES:
                    torch.testing.assert_close(values[name], expected[name], atol=2e-5, rtol=2e-5)
                weights = dict(zip(TERM_NAMES, (1, 1, 1, 1, 0.5, 0.5, -0.1, -2)))
                # This is the RewardManager contract: adapter returns rates;
                # the manager applies each configured weight and control dt.
                actual = sum(values[name] * weights[name] * env.step_dt for name in TERM_NAMES)
                torch.testing.assert_close(
                    actual, model(state, reference, action, previous, dt=env.step_dt), atol=2e-5, rtol=2e-5,
                )
                self.assertTrue(command.motion_lib.calls)
                self.assertTrue(all((frames == 1).all() for frames in command.motion_lib.calls))

    def test_reward_terms_share_step_cache_and_refresh_next_step(self):
        model = WAMTrackingReward().double()
        reference = make_state(model)
        state = reference.clone()
        state[:, 2] += 0.1
        action = state.new_zeros(len(state), 29)
        env, command = make_env(state, reference, action, action)
        first = tracking_reward_term(env, "tracking_anchor_pos").clone()
        reads = len(command.motion_lib.calls)
        for name in TERM_NAMES:
            tracking_reward_term(env, name)
        self.assertEqual(len(command.motion_lib.calls), reads)
        env.common_step_counter += 1
        command.robot_anchor_pos_w[:, 2] += 0.1
        second = tracking_reward_term(env, "tracking_anchor_pos")
        self.assertGreater(len(command.motion_lib.calls), reads)
        self.assertTrue((second < first).all())


class RewardConfigTests(unittest.TestCase):
    def make_config(self):
        return OmegaConf.create({
            "trainer": {
                "_target_": "gear_sonic.trl.trainer.wam_trainer.WAMTrainer",
                "settings": {"align_real_reward": True},
            },
            "world_model": {"cfg": {"state_type": "physical"}},
            "manager_env": {"rewards": {
                "_target_": "gear_sonic.envs.manager_env.mdp.rewards.RewardsCfg",
                "feet_acc": {"weight": -2.5e-6},
                "tracking_vr_5point_local": {"weight": 1.0},
            }},
        })

    def test_wam_replaces_complete_group_with_matching_eight_components(self):
        config = self.make_config()
        OmegaConf.set_struct(config, True)
        self.assertTrue(configure_wam_rewards(config))
        self.assertEqual(set(config.manager_env.rewards) - {"_target_"}, set(TERM_NAMES))
        for name, weight in zip(TERM_NAMES, (1, 1, 1, 1, 0.5, 0.5, -0.1, -2)):
            self.assertEqual(config.manager_env.rewards[name].weight, weight)

    def test_ppo_flat_and_old_checkpoint_configs_are_unchanged(self):
        for case in ("ppo", "flat", "old", "disabled"):
            with self.subTest(case=case):
                config = self.make_config()
                if case == "ppo":
                    config.trainer._target_ = "gear_sonic.trl.trainer.ppo_trainer.PPOTrainer"
                elif case == "flat":
                    config.world_model.cfg.state_type = "flat"
                elif case == "old":
                    del config.trainer.settings.align_real_reward
                else:
                    config.trainer.settings.align_real_reward = False
                before = OmegaConf.to_container(config, resolve=False)
                self.assertFalse(configure_wam_rewards(config))
                self.assertEqual(before, OmegaConf.to_container(config, resolve=False))

    def test_custom_weights_survive_save_reload_and_repeated_selection(self):
        config = self.make_config()
        config.trainer.reward_config = {
            "weights": {"tracking_anchor_pos": 0.8, "action_rate_l2": -0.25},
            "stds": {"root_pos": 0.45},
        }
        configure_wam_rewards(config)
        self.assertEqual(config.manager_env.rewards.tracking_anchor_pos.weight, 0.8)
        self.assertEqual(config.manager_env.rewards.action_rate_l2.weight, -0.25)
        before = OmegaConf.to_container(config, resolve=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            OmegaConf.save(config, path)
            restored = OmegaConf.load(path)
            configure_wam_rewards(restored)
        self.assertEqual(before, OmegaConf.to_container(restored, resolve=False))

    def test_aligned_config_rejects_a_different_imagined_reward_implementation(self):
        config = self.make_config()
        config.trainer.reward_config = {"_target_": "torch.nn.Identity"}
        with self.assertRaisesRegex(ValueError, "shared WAMTrackingReward"):
            configure_wam_rewards(config)

    def test_trainer_uses_shared_reward_config_and_simulator_control_dt(self):
        from gear_sonic.tests.test_wam_training import Config, FSQPolicy
        from gear_sonic.trl.trainer.wam_trainer import WAMTrainer

        reward = WAMTrackingReward()
        reference = make_state(reward)
        state = reference.clone()
        state[:, 13:42] += torch.linspace(-0.1, 0.15, 29)
        action = torch.linspace(-0.2, 0.3, 29).expand(2, -1)
        previous = action / 2
        simulator, _ = make_env(state, reference, action, previous)
        observations = {
            "actor_obs": state.new_zeros(2, 4),
            "tokenizer": torch.cat((state.new_zeros(2, 9), torch.eye(3)[:2]), -1),
            "wam_state": state, "wam_reference": reference,
        }
        env = SimpleNamespace(
            env=simulator, num_envs=2, reset_all=lambda: observations,
            config=Config(robot=Config(actions_dim=29), action_clip_value=20),
        )
        config = self.make_config()
        config.trainer.reward_config = {"weights": {"action_rate_l2": -0.25}}
        configure_wam_rewards(config)
        policy = FSQPolicy()
        policy.decoder = torch.nn.Linear(6, 29)
        with tempfile.TemporaryDirectory() as directory:
            trainer = WAMTrainer(
                args=SimpleNamespace(num_total_batches=1), config=Config(gamma=0.99, lam=0.95),
                env=env, model=policy, log_dir=directory,
                settings={
                    "state_observation_keys": ["wam_state"], "goal_observation_keys": ["tokenizer"],
                    "reference_observation_keys": ["wam_reference"], "align_real_reward": True,
                    "replay_steps": 32,
                },
                world_model_config={
                    "_target_": "gear_sonic.trl.modules.wam.StateSpaceWorldModel",
                    "cfg": {"state_type": "physical", "hidden": [8], "reward_mode": "analytic"},
                },
                reward_config=config.trainer.reward_config,
            )
            self.assertIsInstance(trainer.reward_model, WAMTrackingReward)
            self.assertEqual(trainer.control_dt, simulator.step_dt)
            terms = config.manager_env.rewards
            real_reward = sum(
                tracking_reward_term(simulator, name, terms[name].params.reward_config)
                * terms[name].weight * simulator.step_dt for name in TERM_NAMES
            )
            torch.testing.assert_close(
                trainer._analytic_reward(state, reference, action, previous), real_reward,
            )

    def test_real_sonic_config_switches_only_wam_reward_group(self):
        config_dir = Path(__file__).resolve().parents[1] / "config"
        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            ppo = compose(config_name="base", overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release",
            ])
            wam = compose(config_name="base", overrides=[
                "+exp=manager/universal_token/all_modes/sonic_release",
                "trainer=wam", "world_model=wam_dynamics",
            ])
        old = OmegaConf.to_container(ppo.manager_env.rewards, resolve=False)
        self.assertFalse(configure_wam_rewards(ppo))
        self.assertEqual(old, OmegaConf.to_container(ppo.manager_env.rewards, resolve=False))
        self.assertIn("feet_acc", old)
        self.assertTrue(configure_wam_rewards(wam))
        self.assertEqual(set(wam.manager_env.rewards) - {"_target_"}, set(TERM_NAMES))


class KinematicsReferenceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is only needed for FK reference checks")
    def test_link_poses_and_velocities_match_mujoco_body_jacobians(self):
        import mujoco
        import numpy as np

        reward = WAMTrackingReward().double()
        kinematics = reward.kinematics
        model = mujoco.MjModel.from_xml_path(kinematics.mjcf_path)
        data = mujoco.MjData(model)
        generator = np.random.default_rng(71)
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in G1_MUJOCO_JOINT_NAMES
        ]
        qpos_ids = [model.jnt_qposadr[index] for index in joint_ids]
        dof_ids = [model.jnt_dofadr[index] for index in joint_ids]
        body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in kinematics.tracked_bodies
        ]
        for _ in range(3):
            root_quat = generator.normal(size=4)
            root_quat /= np.linalg.norm(root_quat)
            data.qpos[:3] = generator.normal(size=3)
            data.qpos[3:7] = root_quat
            data.qpos[qpos_ids] = generator.uniform(
                reward.soft_lo.numpy(), reward.soft_hi.numpy(),
            )
            data.qvel[:] = generator.normal(size=model.nv)
            mujoco.mj_forward(model, data)
            jac_p, jac_r = np.empty((3, model.nv)), np.empty((3, model.nv))
            mujoco.mj_jacBody(model, data, jac_p, jac_r, root_id)
            state = torch.tensor(np.concatenate((
                data.xpos[root_id], data.xquat[root_id], jac_p @ data.qvel,
                jac_r @ data.qvel, data.qpos[qpos_ids], data.qvel[dof_ids],
            )), dtype=torch.float64)
            position, rotation, velocity, angular = kinematics(state)
            expected_velocity, expected_angular = [], []
            for body_id in body_ids:
                # mj_objectVelocity is a CoM convention in some versions;
                # body Jacobians give the required link-origin velocity.
                mujoco.mj_jacBody(model, data, jac_p, jac_r, body_id)
                expected_velocity.append((jac_p @ data.qvel).copy())
                expected_angular.append((jac_r @ data.qvel).copy())
            for actual, expected in (
                (position, data.xpos[body_ids]),
                (rotation, data.xmat[body_ids].reshape(-1, 3, 3)),
                (velocity, np.array(expected_velocity)),
                (angular, np.array(expected_angular)),
            ):
                torch.testing.assert_close(
                    actual, torch.tensor(expected, dtype=torch.float64), atol=2e-6, rtol=2e-6,
                )


if __name__ == "__main__":
    unittest.main()
