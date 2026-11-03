"""Synthetic CPU integration of N3 WAM/FSQ; this is not a convergence test."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.envs.env_utils.n3_utils import (
    N3_DEFAULT_JOINT_POS, N3_ISAACLAB_BODY_NAMES, N3_ISAACLAB_DOF_NAMES,
    N3_MUJOCO_JOINT_NAMES,
)
from gear_sonic.tests.test_wam_physical_training import (
    Config, PhysicalFSQPolicy, PROPRIO_DIMS, PROPRIO_NAMES, TOKENIZER_SPECS,
)
from gear_sonic.trl.modules.wam import physical_state_parts
from gear_sonic.trl.trainer.wam_trainer import WAMTrainer
from gear_sonic.utils.wam_context import physical_context, physical_context_schema
from gear_sonic.utils.wam_critic import future_reference_states
from gear_sonic.utils.wam_reward import WAMTrackingReward, tracking_reward_term
from gear_sonic.utils.wam_state import robot_state
from gear_sonic.utils.wam_termination import capture_terminal_state


class Scene(dict):
    def __init__(self, robot, num_envs):
        super().__init__(robot=robot)
        self.num_envs = num_envs
        self.env_origins = torch.arange(num_envs)[:, None] * torch.tensor([[2., 3., 0.]])


class SyntheticPhysicsView:
    """Explicit synthetic context values, labelled with actual N3 body names."""

    def __init__(self, num_envs, bodies):
        self.max_shapes = len(bodies)
        self.materials = torch.tensor([0.8, 0.6, 0.0]).expand(num_envs, len(bodies), -1).clone()
        self.masses = torch.linspace(0.2, 2., len(bodies)).expand(num_envs, -1).clone()
        self.inertias = torch.eye(3).flatten().expand(num_envs, len(bodies), -1).clone()
        self.coms = torch.zeros(num_envs, len(bodies), 7)
        self.coms[..., -1] = 1
        torso = bodies.index("waist_yaw_link")
        self.coms[:, torso, :3] = torch.arange(1, num_envs + 1)[:, None] * 0.001

    def get_material_properties(self):
        return self.materials

    def get_masses(self):
        return self.masses

    def get_inertias(self):
        return self.inertias

    def get_coms(self):
        return self.coms


class SyntheticN3Motion:
    """One artificial reference clip, in the live policy's joint order."""

    def __init__(self, policy_names):
        self.to_policy = [N3_MUJOCO_JOINT_NAMES.index(name) for name in policy_names]
        self.states = torch.zeros(16, 71)
        self.states[:, 0] = torch.arange(16) * 0.001
        self.states[:, 2] = 0.724
        angles = torch.arange(16) * 0.01
        self.states[:, 3], self.states[:, 6] = (angles / 2).cos(), (angles / 2).sin()
        self.states[:, 13:42] = torch.tensor(N3_DEFAULT_JOINT_POS)[None] + torch.arange(16)[:, None] * 0.001
        self.states[:, 42:71] = 0.01

    def get_time_step_total(self, ids):
        return torch.full_like(ids, len(self.states))

    def get_body_pos_w(self, ids, frames):
        return self.states[frames, None, :3]

    def get_body_quat_w(self, ids, frames):
        return self.states[frames, None, 3:7]

    def get_body_lin_vel_w(self, ids, frames):
        return self.states[frames, None, 7:10]

    def get_body_ang_vel_w(self, ids, frames):
        return self.states[frames, None, 10:13]

    def get_dof_pos(self, ids, frames):
        return self.states[frames, 13:42][:, self.to_policy]

    def get_dof_vel(self, ids, frames):
        return self.states[frames, 42:71][:, self.to_policy]


class SyntheticN3Command:
    def __init__(self, simulator, robot):
        self._env, self.robot = simulator, robot
        self.motion_lib = SyntheticN3Motion(robot.joint_names)
        self.motion_ids = torch.zeros(simulator.num_envs, dtype=torch.long)
        self.time_steps = torch.zeros_like(self.motion_ids)
        self.motion_start_time_steps = torch.zeros_like(self.motion_ids)
        self.motion_num_steps = torch.full_like(self.motion_ids, 16)
        self.robot_anchor_body_index = self.motion_anchor_body_index = 0
        self.num_future_frames = self.smpl_num_future_frames = 2
        self.future_time_steps_init = self.smpl_future_time_steps_init = torch.tensor([[0, 1]]).expand(simulator.num_envs, -1)
        self.vr_3point_body_offsets = torch.zeros(simulator.num_envs, 3, 3)
        self.encoder_index = torch.eye(3)[torch.arange(simulator.num_envs) % 3]

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device

    @property
    def robot_anchor_pos_w(self):
        return self.robot.data.body_pos_w[:, 0]

    @property
    def robot_anchor_quat_w(self):
        return self.robot.data.body_quat_w[:, 0]

    @property
    def robot_anchor_ang_vel_w(self):
        return self.robot.data.body_ang_vel_w[:, 0]


def goal_term(env, name):
    command = env.command_manager.get_term("motion")
    if name == "encoder_index":
        return command.encoder_index
    cursor = command.time_steps + command.motion_start_time_steps
    if name == "vr_3point_local_target":
        return (torch.arange(9).reshape(1, 3, 3) * 0.01 + cursor[:, None, None] * 0.001).float()
    times = (cursor[:, None] + torch.tensor([0, 1])).clamp_max(15)
    reference = command.motion_lib.states[times]
    robot = reference[:, 0].clone()
    robot[:, 3:7] = command.robot_anchor_quat_w
    rotation = physical_state_parts(robot)[1]
    reference_rotation = physical_state_parts(reference)[1]
    relative = (rotation[:, None].transpose(-1, -2) @ reference_rotation)[..., :2].flatten(-2)
    return relative[:, 0] if name == "motion_anchor_ori_b" else relative


class SyntheticN3Env:
    def __init__(self):
        self.num_envs = 6
        self.config = Config(robot=Config(type="n3", actions_dim=29), action_clip_value=20.)
        self.action_transform_module = None
        # Deliberately differ from expected BFS to exercise live-name mapping.
        names = tuple(reversed(N3_ISAACLAB_DOF_NAMES))
        self.to_native = [names.index(name) for name in N3_MUJOCO_JOINT_NAMES]
        self.to_policy = [N3_MUJOCO_JOINT_NAMES.index(name) for name in names]
        nominal = torch.tensor(N3_DEFAULT_JOINT_POS)[self.to_policy]
        offsets = torch.arange(1, self.num_envs + 1)[:, None] * torch.linspace(0.001, 0.003, 29)[None]
        physics = SyntheticPhysicsView(self.num_envs, list(N3_ISAACLAB_BODY_NAMES))
        self.robot = SimpleNamespace(
            joint_names=names, body_names=list(N3_ISAACLAB_BODY_NAMES), device=torch.device("cpu"),
            data=SimpleNamespace(
                default_joint_pos=nominal + offsets, default_joint_pos_nominal=nominal,
                joint_pos=torch.zeros(self.num_envs, 29), joint_vel=torch.zeros(self.num_envs, 29),
            ),
            root_physx_view=physics, _wam_shapes_per_body=(1,) * len(N3_ISAACLAB_BODY_NAMES),
            _wam_nominal_coms=torch.zeros_like(physics.coms),
        )
        self.state = torch.zeros(self.num_envs, 71)
        self.state[:, 2:4] = torch.tensor([0.724, 1.])
        self.state[:, 13:42] = (nominal + offsets)[:, self.to_native] + 0.01
        self.actions = torch.zeros(self.num_envs, 29)
        self.env = SimpleNamespace(
            num_envs=self.num_envs, device="cpu", scene=Scene(self.robot, self.num_envs),
            step_dt=0.02, common_step_counter=0,
            action_manager=SimpleNamespace(action=self.actions, prev_action=self.actions.clone()),
        )
        self.command = SyntheticN3Command(self.env, self.robot)
        self.env.command_manager = SimpleNamespace(get_term=lambda _: self.command)
        configs = [SimpleNamespace(
            func=goal_term, params={"name": name}, modifiers=None, noise=None, clip=None, scale=None,
        ) for name, *_ in TOKENIZER_SPECS]
        self.env.observation_manager = SimpleNamespace(
            _group_obs_term_names={"policy": PROPRIO_NAMES, "wam": PROPRIO_NAMES,
                                   "tokenizer": [spec[0] for spec in TOKENIZER_SPECS]},
            _group_obs_term_dim={"policy": PROPRIO_DIMS, "wam": PROPRIO_DIMS},
            _group_obs_term_cfgs={"tokenizer": configs},
        )

    def _obs(self):
        data = self.robot.data
        data.joint_pos = self.state[:, 13:42][:, self.to_policy]
        data.joint_vel = self.state[:, 42:71][:, self.to_policy]
        data.body_pos_w = (self.state[:, :3] + self.env.scene.env_origins)[:, None]
        data.body_quat_w = self.state[:, None, 3:7]
        data.body_link_lin_vel_w = self.state[:, None, 7:10]
        data.body_ang_vel_w = self.state[:, None, 10:13]
        _, rotation, _, angular, q, qd = physical_state_parts(self.state)
        frame = torch.cat((
            -rotation[:, 2], (rotation.transpose(-1, -2) @ angular[..., None]).squeeze(-1),
            q[:, self.to_policy] - data.default_joint_pos, qd[:, self.to_policy], self.actions,
        ), -1)
        context = physical_context(self.robot)
        self.env._wam_context_schema = physical_context_schema(self.robot)
        return {
            "actor_obs": frame.clone(), "wam": frame, "wam_state": robot_state(self.env),
            "wam_context": context, "wam_reference": future_reference_states(self.command, num_frames=1).squeeze(1),
            "tokenizer": torch.cat([goal_term(self.env, name).flatten(1) for name, *_ in TOKENIZER_SPECS], -1),
        }

    def reset_all(self):
        return self._obs()

    def step(self, payload):
        previous = self.actions.clone()
        self.actions = payload["actions"].detach().clone()
        native = self.actions[:, self.to_native]
        self.state[:, 13:42] += 0.001 * native
        self.state[:, 42:71] = native
        self.state[:, 0] += 0.001
        angle = (self.env.common_step_counter + 1) * 0.01
        self.state[:, 3], self.state[:, 6] = torch.cos(torch.tensor(angle / 2)), torch.sin(torch.tensor(angle / 2))
        self.env.common_step_counter += 1
        self.env.action_manager.action, self.env.action_manager.prev_action = self.actions, previous
        self._obs()  # Refresh fake articulation data before real reward callbacks.
        capture_terminal_state(self.env)
        weights = WAMTrackingReward(robot_type="n3").weights
        rewards = sum(weight * tracking_reward_term(self.env, name, {"robot_type": "n3"})
                      for name, weight in weights.items()) * self.env.step_dt
        self.command.time_steps += 1
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), rewards, dones, {"terminated": dones}


def make_trainer(directory, **overrides):
    torch.manual_seed(382)
    env = SyntheticN3Env()
    settings = dict(
        state_observation_keys=["wam_state"], goal_observation_keys=["tokenizer"],
        context_observation_keys=["wam_context"], reference_observation_keys=["wam_reference"],
        complete_context=True, align_real_reward=True, align_real_termination=True,
        independent_imagination=True, compact_reference=True, compact_replay=True,
        window=2, horizon=2, replay_steps=8, replay_envs=6, batch_size=4, policy_batch_size=12,
        world_updates=1, critic_epochs=1, critic_minibatches=1, action_noise=0.,
        observation_noise={name: 0. for name in PROPRIO_NAMES}, gradient_diagnostics_interval=1,
        critic_future_frames=2, save_replay=True,
    )
    settings.update(overrides)
    return WAMTrainer(
        args=SimpleNamespace(num_total_batches=1), config=Config(gamma=0.99, lam=0.95),
        env=env, model=PhysicalFSQPolicy(), settings=settings, log_dir=directory,
        world_model_config={
            "_target_": "gear_sonic.trl.modules.wam.StateSpaceWorldModel",
            "cfg": dict(state_type="physical", hidden=[16], learn_reward=False,
                        learn_termination=False, reward_mode="analytic", termination_mode="analytic"),
        },
    )


class N3SyntheticTrainingTests(unittest.TestCase):
    def test_n3_complete_context_state_action_and_actor_projection_use_named_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            env = trainer.env
            self.assertEqual(trainer.robot_type, "n3")
            self.assertEqual(trainer.robot_spec["root_body"], "base_link")
            self.assertEqual(trainer.context_schema["prefix_joint_names"], list(N3_MUJOCO_JOINT_NAMES))
            self.assertIn("waist_yaw_link", trainer.context_schema["body_names"])
            torch.testing.assert_close(trainer.obs["wam_state"], env.state)
            offsets = env.robot.data.default_joint_pos - env.robot.data.default_joint_pos_nominal
            torch.testing.assert_close(trainer.obs["wam_context"][:, 1:30], offsets[:, env.to_native])
            torch.testing.assert_close(trainer.obs["wam_context"][:, 30:33],
                                       env.robot.root_physx_view.coms[:, env.robot.body_names.index("waist_yaw_link"), :3])
            action = torch.arange(29).expand(env.num_envs, -1).float()
            torch.testing.assert_close(trainer._to_wam_action(action), action[:, env.to_native])
            frame = trainer._physical_actor_frame(env.state, env.actions, trainer.obs["wam_context"])
            torch.testing.assert_close(frame, trainer.obs["wam"], atol=1e-7, rtol=1e-6)

    def test_world_fitting_and_fsq_actor_critic_updates_are_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            trainer.collect(4)
            before = {name: value.detach().clone() for name, value in trainer.policy.named_parameters()}
            world = trainer.fit_world_model()
            actor = trainer.improve_policy()
            self.assertEqual(trainer.world_updates, 1)
            self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in {**world, **actor}.values()))
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in trainer.policy.named_parameters()))
            for index, encoder in enumerate(trainer.policy.encoders):
                self.assertTrue(any(
                    not torch.equal(before[f"encoders.{index}.{name}"], value)
                    for name, value in encoder.named_parameters()
                ), f"modality {index} did not receive the synthetic return gradient")
            self.assertTrue(trainer.world_optimizer.state)
            self.assertTrue(trainer.critic_optimizer.state)
            self.assertEqual(trainer.reward_model.kinematics.root_body, "base_link")
            self.assertIn("head_yaw_link", trainer.critic_features.kinematics.tracked_bodies)
            self.assertEqual(trainer.replay._storage["goal"].shape[-1], 3)
            torch.testing.assert_close(trainer.obs["wam_state"], trainer.env.state)

    def test_n3_checkpoint_roundtrip_restores_policy_models_and_reference_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_trainer(Path(directory) / "source")
            source.collect(3)
            source.fit_world_model()
            source.improve_policy()
            checkpoint = Path(directory) / "n3.pt"
            source.save_checkpoint(checkpoint)
            restored = make_trainer(Path(directory) / "restored")
            restored.load_checkpoint(checkpoint, resume=True)
            self.assertEqual(source.robot_spec, restored.robot_spec)
            for before, after in ((source.policy, restored.policy), (source.world_model, restored.world_model),
                                  (source.critic, restored.critic), (source.critic_norm, restored.critic_norm)):
                for name, value in before.state_dict().items():
                    torch.testing.assert_close(after.state_dict()[name], value)
            self.assertEqual(len(restored.replay), len(source.replay))
            self.assertTrue(restored.compact_reference)
            self.assertEqual(set(restored.reference_bank.clips), set(source.reference_bank.clips))
            with torch.no_grad():
                first = source._action(source.obs["wam_state"], source.obs["tokenizer"], actor_obs=source.actor_obs)
                second = restored._action(source.obs["wam_state"], source.obs["tokenizer"], actor_obs=source.actor_obs)
            torch.testing.assert_close(second, first)

    def test_n3_rejects_g1_and_unlabelled_checkpoints_before_policy_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory)
            original = deepcopy(trainer.policy.state_dict())
            for kind in ("g1", "unlabelled"):
                payload = trainer._checkpoint_payload()
                if kind == "g1":
                    payload["wam_robot_spec"] = joint_utils.get_robot_spec("g1")
                else:
                    payload.pop("wam_robot_spec")
                for value in payload["policy_state_dict"].values():
                    value.add_(0.1)
                path = Path(directory) / f"{kind}.pt"
                torch.save(payload, path)
                # state_dict tensors share storage; restore the fixture before loading.
                trainer.policy.load_state_dict(original)
                for resume in (False, True):
                    with self.subTest(kind=kind, resume=resume), self.assertRaisesRegex(ValueError, "N3|robot|embodiment"):
                        trainer.load_checkpoint(path, resume=resume)
                    for name, value in trainer.policy.state_dict().items():
                        torch.testing.assert_close(value, original[name])


if __name__ == "__main__":
    unittest.main()
