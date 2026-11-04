"""Exercise production SMPL VR5 callbacks and WAM reference replay on CPU.

The simulator is not required: only the command base class and Isaac's frame
transform helper are replaced. Tensor rotation operations use the repository's
real wxyz utilities, and observation/command bodies are compiled from source.
"""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.envs.env_utils.n3_utils import N3_ISAACLAB_DOF_NAMES
from gear_sonic.isaac_utils import rotations
from gear_sonic.tests.test_wam_reference_integration import (
    CommandBase, MovingClip, production_command,
)
from gear_sonic.trl.utils import torch_transform
from gear_sonic.utils.smpl_vr import (
    smpl_vr5_local_orientations, smpl_vr5_local_positions,
)
from gear_sonic.utils.wam_observations import WAMGoalObservations
from gear_sonic.utils.wam_reference import WAMReferenceBank


REPOSITORY = Path(__file__).resolve().parents[2]
VR5_TERMS = (
    "smpl_vr_5point_local_target", "smpl_vr_5point_local_orn_target", "smpl_root_ori_b",
)


def _subtract_frames(position, quaternion, target_position, target_quaternion):
    """Minimal frame transform used only by the existing root observation."""
    inverse = torch_transform.quat_inv(quaternion)
    relative_position = None if target_position is None else torch_transform.quat_apply(
        inverse, target_position - position,
    )
    return relative_position, torch_transform.quat_mul(inverse, target_quaternion)


def production_namespace():
    namespace = {
        "__name__": "n3_smpl_vr5_integration_sources",
        "torch": torch, "np": np, "rotations": rotations,
        "torch_transform": torch_transform, "CommandBase": CommandBase,
        "quat_mul": torch_transform.quat_mul,
        "quat_apply": torch_transform.quat_apply,
        "quat_inv": torch_transform.quat_inv,
        "joint_utils": joint_utils,
        "matrix_from_quat": rotations.quaternion_to_matrix,
        "subtract_frame_transforms": _subtract_frames,
        "smpl_vr5_local_positions": smpl_vr5_local_positions,
        "smpl_vr5_local_orientations": smpl_vr5_local_orientations,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)

    def execute(nodes, path):
        tree = ast.Module(body=[deepcopy(future), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)

    path = REPOSITORY / "gear_sonic/envs/manager_env/mdp/commands.py"
    command = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "TrackingCommand")
    command.bases = [ast.Name(id="CommandBase", ctx=ast.Load())]
    execute([command], path)
    namespace["commands"] = SimpleNamespace(TrackingCommand=namespace["TrackingCommand"])
    path = REPOSITORY / "gear_sonic/envs/manager_env/mdp/observations.py"
    execute([node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef)], path)
    return namespace


class SmplClip(MovingClip):
    """Independent robot and human motion expose accidental robot references."""

    smpl_y_up = False

    @staticmethod
    def root_quaternions(frames):
        vectors = torch.zeros(len(frames), 3)
        vectors[:, 2] = frames.float() * 0.03
        return torch_transform.angle_axis_to_quaternion(vectors)

    def get_smpl_pose(self, ids, frames):
        result = torch.zeros(len(frames), 24, 3)
        # Existing SMPL roots remove the fixed rest-pose base rotation.
        base = torch.full((len(frames), 4), 0.5)
        root = torch_transform.quat_mul(self.root_quaternions(frames), base)
        result[:, 0] = torch_transform.quaternion_to_angle_axis(root)
        result[:, 18, 1] = frames.float() * 0.01
        result[:, 20, 2] = frames.float() * 0.02
        result[:, 15, 0] = frames.float() * 0.015
        return result.flatten(1)

    @staticmethod
    def local_joints(frames):
        result = torch.arange(72).float().reshape(1, 24, 3).repeat(len(frames), 1, 1) * 0.01
        result -= result[:, :1].clone()
        result[:, 20, 0] += frames.float() * 0.005
        result[:, 15, 2] += frames.float() * 0.002
        return result

    def get_smpl_joints(self, ids, frames):
        matrices = rotations.quaternion_to_matrix(self.root_quaternions(frames))
        local = self.local_joints(frames)
        # A frame-dependent world translation checks pelvis subtraction too.
        translation = frames[:, None, None].float() * torch.tensor([[[0.1, -0.2, 0.3]]])
        return (matrices[:, None] @ local[..., None]).squeeze(-1) + translation


def command_for(namespace):
    command = production_command(namespace)
    command.motion_lib = SmplClip()
    command.robot.joint_names = list(N3_ISAACLAB_DOF_NAMES)
    return command


def tokenizer(namespace, command):
    with initialize_config_dir(config_dir=str(REPOSITORY / "gear_sonic/config"), version_base="1.1"):
        config = compose(config_name="base", overrides=[
            "+exp=manager/universal_token/all_modes/sonic_n3_smpl_vr5",
        ])
    values, specs, configurations, names, offset = [], [], [], [], 0
    for name, term in config.manager_env.observations.tokenizer.items():
        if not OmegaConf.is_dict(term):
            continue
        function = namespace[term.func.rsplit(":", 1)[-1]]
        parameters = OmegaConf.to_container(term.get("params", {}))
        value = function(command._env, **parameters)
        flat = value.reshape(command.num_envs, -1)
        specs.append((name, offset, offset + flat.shape[-1], tuple(value.shape[1:])))
        offset += flat.shape[-1]
        names.append(name)
        values.append(flat)
        configurations.append(SimpleNamespace(
            func=function, params=parameters, noise=None, clip=None, scale=None, modifiers=None,
        ))
    manager = SimpleNamespace(
        _group_obs_term_names={"tokenizer": names},
        _group_obs_term_cfgs={"tokenizer": configurations},
        _group_obs_cfg={"tokenizer": SimpleNamespace(enable_corruption=False)},
    )
    return specs, manager, torch.cat(values, -1)


class N3SmplVR5IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = production_namespace()

    def test_five_point_callbacks_use_smpl_positions_and_not_robot_reference(self):
        command = command_for(self.namespace)
        position = self.namespace[VR5_TERMS[0]](command._env, "motion")
        orientation = self.namespace[VR5_TERMS[1]](command._env, "motion")
        self.assertEqual(position.shape, (2, 15))
        self.assertEqual(orientation.shape, (2, 20))
        expected = command.motion_lib.local_joints(command.time_steps)[:, [20, 21, 15, 7, 8]].flatten(1)
        torch.testing.assert_close(position, expected, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(orientation.reshape(2, 5, 4).norm(dim=-1), torch.ones(2, 5))
        with patch.object(command.motion_lib, "get_dof_pos", side_effect=AssertionError("robot joints read")), \
                patch.object(command.motion_lib, "get_body_pos_w", side_effect=AssertionError("robot body read")), \
                patch.object(command.motion_lib, "get_body_quat_w", side_effect=AssertionError("robot body read")):
            command.robot.data.body_quat_w = torch_transform.angle_axis_to_quaternion(
                torch.tensor([[0.6, -0.2, 1.0], [-0.4, 0.7, -0.3]])
            )[:, None]
            torch.testing.assert_close(self.namespace[VR5_TERMS[0]](command._env, "motion"), position)
            torch.testing.assert_close(self.namespace[VR5_TERMS[1]](command._env, "motion"), orientation)

    def test_wam_capture_uses_new_layout_and_reconstructs_live_robot_root(self):
        command = command_for(self.namespace)
        specs, manager, live = tokenizer(self.namespace, command)
        widths = {name: end - start for name, start, end, _ in specs}
        for name, width in zip(VR5_TERMS, (15, 20, 6)):
            self.assertEqual(widths[name], width)
        self.assertNotIn("vr_3point_local_target", widths)
        self.assertNotIn("command_multi_future_lower_body", widths)
        bank = WAMReferenceBank(specs)
        snapshots = {key: value.clone() for key, value in vars(command).items() if torch.is_tensor(value)}
        indices = bank.capture(command, manager)
        for key, value in snapshots.items():
            torch.testing.assert_close(getattr(command, key), value)
        timeline = bank.trajectory(indices, live, 3, 10, 5, "cpu")
        state = torch.zeros(2, 71)
        state[:, 3:7] = command.robot_anchor_quat_w
        reconstructed = WAMGoalObservations(specs).to_robot(timeline["goals"][:, 0], state)
        torch.testing.assert_close(reconstructed, live, atol=2e-6, rtol=1e-5)

        for offset in (1, 2, 3):
            command.time_steps = snapshots["time_steps"] + offset
            direct = torch.cat([
                config.func(command._env, **config.params).reshape(2, -1)
                for config in manager._group_obs_term_cfgs["tokenizer"]
            ], -1)
            rebuilt = WAMGoalObservations(specs).to_robot(timeline["goals"][:, offset], state)
            torch.testing.assert_close(rebuilt, direct, atol=2e-6, rtol=1e-5)

    def test_imagined_root_rotation_changes_root_goal_and_keeps_five_points_fixed(self):
        command = command_for(self.namespace)
        specs, manager, live = tokenizer(self.namespace, command)
        bank = WAMReferenceBank(specs)
        indices = bank.capture(command, manager)
        world = bank.trajectory(indices, live, 1, 1, 1, "cpu")["goals"][:, 0]
        state = torch.zeros(2, 71)
        state[:, 3:7] = torch_transform.angle_axis_to_quaternion(
            torch.tensor([[0.4, -0.2, 0.7], [-0.3, 0.5, -0.4]])
        )
        rebuilt = WAMGoalObservations(specs).to_robot(world, state)
        command.robot.data.body_quat_w = state[:, None, 3:7]
        for name, start, end, _ in specs:
            if name == "smpl_root_ori_b":
                expected = self.namespace[name](command._env, "motion")
                torch.testing.assert_close(rebuilt[:, start:end], expected, atol=2e-6, rtol=1e-5)
                self.assertFalse(torch.allclose(world[:, start:end], rebuilt[:, start:end]))
            elif name in VR5_TERMS[:2]:
                torch.testing.assert_close(rebuilt[:, start:end], world[:, start:end], atol=0, rtol=0)

    def test_real_actor_fsq_backpropagates_all_three_configured_input_branches(self):
        from gear_sonic.trl.modules.actor_critic_modules import Actor

        torch.manual_seed(183)
        command = command_for(self.namespace)
        specs, _, live = tokenizer(self.namespace, command)
        dims = {name: list(shape) for name, _, _, shape in specs}
        self.assertEqual(dims["smpl_joints_multi_future_local_nonflat"], [10, 72])
        self.assertEqual(dims["smpl_root_ori_b_multi_future"], [10, 6])
        self.assertEqual(dims["joint_pos_multi_future_wrist_for_smpl"], [10, 6])
        env_config = OmegaConf.create({
            "robot": {"actions_dim": 29, "algo_obs_dim_dict": {"actor_obs": 93}},
            "obs": {"group_obs_dims": {"tokenizer": dims},
                    "group_obs_names": {"tokenizer": list(dims)}},
        })
        with initialize_config_dir(config_dir=str(REPOSITORY / "gear_sonic/config"), version_base="1.1"):
            config = compose(config_name="base", overrides=[
                "+exp=manager/universal_token/all_modes/sonic_n3_smpl_vr5",
            ])
        backbone = config.algo.config.actor.backbone
        # Reduce network width and FSQ alphabet only. Retain the actual branch
        # inputs, temporal layouts, decoder outputs and auxiliary-loss setup.
        backbone.num_fsq_levels = 2
        backbone.fsq_level_list = [4, 4]
        for branch in (*backbone.encoders.values(), *backbone.decoders.values()):
            branch.params.module_config_dict.layer_config.hidden_dims = [16]
        backbone = OmegaConf.create(OmegaConf.to_container(backbone, resolve=True))
        actor = Actor(env_config, OmegaConf.create({"init_noise_std": 0.05}), backbone,
                      input_obs_dict=True, has_aux_loss=True)
        goals = live.repeat(3, 1).clone()
        route_start, route_end = next((start, end) for name, start, end, _ in specs
                                     if name == "encoder_index")
        goals[:, route_start:route_end] = torch.eye(3).repeat_interleave(2, dim=0)
        goals.requires_grad_(True)
        actions = actor({"actor_obs": torch.randn(6, 1, 93), "tokenizer": goals[:, None]},
                        is_training=True, cache_encoded_tensors=False)
        self.assertEqual(actions.shape, (6, 1, 29))
        self.assertTrue(torch.isfinite(actions).all())
        weights = torch.linspace(-1, 1, 29)
        (actions * weights).sum().backward()
        for name in ("g1", "teleop", "smpl"):
            gradients = [parameter.grad for parameter in actor.actor_module.encoders[name].parameters()]
            self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in gradients), name)
            self.assertGreater(sum(value.abs().sum().item() for value in gradients), 0, name)
        for name in VR5_TERMS:
            start, end = next((start, end) for term, start, end, _ in specs if term == name)
            self.assertGreater(goals.grad[2:4, start:end].abs().sum().item(), 0, name)


if __name__ == "__main__":
    unittest.main()
