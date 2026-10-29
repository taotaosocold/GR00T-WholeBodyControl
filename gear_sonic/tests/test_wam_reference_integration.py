"""Run production reference properties and observation terms without Isaac Sim.

Only the simulator base class is replaced. The TrackingCommand properties,
tokenizer callbacks and Isaac Lab quaternion math are compiled from their real
source definitions, so this checks more than a mock tokenizer implementation.
The tests skip when no local Isaac Lab math source is available.
"""

import ast
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from gear_sonic.isaac_utils import rotations
from gear_sonic.trl.utils import torch_transform
from gear_sonic.utils.wam_observations import WAMGoalObservations
from gear_sonic.utils.wam_reference import WAMReferenceBank


REPOSITORY = Path(__file__).resolve().parents[2]


class CommandBase:
    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device


def isaac_math_source():
    roots = [Path(value) for value in sys.path if value]
    roots.extend(REPOSITORY.parent.glob("IsaacLab*"))
    roots.append(REPOSITORY / "IsaacLab")
    for name in ("ISAACLAB_PATH", "ISAACLAB_ROOT"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]))
    for root in roots:
        for suffix in ("isaaclab/utils/math.py", "source/isaaclab/isaaclab/utils/math.py"):
            path = root / suffix
            if path.is_file():
                return path
    raise unittest.SkipTest("Local Isaac Lab math.py source is required for production reference integration")


def load_production_sources():
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "__name__": "wam_reference_integration_sources",
        "torch": torch, "np": np, "math": math,
        "torch_transform": torch_transform, "rotations": rotations, "CommandBase": CommandBase,
    }

    def execute(nodes, path):
        module = ast.Module(body=[deepcopy(future), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)

    math_path = isaac_math_source()
    functions = [node for node in ast.parse(math_path.read_text()).body if isinstance(node, ast.FunctionDef)]
    # JIT compilation needs a fully imported simulator module. Removing only
    # decorators leaves the actual tensor math implementation unchanged.
    for node in functions:
        node.decorator_list = []
    execute(functions, math_path)

    command_path = REPOSITORY / "gear_sonic/envs/manager_env/mdp/commands.py"
    command_class = next(
        node for node in ast.parse(command_path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "TrackingCommand"
    )
    command_class.bases = [ast.Name(id="CommandBase", ctx=ast.Load())]
    execute([command_class], command_path)
    namespace["commands"] = SimpleNamespace(TrackingCommand=namespace["TrackingCommand"])

    observation_path = REPOSITORY / "gear_sonic/envs/manager_env/mdp/observations.py"
    execute([
        node for node in ast.parse(observation_path.read_text()).body if isinstance(node, ast.FunctionDef)
    ], observation_path)
    return namespace


class MovingClip:
    """Distinct frames expose wrong cursors, future grids and selected bodies."""

    smpl_y_up = True

    def get_time_step_total(self, ids):
        return torch.full_like(ids, 30)

    def get_dof_pos(self, ids, frames):
        return frames[:, None].float() * 0.01 + torch.arange(29)[None] * 0.001

    def get_dof_vel(self, ids, frames):
        return frames[:, None].float() * 0.02 + torch.arange(29)[None] * 0.002

    def get_body_pos_w(self, ids, frames):
        result = torch.arange(12).float().reshape(1, 4, 3).repeat(len(ids), 1, 1) * 0.1
        result[:, :, 0] += frames[:, None] * 0.1
        return result

    def get_body_quat_w(self, ids, frames):
        vector = torch.zeros(len(ids), 3)
        vector[:, 2] = frames.float() * 0.01
        return torch_transform.angle_axis_to_quaternion(vector)[:, None].repeat(1, 4, 1)

    def get_body_lin_vel_w(self, ids, frames):
        return torch.ones(len(ids), 4, 3) * 0.1

    def get_body_ang_vel_w(self, ids, frames):
        return torch.zeros(len(ids), 4, 3)

    def get_root_pos_w(self, ids, frames):
        return self.get_body_pos_w(ids, frames)[:, 0]

    def get_root_quat_w(self, ids, frames):
        return self.get_body_quat_w(ids, frames)[:, 0]

    def get_smpl_pose(self, ids, frames):
        result = torch.zeros(len(ids), 72)
        result[:, 1] = frames.float() * 0.001
        return result

    def get_smpl_joints(self, ids, frames):
        return torch.arange(72).float().reshape(1, 24, 3).repeat(len(ids), 1, 1) * 0.01


def production_command(namespace):
    command = namespace["TrackingCommand"].__new__(namespace["TrackingCommand"])
    command._env = SimpleNamespace(
        num_envs=2, device="cpu", scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
    )
    command.motion_lib = MovingClip()
    command.motion_ids = torch.zeros(2, dtype=torch.long)
    command.time_steps = torch.tensor([0, 4])
    command.motion_start_time_steps = torch.zeros(2, dtype=torch.long)
    command.motion_num_steps = torch.full((2,), 30)
    command.num_future_frames = command.smpl_num_future_frames = 10
    command.future_time_steps_init = torch.arange(10)[None].repeat(2, 1) * 5
    command.smpl_future_time_steps_init = torch.arange(10)[None].repeat(2, 1)
    command.future_motion_ids = command.motion_ids.repeat_interleave(10)
    command.smpl_future_motion_ids = command.future_motion_ids.clone()
    command.encoder_index = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    command.vr_3point_body_offsets = torch.zeros(2, 3, 3)
    command.cfg = SimpleNamespace(vr_3point_body=["left", "right", "head"])
    command.vr_3point_body_indices_motion = [1, 2, 3]
    command.lower_joint_isaaclab_indices = list(range(12))
    command.robot_anchor_body_index = command.motion_anchor_body_index = 0
    command._offline = command.use_ref_motion_root_quat_w_as_anchor = False
    command.ref_motion_root_rotation_noise = command.per_env_num_frames = None
    quaternion = torch_transform.angle_axis_to_quaternion(torch.tensor([[0.2, 0., 0.7], [0., -0.3, -0.4]]))
    command.robot = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=torch.zeros(2, 1, 3), body_quat_w=quaternion[:, None],
    ))
    command._env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return command


def tokenizer_layout(namespace, command):
    with initialize_config_dir(config_dir=str(REPOSITORY / "gear_sonic/config"), version_base="1.1"):
        config = compose(config_name="base", overrides=[
            "+exp=manager/universal_token/all_modes/sonic_release", "trainer=wam", "world_model=wam_dynamics",
        ])
    specs, names, configs, values, offset = [], [], [], [], 0
    for name, term in config.manager_env.observations.tokenizer.items():
        if not OmegaConf.is_dict(term):
            continue
        function = namespace[term.func.rsplit(":", 1)[-1]]
        parameters = OmegaConf.to_container(term.get("params", {}))
        value = function(command._env, **parameters)
        flat = value.reshape(command.num_envs, -1)
        configs.append(SimpleNamespace(
            func=function, params=parameters, noise=None, clip=None, scale=None, modifiers=None,
        ))
        specs.append((name, offset, offset + flat.shape[-1], tuple(value.shape[1:])))
        names.append(name)
        offset += flat.shape[-1]
        values.append(flat)
    manager = SimpleNamespace(
        _group_obs_term_names={"tokenizer": names}, _group_obs_term_cfgs={"tokenizer": configs},
    )
    return specs, manager, torch.cat(values, -1)


class ProductionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_production_sources()

    def test_actual_twelve_terms_match_live_callbacks_after_imagined_robot_rotation(self):
        command = production_command(self.namespace)
        specs, manager, live = tokenizer_layout(self.namespace, command)
        self.assertEqual(len(specs), 12)
        bank = WAMReferenceBank(specs)
        indices = bank.capture(command, manager)
        trajectory = bank.trajectory(indices, live, 8, 10, 5, "cpu")
        physical = torch.zeros(2, 71)
        physical[:, 3:7] = command.robot_anchor_quat_w
        imagined_goals = WAMGoalObservations(specs).to_robot(trajectory["goals"][:, 0], physical)
        self.assertEqual(tuple(imagined_goals.shape), (2, 1761))
        for name, start, end, _ in specs:
            with self.subTest(term=name):
                torch.testing.assert_close(imagined_goals[:, start:end], live[:, start:end])
        shapes = {name: shape for name, _, _, shape in specs}
        self.assertEqual(shapes["smpl_joints_multi_future_local_nonflat"], (10, 72))
        self.assertEqual(shapes["smpl_root_ori_b_multi_future"], (10, 6))
        self.assertEqual(shapes["joint_pos_multi_future_wrist_for_smpl"], (10, 6))
        self.assertEqual(shapes["vr_3point_local_target"], (9,))
        self.assertEqual(shapes["vr_3point_local_orn_target"], (12,))

    def test_render_batch_uses_its_own_cursors_grids_and_does_not_mutate_live_command(self):
        command = production_command(self.namespace)
        specs, manager, _ = tokenizer_layout(self.namespace, command)
        snapshots = {name: value.clone() for name, value in vars(command).items() if isinstance(value, torch.Tensor)}
        robot_rotation = command.robot.data.body_quat_w.clone()
        bank = WAMReferenceBank(specs)
        # Capture renders 30 frames at once, despite the live environment's N=2.
        bank.capture(command, manager)
        frames = torch.tensor([1, 4, 28])
        proxy, proxy_env = bank._command_view(command, 0, frames, 30)
        self.assertEqual(proxy.num_envs, 3)
        self.assertEqual(proxy_env.num_envs, 3)
        torch.testing.assert_close(
            proxy.future_time_steps.reshape(3, 10), (frames[:, None] + torch.arange(10) * 5).clamp_max(29),
        )
        torch.testing.assert_close(
            proxy.smpl_future_time_steps.reshape(3, 10), (frames[:, None] + torch.arange(10)).clamp_max(29),
        )
        for name, before in snapshots.items():
            with self.subTest(attribute=name):
                torch.testing.assert_close(getattr(command, name), before)
        torch.testing.assert_close(command.robot.data.body_quat_w, robot_rotation)


if __name__ == "__main__":
    unittest.main()
