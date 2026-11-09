"""Physical WAM resets match SuperMimic without changing SONIC/PPO resets."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gear_sonic.envs.env_utils import joint_utils
from gear_sonic.envs.env_utils.n3_utils import N3_MUJOCO_JOINT_NAMES
from gear_sonic.trl.modules.wam import _quat_multiply, _quat_to_mat, _rotvec_to_quat
from gear_sonic.utils.wam_reset import apply_supermimic_reset


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def uniform(low, high, shape, device):
    low, high = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    return low + (high - low) * torch.rand(shape, device=device)


def euler_quat(roll, pitch, yaw):
    # Kept as an independent legacy-path fixture. Exp([r,p,y]) is deliberately
    # not this Euler product when more than one component is nonzero.
    vectors = [torch.zeros(len(roll), 3) for _ in range(3)]
    vectors[0][:, 0], vectors[1][:, 1], vectors[2][:, 2] = roll, pitch, yaw
    x, y, z = [_rotvec_to_quat(value) for value in vectors]
    return _quat_multiply(z, _quat_multiply(y, x))


def reset_excerpt():
    """Execute the real simulator-write portion without importing Isaac Lab."""
    source = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/commands.py"
    tree = ast.parse(source.read_text())
    command = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrackingCommand")
    method = next(node for node in command.body if isinstance(node, ast.FunctionDef) and node.name == "_resample_command")
    first = next(i for i, node in enumerate(method.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "root_pos" for t in node.targets))
    last = next(i for i, node in enumerate(method.body[first:], first)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "write_root_state_to_sim")
    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg="self"), ast.arg(arg="env_ids")],
                         kwonlyargs=[], kw_defaults=[], defaults=[])
    function = ast.FunctionDef(name="run_reset", args=args, body=deepcopy(method.body[first:last + 1]), decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = dict(torch=torch, joint_utils=joint_utils, apply_supermimic_reset=apply_supermimic_reset,
                 sample_uniform=uniform, quat_mul=_quat_multiply, quat_from_euler_xyz=euler_quat)
    exec(compile(module, str(source), "exec"), scope)
    return scope["run_reset"]


class Robot:
    device = "cpu"

    def __init__(self, names):
        self.joint_names = names
        hard = torch.tensor([-1.0, 1.0]).expand(4, len(names), 2).clone()
        soft = hard * 0.5
        self.data = SimpleNamespace(joint_pos_limits=hard, soft_joint_pos_limits=soft)

    def write_joint_state_to_sim(self, positions, velocities, *, env_ids):
        self.written_q, self.written_qd = positions.clone(), velocities.clone()
        self.written_ids = torch.as_tensor(env_ids).clone()

    def write_root_state_to_sim(self, state, *, env_ids):
        self.written_root = state.clone()


def command_fixture(kind="g1", profile="supermimic", *, evaluating=False, q=0.0):
    canonical = joint_utils.G1_MUJOCO_JOINT_NAMES if kind == "g1" else N3_MUJOCO_JOINT_NAMES
    # Interleaving extra fingers catches assumptions that the body occupies
    # indices 0..28 or that every simulator joint belongs in physical state.
    names = list(reversed(canonical))
    for index in range(3):
        names.insert(index * 9 + 2, f"extra_finger_{index}")
    robot = Robot(names)
    body = [names.index(name) for name in canonical]
    extra = [i for i, name in enumerate(names) if name.startswith("extra_")]
    position = torch.tensor([0.2, -0.1, 0.7]).expand(4, 1, 3).clone()
    quaternion = _rotvec_to_quat(torch.tensor([0.31, -0.22, 0.17])).expand(4, 1, 4).clone()
    pose = dict(zip(("x", "y", "z", "roll", "pitch", "yaw"),
                    ((.02, .02), (-.03, -.03), (.005, .005), (.1, .1), (-.1, -.1), (.2, .2))))
    velocity = dict(zip(("x", "y", "z", "roll", "pitch", "yaw"),
                        ((.4, .4), (-.3, -.3), (.1, .1), (.2, .2), (-.4, -.4), (.6, .6))))
    return SimpleNamespace(
        cfg=SimpleNamespace(reset_distribution=profile, pose_range=pose, velocity_range=velocity,
                            joint_position_range=(.09, .09), joint_velocity_range=(1., 1.)),
        body_pos_w=position, body_quat_w=quaternion,
        body_lin_vel_w=torch.zeros(4, 1, 3), body_ang_vel_w=torch.zeros(4, 1, 3),
        anchor_pos_w=position[:, 0], running_ref_root_height=torch.zeros(4),
        joint_pos=torch.full((4, 29), q), joint_vel=torch.full((4, 29), .25),
        robot=robot, has_dof_mismatch=True, num_envs=4, robot_num_dof=len(names), device="cpu",
        body_joint_indices=body, extra_joint_indices=extra,
        extra_default_positions=torch.full((len(extra),), .2), extra_default_velocities=torch.zeros(len(extra)),
        is_evaluating=evaluating,
    )


class WAMResetTests(unittest.TestCase):
    def test_g1_and_n3_real_reset_uses_world_rotvec_and_named_body_joints(self):
        run = reset_excerpt()
        ids = torch.tensor([1, 3])
        for kind in ("g1", "n3"):
            with self.subTest(kind=kind):
                command = command_fixture(kind)
                before_pos, before_quat = command.body_pos_w.clone(), command.body_quat_w.clone()
                run(command, ids)
                robot = command.robot
                delta = torch.tensor([.1, -.1, .2]).expand(len(ids), -1)
                expected_R = torch.linalg.matrix_exp(skew(delta)) @ _quat_to_mat(before_quat[ids, 0])
                torch.testing.assert_close(_quat_to_mat(robot.written_root[:, 3:7]), expected_R)
                torch.testing.assert_close(robot.written_root[:, :3], before_pos[ids, 0] + torch.tensor([.02, -.03, .005]))
                torch.testing.assert_close(robot.written_root[:, 7:], torch.tensor([.4, -.3, .1, .2, -.4, .6]).expand(2, -1))
                torch.testing.assert_close(robot.written_q[:, command.body_joint_indices], torch.full((2, 29), .09))
                torch.testing.assert_close(robot.written_q[:, command.extra_joint_indices], torch.full((2, 3), .2))
                # Even a nonzero legacy qdot range must not randomize this profile.
                torch.testing.assert_close(robot.written_qd[:, command.body_joint_indices], torch.full((2, 29), .25))
                torch.testing.assert_close(robot.written_qd[:, command.extra_joint_indices], torch.zeros(2, 3))
                torch.testing.assert_close(command.body_pos_w, before_pos)
                torch.testing.assert_close(command.body_quat_w, before_quat)
                torch.testing.assert_close(robot.written_ids, ids)

    def test_evaluation_has_no_random_draws_and_keeps_states_inside_hard_limits(self):
        run = reset_excerpt()
        for kind in ("g1", "n3"):
            command = command_fixture(kind, evaluating=True, q=.8)
            with self.subTest(kind=kind), patch("torch.rand", side_effect=AssertionError("evaluation must not sample noise")):
                run(command, torch.tensor([0, 2]))
            torch.testing.assert_close(command.robot.written_q[:, command.body_joint_indices], torch.full((2, 29), .8))
            torch.testing.assert_close(command.robot.written_root[:, :3], command.body_pos_w[[0, 2], 0])
            torch.testing.assert_close(command.robot.written_root[:, 3:7], command.body_quat_w[[0, 2], 0])

    def test_training_clips_to_hard_limits_instead_of_soft_limits(self):
        command = command_fixture(q=.98)
        reset_excerpt()(command, torch.tensor([0]))
        torch.testing.assert_close(command.robot.written_q[:, command.body_joint_indices], torch.ones(1, 29))

    def test_legacy_profile_preserves_euler_noise_extra_joint_noise_and_soft_limits(self):
        command = command_fixture(profile="sonic")
        reset_excerpt()(command, torch.tensor([0, 2]))
        robot = command.robot
        delta = euler_quat(torch.full((2,), .1), torch.full((2,), -.1), torch.full((2,), .2))
        expected = _quat_multiply(delta, command.body_quat_w[[0, 2], 0])
        torch.testing.assert_close(robot.written_root[:, 3:7], expected)
        torch.testing.assert_close(robot.written_q[:, command.extra_joint_indices], torch.full((2, 3), .29))
        torch.testing.assert_close(robot.written_qd[:, command.body_joint_indices], torch.full((2, 29), 1.25))
        command = command_fixture(profile="sonic", evaluating=True, q=.8)
        reset_excerpt()(command, torch.tensor([1]))
        torch.testing.assert_close(command.robot.written_q[:, command.body_joint_indices], torch.full((1, 29), .5))

    def test_zero_rotvec_is_finite_and_partial_reset_preserves_other_rows(self):
        count = 4
        position = torch.zeros(count, 3, dtype=torch.float64)
        quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(count, -1).clone()
        velocity, omega = position.clone(), position.clone()
        joints = torch.zeros(count, 29, dtype=torch.float64)
        apply_supermimic_reset(position, quaternion, velocity, omega, joints, [2], range(29),
                              torch.tensor([-1., 1.], dtype=torch.float64).expand(29, 2),
                              pose_range={}, velocity_range={}, joint_position_range=(.05, .05))
        self.assertTrue(torch.isfinite(quaternion).all())
        torch.testing.assert_close(quaternion, torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(count, -1))
        torch.testing.assert_close(joints[[0, 1, 3]], torch.zeros(3, 29, dtype=torch.float64))
        torch.testing.assert_close(joints[2], torch.full((29,), .05, dtype=torch.float64))

    def test_config_default_remains_sonic_and_unknown_profile_fails(self):
        path = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/commands.py"
        tree = ast.parse(path.read_text())
        config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrackingCommandCfg")
        field = next(n for n in config.body if isinstance(n, ast.AnnAssign)
                     and isinstance(n.target, ast.Name) and n.target.id == "reset_distribution")
        self.assertEqual(ast.literal_eval(field.value), "sonic")
        command = command_fixture(profile="typo")
        with self.assertRaisesRegex(ValueError, "reset_distribution"):
            reset_excerpt()(command, torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
