"""Independent reference timelines, without requiring the simulator runtime."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from gear_sonic.utils.wam_reference import WAMReferenceBank


SPECS = (("encoder_index", 0, 3, (3,)), ("command_z", 3, 4, (1,)))


class FakeMotionLib:
    def __init__(self):
        self.offset = 0.0

    def get_time_step_total(self, ids):
        return torch.full_like(ids, 6)

    def get_body_pos_w(self, ids, frames):
        x = frames.float() + 10 * ids.float() + self.offset
        return torch.stack((x, x * 0, x * 0 + 0.8), -1)[:, None]

    def get_body_quat_w(self, ids, frames):
        q = torch.zeros(len(ids), 1, 4)
        q[..., 0] = 1
        return q

    def get_body_lin_vel_w(self, ids, frames):
        return torch.zeros(len(ids), 1, 3)

    get_body_ang_vel_w = get_body_lin_vel_w

    def get_dof_pos(self, ids, frames):
        return frames[:, None].float().expand(-1, 29) * 0.01

    get_dof_vel = get_dof_pos


class FakeCommand:
    def __init__(self):
        self._env = SimpleNamespace(num_envs=2, device="cpu")
        self.motion_lib = FakeMotionLib()
        self.motion_ids = torch.tensor([0, 1])
        self.motion_start_time_steps = torch.tensor([1, 0])
        self.time_steps = torch.tensor([0, 1])
        self.motion_anchor_body_index = self.robot_anchor_body_index = 0
        self.num_future_frames = self.smpl_num_future_frames = 2
        self.future_time_steps_init = self.smpl_future_time_steps_init = torch.tensor([[0, 2], [0, 2]])
        self.vr_3point_body_offsets = torch.zeros(2, 3, 3)
        self.encoder_index = torch.tensor([[1, 0, 0], [0, 0, 1]])

    @property
    def num_envs(self):
        return self._env.num_envs

    @property
    def device(self):
        return self._env.device


def routing(env, command_name):
    return env.command_manager.get_term(command_name).encoder_index


def goal_height(env, command_name):
    command = env.command_manager.get_term(command_name)
    # A deliberate frame-dependent value proves the callback sees new cursors.
    return command.motion_lib.get_body_pos_w(command.motion_ids, command.time_steps)[:, 0, :1]


def manager():
    return SimpleNamespace(
        _group_obs_term_names={"tokenizer": [s[0] for s in SPECS]},
        _group_obs_term_cfgs={"tokenizer": [
            SimpleNamespace(func=routing, params={"command_name": "motion"}, noise=None),
            SimpleNamespace(func=goal_height, params={"command_name": "motion"},
                            noise=SimpleNamespace(n_min=0.1, n_max=0.1, operation="add"),
                            clip=(-1, 2), scale=3),
        ]},
        _group_obs_cfg={"tokenizer": SimpleNamespace(enable_corruption=True)},
    )


class WAMReferenceTests(unittest.TestCase):
    def test_capture_reuses_terms_without_advancing_live_command(self):
        command = FakeCommand()
        bank = WAMReferenceBank(SPECS)
        original = (command.motion_ids.clone(), command.time_steps.clone(), command.motion_start_time_steps.clone())
        index = bank.capture(command, manager())
        for value, expected in zip((command.motion_ids, command.time_steps, command.motion_start_time_steps), original):
            torch.testing.assert_close(value, expected)
        torch.testing.assert_close(index[:, 1], torch.tensor([1, 1]))
        result = bank.trajectory(index, command.encoder_index.float().new_tensor([[0, 1, 0, 0], [1, 0, 0, 0]]),
                                 3, 2, 2, "cpu")
        torch.testing.assert_close(result["references"][0, :, 0], torch.tensor([1., 2., 3., 4.]))
        torch.testing.assert_close(result["goals"][0, :, 3], torch.tensor([1., 2., 3., 4.]))
        torch.testing.assert_close(result["goals"][0, :, :3], torch.tensor([[0., 1., 0.]]).expand(4, -1))
        transformed = bank.process_goals(result["goals"])
        torch.testing.assert_close(transformed[0, :, 3], torch.tensor([3.3, 6., 6., 6.]))

    def test_clip_boundary_and_future_window_never_read_next_motion(self):
        bank = WAMReferenceBank(SPECS)
        command = FakeCommand()
        index = bank.capture(command, manager())
        index[0, 1] = 4
        result = bank.trajectory(index[:1], torch.tensor([[1., 0., 0., 0.]]), 4, 3, 2, "cpu")
        self.assertEqual(result["mask"].tolist(), [[True, False, False, False]])
        torch.testing.assert_close(result["references"][0, :, 0], torch.tensor([4., 5., 5., 5., 5.]))
        future = result["future_references"].reshape(1, 5, 3, 71)
        torch.testing.assert_close(future[0, 0, :, 0], torch.tensor([4., 5., 5.]))

    def test_old_references_survive_reload_and_replay_checkpoint(self):
        command, bank = FakeCommand(), WAMReferenceBank(SPECS)
        old_index = bank.capture(command, manager())
        old_states = bank.clips[int(old_index[0, 0])][0].clone()
        bank.begin_generation(old_index)
        command.motion_lib.offset = 1000
        new_index = bank.capture(command, manager())
        self.assertNotEqual(old_index[0, 0], new_index[0, 0])
        torch.testing.assert_close(bank.clips[int(old_index[0, 0])][0], old_states)
        restored = WAMReferenceBank(SPECS)
        restored.load_state_dict(deepcopy(bank.state_dict()))
        result = restored.trajectory(old_index, torch.zeros(2, 4), 2, 1, 1, "cpu")
        self.assertLess(result["references"].abs().max(), 100)
        restored.begin_generation(new_index)
        self.assertNotIn(int(old_index[0, 0]), restored.clips)

    def test_disabled_corruption_keeps_clip_and_scale(self):
        command, bank, obs_manager = FakeCommand(), WAMReferenceBank(SPECS), manager()
        obs_manager._group_obs_cfg["tokenizer"].enable_corruption = False
        bank.capture(command, obs_manager)
        result = bank.process_goals(torch.tensor([[1., 0., 0., 1.]]))
        self.assertEqual(result[0, 3], 3)


if __name__ == "__main__":
    unittest.main()
