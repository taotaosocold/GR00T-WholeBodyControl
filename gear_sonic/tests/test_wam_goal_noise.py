"""Current-frame reference noise is identical in real and imagined observations."""

from copy import deepcopy
import io
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gear_sonic.tests.test_wam_reference import FakeCommand
from gear_sonic.utils.wam_noise import (
    apply_reference_noise,
    current_frame_orientation_noise,
    current_frame_orientation_noise_cfg,
    current_frame_orientation_noise_flat,
    reference_noise_spec,
    supermimic_goal_noise_configs,
    wam_observation_noise,
)
from gear_sonic.utils.wam_reference import WAMReferenceBank


def zeros_observation(env, shape):
    return torch.zeros(env.num_envs, *shape)


def make_bank(terms, *, corrupt=True):
    """Use actual reference-bank capture, with lightweight simulator fixtures."""
    specs, configs, offset = [], [], 0
    replacements = supermimic_goal_noise_configs(name for name, _ in terms)
    for name, shape in terms:
        width = torch.Size(shape).numel()
        specs.append((name, offset, offset + width, shape))
        offset += width
        config = replacements[name]
        noise = None
        if config is not None:
            values = {key: value for key, value in config.items() if key != "_target_"}
            flattened = values.pop("flattened", None)
            if flattened is not None:
                values["func"] = (current_frame_orientation_noise_flat if flattened
                                  else current_frame_orientation_noise)
            noise = SimpleNamespace(**values)
        configs.append(SimpleNamespace(func=zeros_observation, params={"shape": shape},
                                       noise=noise, clip=None, scale=None))
    manager = SimpleNamespace(
        _group_obs_term_names={"tokenizer": [name for name, _ in terms]},
        _group_obs_term_cfgs={"tokenizer": configs},
        _group_obs_cfg={"tokenizer": SimpleNamespace(enable_corruption=corrupt)},
    )
    bank = WAMReferenceBank(specs)
    bank.capture(FakeCommand(), manager)
    return bank, configs


class WAMGoalNoiseTests(unittest.TestCase):
    def test_profile_keeps_proprioception_but_clears_other_goal_noise(self):
        self.assertEqual(wam_observation_noise(), {
            "gravity_dir": 0.05, "base_ang_vel": 0.2, "joint_pos": 0.01,
            "joint_vel": 0.5, "actions": 0.0,
        })
        names = [
            "motion_anchor_ori_b_mf_nonflat", "smpl_root_ori_b_multi_future",
            "motion_anchor_ori_b", "smpl_root_ori_b", "encoder_index",
            "smpl_joints_multi_future_local_nonflat", "vr_3point_local_target",
            "smpl_vr_5point_local_orn_target", "command_multi_future_nonflat",
        ]
        replacements = supermimic_goal_noise_configs(names)
        for name in names[:4]:
            self.assertEqual((replacements[name]["n_min"], replacements[name]["n_max"]),
                             (-0.05, 0.05))
        for name in names[4:]:
            self.assertIsNone(replacements[name])

    def test_nonflat_noise_preserves_future_input_device_dtype_and_gradient(self):
        # Noncontiguous input also exercises observation tensors derived from FK.
        data = torch.zeros(32, 6, 10, dtype=torch.float64).transpose(-1, -2).requires_grad_()
        before = data.detach().clone()
        result = current_frame_orientation_noise(data, SimpleNamespace(n_min=-0.05, n_max=0.05))
        self.assertEqual(result.shape, data.shape)
        self.assertEqual((result.device, result.dtype), (data.device, data.dtype))
        torch.testing.assert_close(data, before, rtol=0, atol=0)
        torch.testing.assert_close(result[:, 1:], before[:, 1:], rtol=0, atol=0)
        self.assertTrue((result[:, 0].abs() <= 0.05).all())
        self.assertTrue((result[:, 0] != 0).any())
        result.sum().backward()
        torch.testing.assert_close(data.grad, torch.ones_like(data), rtol=0, atol=0)

    def test_flat_noise_handles_rollout_and_single_reference_frame(self):
        cfg = SimpleNamespace(n_min=0.025, n_max=0.025)
        result = current_frame_orientation_noise_flat(torch.zeros(3, 4, 30), cfg).reshape(3, 4, 5, 6)
        torch.testing.assert_close(result[..., 0, :], torch.full((3, 4, 6), 0.025))
        torch.testing.assert_close(result[..., 1:, :], torch.zeros(3, 4, 4, 6), rtol=0, atol=0)
        # A one-reference-frame term must corrupt all rollout steps, not only t=0.
        single = current_frame_orientation_noise_flat(torch.zeros(3, 4, 6), cfg)
        torch.testing.assert_close(single, torch.full_like(single, 0.025))

    def test_factory_installs_callable_without_importing_isaac_at_module_load(self):
        fake_noise = ModuleType("isaaclab.utils.noise")
        fake_noise.AdditiveUniformNoiseCfg = SimpleNamespace
        modules = {"isaaclab": ModuleType("isaaclab"),
                   "isaaclab.utils": ModuleType("isaaclab.utils"),
                   "isaaclab.utils.noise": fake_noise}
        with patch.dict(sys.modules, modules):
            for flattened, function in ((False, current_frame_orientation_noise),
                                        (True, current_frame_orientation_noise_flat)):
                cfg = current_frame_orientation_noise_cfg(flattened=flattened)
                self.assertIs(cfg.func, function)
                self.assertEqual(reference_noise_spec(cfg)["flattened"], flattened)

    def test_real_and_imagined_noise_match_for_all_interfaces(self):
        terms = [
            ("encoder_index", (3,)),
            ("motion_anchor_ori_b_mf_nonflat", (4, 6)),
            ("smpl_root_ori_b_multi_future", (3, 6)),
            ("motion_anchor_ori_b", (6,)),
            ("smpl_root_ori_b", (6,)),
            ("smpl_joints_multi_future_local_nonflat", (3, 72)),
            ("smpl_vr_5point_local_target", (15,)),
        ]
        bank, configs = make_bank(terms)
        goals = torch.linspace(-1, 1, 2 * 5 * bank.goal_dim).reshape(2, 5, bank.goal_dim)
        before = goals.clone()
        torch.manual_seed(83)
        expected = []
        for spec, cfg in zip(bank.specs, configs):
            _, start, end, shape = spec
            value = goals[..., start:end].reshape(2, 5, *shape)
            if cfg.noise is not None:
                function = getattr(cfg.noise, "func", None)
                if function is not None:
                    value = function(value, cfg.noise)
                else:
                    low, high = cfg.noise.n_min, cfg.noise.n_max
                    value = value + low + torch.rand_like(value) * (high - low)
            expected.append(value.flatten(2))
        torch.manual_seed(83)
        actual = bank.process_goals(goals)
        torch.testing.assert_close(actual, torch.cat(expected, -1), rtol=0, atol=0)
        torch.testing.assert_close(goals, before, rtol=0, atol=0)
        for name, start, end, shape in bank.specs:
            if name in {"motion_anchor_ori_b_mf_nonflat", "smpl_root_ori_b_multi_future"}:
                result = actual[..., start:end].reshape(2, 5, *shape)
                original = goals[..., start:end].reshape(2, 5, *shape)
                torch.testing.assert_close(result[..., 1:, :], original[..., 1:, :], rtol=0, atol=0)

    def test_flattened_reference_noise_roundtrips_checkpoint_without_isaac(self):
        bank, _ = make_bank([("motion_anchor_ori_b_mf", (24,)),
                             ("smpl_root_ori_b_multi_future_flatten", (6,))])
        stream = io.BytesIO()
        torch.save(bank.state_dict(), stream)
        stream.seek(0)
        state = torch.load(stream, weights_only=True)
        restored = WAMReferenceBank(bank.specs)
        restored.load_state_dict(state)
        goals = torch.zeros(2, 3, bank.goal_dim)
        torch.manual_seed(71)
        expected = bank.process_goals(goals)
        torch.manual_seed(71)
        actual = restored.process_goals(goals)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[..., 6:24], torch.zeros(2, 3, 18), rtol=0, atol=0)
        self.assertTrue((actual[..., 24:] != 0).all())

    def test_disabled_corruption_keeps_clip_and_scale_order(self):
        bank, _ = make_bank([("motion_anchor_ori_b_mf_nonflat", (3, 6))], corrupt=False)
        goals = torch.zeros(2, 4, bank.goal_dim)
        torch.testing.assert_close(bank.process_goals(goals), goals, rtol=0, atol=0)
        # With corruption enabled, only current orientation is noised, then
        # clipping and scaling follow exactly the ObservationManager order.
        bank.transforms = ((0, 18, {"kind": "current_reference_orientation",
                                   "n_min": 0.04, "n_max": 0.04, "flattened": False},
                            (-0.03, 0.03), 2.0),)
        result = bank.process_goals(goals).reshape(2, 4, 3, 6)
        torch.testing.assert_close(result[..., 0, :], torch.full((2, 4, 6), 0.06))
        torch.testing.assert_close(result[..., 1:, :], torch.zeros(2, 4, 2, 6), rtol=0, atol=0)

    def test_legacy_uniform_reference_noise_still_applies_to_every_frame(self):
        spec = reference_noise_spec(SimpleNamespace(n_min=0.02, n_max=0.02))
        self.assertEqual(spec, (0.02, 0.02))
        result = apply_reference_noise(torch.zeros(2, 4, 6), spec)
        torch.testing.assert_close(result, torch.full_like(result, 0.02))

    def test_ppo_keeps_existing_goal_noise_even_if_profile_setting_is_present(self):
        from omegaconf import OmegaConf
        from gear_sonic.utils.config_utils import configure_wam_observations

        config = OmegaConf.create({
            "trainer": {"_target_": "example.PPOTrainer", "settings": {
                "domain_randomization": "supermimic",
            }},
            "world_model": {"cfg": {"state_type": "physical"}},
            "manager_env": {"observations": {"tokenizer": {
                "enable_corruption": True,
                "smpl_joints_multi_future_local_nonflat": {"noise": {"n_min": -0.4, "n_max": 0.4}},
                "smpl_root_ori_b_multi_future": {"noise": {"n_min": -0.2, "n_max": 0.2}},
            }}},
        })
        before = deepcopy(OmegaConf.to_container(config))
        self.assertFalse(configure_wam_observations(config))
        self.assertEqual(OmegaConf.to_container(config), before)

    def test_rejects_unknown_noise_function_and_invalid_shape_or_bounds(self):
        cfg = SimpleNamespace(n_min=-0.05, n_max=0.05, func=lambda data, _: data)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            reference_noise_spec(cfg)
        for cfg in (SimpleNamespace(n_min=0.1, n_max=-0.1),
                    SimpleNamespace(n_min=float("nan"), n_max=0.1),
                    SimpleNamespace(n_min=0, n_max=1, operation="scale")):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                current_frame_orientation_noise(torch.zeros(2, 3, 6), cfg)
        with self.assertRaisesRegex(ValueError, "T\\*6"):
            current_frame_orientation_noise_flat(torch.zeros(2, 7), SimpleNamespace(n_min=0, n_max=1))


if __name__ == "__main__":
    unittest.main()
