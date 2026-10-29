"""Complete physics snapshots distinguish dynamics hidden by the old ξ mean."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from gear_sonic.utils.wam_context import (
    get_push_versions,
    invalidate_physical_context,
    physical_context,
    physical_context_schema,
    record_push,
)
from gear_sonic.utils.wam_kinematics import G1_MUJOCO_JOINT_NAMES


class PhysicsView:
    max_shapes = 3
    link_paths = [["/World/envs/env_0/robot/pelvis", "/World/envs/env_0/robot/torso_link"]]

    def __init__(self):
        self.materials = torch.tensor([
            [[0.4, 0.3, 0.0], [0.8, 0.5, 0.1], [1.2, 0.7, 0.2]],
            [[0.8, 0.7, 0.0], [0.8, 0.5, 0.1], [0.8, 0.3, 0.2]],
        ])
        self.masses = torch.tensor([[3.0, 7.0], [3.0, 14.0]])
        self.inertias = torch.eye(3).flatten().expand(2, 2, -1).clone()
        self.inertias[1, 1] *= 2
        self.coms = torch.zeros(2, 2, 7)
        self.coms[..., -1] = 1
        self.coms[1, 1, :3] = torch.tensor([0.01, -0.02, 0.03])

    def get_material_properties(self):
        return self.materials

    def get_masses(self):
        return self.masses

    def get_inertias(self):
        return self.inertias

    def get_coms(self):
        return self.coms


def asset_fixture():
    names = list(reversed(G1_MUJOCO_JOINT_NAMES))
    nominal = torch.arange(29).float() / 20
    offsets = torch.stack((torch.arange(29).float() / 1000, -torch.arange(29).float() / 1000))
    asset = SimpleNamespace(
        joint_names=names,
        body_names=["pelvis", "torso_link"],
        data=SimpleNamespace(
            joint_pos=torch.zeros(2, 29),
            default_joint_pos=nominal + offsets,
            default_joint_pos_nominal=nominal,
        ),
        root_physx_view=PhysicsView(),
        _physics_sim_view=SimpleNamespace(create_rigid_body_view=lambda path: SimpleNamespace(
            max_shapes=1 if path.endswith("/pelvis") else 2,
        )),
    )
    asset._wam_nominal_coms = asset.root_physx_view.coms.clone()
    asset._wam_nominal_coms[..., :3] = 0
    return asset, offsets


class CompleteContextTests(unittest.TestCase):
    def test_equal_mean_materials_remain_distinguishable(self):
        asset, offsets = asset_fixture()
        context = physical_context(asset)
        schema = physical_context_schema(asset)
        material_field = next(field for field in schema["fields"] if field["name"] == "shape_materials")
        self.assertEqual(context.shape, (2, 33 + 3 * 3 + 17 * 2 + 29))
        self.assertAlmostEqual(float(context[0, 0]), float(context[1, 0]), places=6)
        detailed = context[:, material_field["start"]:material_field["end"]]
        self.assertFalse(torch.equal(detailed[0], detailed[1]))
        torch.testing.assert_close(detailed.reshape(2, 3, 3), asset.root_physx_view.materials)
        torch.testing.assert_close(context[:, 1:30], offsets.flip(-1))
        torch.testing.assert_close(context[:, 30:33], asset.root_physx_view.coms[:, 1, :3])
        torch.testing.assert_close(context[:, -29:], offsets)

    def test_mass_inertia_and_com_pose_are_actual_per_body_values(self):
        asset, _ = asset_fixture()
        asset.root_physx_view.coms[0, 1, 3:] = torch.tensor([0.0, 0.0, 0.6, 0.8])
        context = physical_context(asset)
        fields = {field["name"]: field for field in physical_context_schema(asset)["fields"]}
        for field_name, expected in (("body_mass", asset.root_physx_view.masses),
                                     ("body_inertia", asset.root_physx_view.inertias)):
            field = fields[field_name]
            torch.testing.assert_close(context[:, field["start"]:field["end"]], expected.flatten(1))
        field = fields["body_com_pose"]
        com = context[:, field["start"]:field["end"]].reshape(2, 2, 7)
        torch.testing.assert_close(com[..., :3], asset.root_physx_view.coms[..., :3])
        torch.testing.assert_close(com[0, 1, 3:], torch.tensor([0.8, 0.0, 0.0, 0.6]))

    def test_calibration_updates_immediately_and_static_cache_can_be_invalidated(self):
        asset, _ = asset_fixture()
        before = physical_context(asset)
        asset.data.default_joint_pos += 0.02
        asset.root_physx_view.materials[..., 0] += 0.5
        asset.root_physx_view.masses += 1
        cached = physical_context(asset)
        torch.testing.assert_close(cached[:, 1:30], before[:, 1:30] + 0.02)
        torch.testing.assert_close(cached[:, 33:-29], before[:, 33:-29])
        torch.testing.assert_close(cached[:, -29:], before[:, -29:] + 0.02)
        invalidate_physical_context(asset)
        after = physical_context(asset)
        torch.testing.assert_close(after[:, 0], before[:, 0] + 0.5)
        self.assertFalse(torch.equal(after[:, 33:], before[:, 33:]))

    def test_schema_distinguishes_same_size_body_layout_and_is_not_mutable(self):
        asset, _ = asset_fixture()
        schema = physical_context_schema(asset)
        other, _ = asset_fixture()
        other.body_names.reverse()
        changed = physical_context_schema(other)
        self.assertEqual(schema["dim"], changed["dim"])
        self.assertNotEqual(schema, changed)
        schema["body_names"].clear()
        self.assertEqual(physical_context_schema(asset)["body_names"], ["pelvis", "torso_link"])

    def test_incomplete_physics_metadata_fails_instead_of_hiding_randomization(self):
        asset, _ = asset_fixture()
        asset.root_physx_view.get_inertias = None
        with self.assertRaisesRegex(ValueError, "get_inertias"):
            physical_context(asset)
        asset, _ = asset_fixture()
        asset._wam_shapes_per_body = (1, 1)
        with self.assertRaisesRegex(ValueError, "collision-shape layout"):
            physical_context(asset)

    def test_nominal_calibration_required_when_environment_offsets_differ(self):
        asset, _ = asset_fixture()
        del asset.data.default_joint_pos_nominal
        with self.assertRaisesRegex(ValueError, "nominal joint positions"):
            physical_context(asset)

    def test_push_snapshots_are_monotonic_and_do_not_change_physics(self):
        env = SimpleNamespace(scene=SimpleNamespace(num_envs=3), device="cpu")
        old = get_push_versions(env)
        record_push(env, torch.tensor([0, 2]))
        torch.testing.assert_close(old, torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(get_push_versions(env), torch.tensor([1, 0, 1]))
        record_push(env, None)
        torch.testing.assert_close(get_push_versions(env), torch.tensor([2, 1, 2]))
        adapter = SimpleNamespace(scene={})
        torch.testing.assert_close(get_push_versions(adapter, num_envs=2), torch.zeros(2, dtype=torch.long))

    def test_push_wrapper_delegates_original_call_and_only_marks_applied_rows(self):
        path = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/events.py"
        tree = ast.parse(path.read_text())
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                    and item.name == "push_by_setting_velocity")
        calls = []
        scope = {
            "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
            "_push_by_setting_velocity": lambda *args, **kwargs: calls.append((args, kwargs)),
            "record_push": record_push,
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        env = SimpleNamespace(scene=SimpleNamespace(num_envs=3), device="cpu")
        ids, velocities = torch.tensor([1]), {"x": (-0.5, 0.5)}
        scope["push_by_setting_velocity"](env, ids, velocities)
        self.assertIs(calls[0][0][0], env)
        self.assertIs(calls[0][0][1], ids)
        self.assertIs(calls[0][0][2], velocities)
        torch.testing.assert_close(get_push_versions(env), torch.tensor([0, 1, 0]))

    def test_mass_wrapper_preserves_event_arguments_and_refreshes_snapshot(self):
        path = Path(__file__).resolve().parents[1] / "envs/manager_env/mdp/events.py"
        tree = ast.parse(path.read_text())
        block = next(item for item in tree.body if isinstance(item, ast.If)
                     and isinstance(item.test, ast.Call)
                     and any(isinstance(arg, ast.Name) and arg.id == "_randomize_rigid_body_mass"
                             for arg in item.test.args))
        calls = []

        class MassEvent:
            def __call__(self, *args, **kwargs):
                calls.append((args, kwargs))
                args[0].scene["robot"].root_physx_view.masses += 2

        scope = {"_randomize_rigid_body_mass": MassEvent,
                 "invalidate_physical_context": invalidate_physical_context}
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), scope)
        asset, _ = asset_fixture()
        before = physical_context(asset)
        env = SimpleNamespace(scene={"robot": asset})
        ids, cfg, distribution = torch.tensor([0, 1]), SimpleNamespace(name="robot"), (0.8, 2.5)
        scope["randomize_rigid_body_mass"]()(
            env, ids, cfg, distribution, "scale", distribution="uniform", recompute_inertia=False,
        )
        self.assertEqual(calls[0][0], (env, ids, cfg, distribution, "scale"))
        self.assertEqual(calls[0][1], {"distribution": "uniform", "recompute_inertia": False})
        after = physical_context(asset)
        field = next(field for field in physical_context_schema(asset)["fields"] if field["name"] == "body_mass")
        torch.testing.assert_close(after[:, field["start"]:field["end"]],
                                   before[:, field["start"]:field["end"]] + 2)


if __name__ == "__main__":
    unittest.main()
