"""N3 asset provenance, coordinate order and MJLab actuator parameters on CPU."""

from collections import defaultdict, deque
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from gear_sonic.envs.env_utils import n3_utils as n3


REPOSITORY = Path(__file__).resolve().parents[2]
ASSETS = REPOSITORY / "gear_sonic/data/assets/robot_description"

# 2026-11-03_n3_current/deployment/noetix_robots/n3/config/robot.yaml
# robot.joint_order; kept local so portable export tests need no source package.
SOURCE_DEPLOYMENT_JOINT_ORDER = (
    "waist_yaw_joint", "head_pitch_joint", "head_yaw_joint",
    "l_arm_pitch_joint", "l_arm_roll_joint", "l_arm_yaw_joint", "l_elbow_pitch_joint",
    "l_hand_yaw_joint", "l_hand_pitch_joint", "l_hand_roll_joint",
    "r_arm_pitch_joint", "r_arm_roll_joint", "r_arm_yaw_joint", "r_elbow_pitch_joint",
    "r_hand_yaw_joint", "r_hand_pitch_joint", "r_hand_roll_joint",
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_pitch_joint",
    "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_pitch_joint",
    "r_ankle_pitch_joint", "r_ankle_roll_joint",
)


class N3AssetTests(unittest.TestCase):
    def test_vendored_files_match_the_training_snapshot_except_mesh_paths(self):
        source = json.loads((ASSETS / "urdf/n3/source_manifest.json").read_text())
        self.assertEqual(source["source_snapshot"], "2026-11-03_n3_current")
        self.assertEqual(source["training_definition"]["source"], "training/n3_v3_asset_definition.py")
        self.assertEqual(len(source["files"]), 32)
        for name, entry in source["files"].items():
            with self.subTest(file=name):
                data = (ASSETS / name).read_bytes()
                if name in source["path_rewrites"]:
                    rewrite = source["path_rewrites"][name]
                    self.assertIn(rewrite["to"].encode(), data)
                    data = data.replace(rewrite["to"].encode(), rewrite["from"].encode())
                self.assertTrue(entry["source"].startswith("training/"))
                self.assertEqual(len(data), entry["size"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])

    def test_mesh_paths_resolve_and_collision_primitives_are_retained(self):
        mjcf_path, urdf_path = REPOSITORY / n3.N3_MJCF_PATH, REPOSITORY / n3.N3_URDF_PATH
        mjcf, urdf = ET.parse(mjcf_path).getroot(), ET.parse(urdf_path).getroot()
        mesh_dir = mjcf_path.parent / mjcf.find("compiler").get("meshdir")
        for mesh in mjcf.findall("./asset/mesh"):
            self.assertTrue((mesh_dir / mesh.get("file")).is_file())
        for mesh in urdf.findall(".//mesh"):
            self.assertTrue((urdf_path.parent / mesh.get("filename")).is_file())
        self.assertEqual(len(urdf.findall(".//collision//mesh")), 0)
        self.assertIsNotNone(mjcf.find(".//geom[@name='l_foot7_collision']"))
        self.assertIsNotNone(mjcf.find(".//geom[@name='r_foot7_collision']"))
        for side in ("l", "r"):
            self.assertEqual(urdf.find(f"./joint[@name='{side}_toe_fixed']").get("type"), "fixed")

    def test_native_joint_and_body_metadata_follow_the_mjcf(self):
        model = ET.parse(REPOSITORY / n3.N3_MJCF_PATH).getroot()
        joints = tuple(joint.get("name") for joint in model.findall(".//body/joint")
                       if joint.get("type", "hinge") == "hinge")
        bodies = tuple(body.get("name") for body in model.findall("./worldbody//body"))
        self.assertEqual(joints, n3.N3_MUJOCO_JOINT_NAMES)
        self.assertEqual(bodies, n3.N3_MUJOCO_BODY_NAMES)
        self.assertEqual(len(joints), 29)
        self.assertEqual(len(bodies), 32)
        self.assertEqual(joints[0], "waist_yaw_joint")
        self.assertEqual(joints[-1], "r_leg_ankle_roll_joint")

    def test_training_urdf_and_mjcf_joint_frames_axes_and_ranges_match(self):
        urdf = ET.parse(REPOSITORY / n3.N3_URDF_PATH).getroot()
        mjcf = ET.parse(REPOSITORY / n3.N3_MJCF_PATH).getroot()
        bodies = {body.get("name"): body for body in mjcf.findall("./worldbody//body")}
        joints = {joint.get("name"): joint for joint in mjcf.findall(".//body/joint")}

        def vector(value):
            return tuple(map(float, value.split()))

        for joint in urdf.findall("joint"):
            with self.subTest(joint=joint.get("name")):
                body = bodies[joint.find("child").get("link")]
                origin = joint.find("origin")
                self.assertEqual(vector(origin.get("xyz", "0 0 0")), vector(body.get("pos", "0 0 0")))
                roll, pitch, yaw = vector(origin.get("rpy", "0 0 0"))
                cr, cp, cy = (math.cos(angle / 2) for angle in (roll, pitch, yaw))
                sr, sp, sy = (math.sin(angle / 2) for angle in (roll, pitch, yaw))
                urdf_quat = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                             cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
                mjcf_quat = vector(body.get("quat", "1 0 0 0"))
                norm = math.sqrt(sum(value * value for value in mjcf_quat))
                for actual, expected in zip(mjcf_quat, urdf_quat):
                    self.assertAlmostEqual(actual / norm, expected, places=8)
                if joint.get("type") != "fixed":
                    native = joints[joint.get("name")]
                    self.assertEqual(vector(joint.find("axis").get("xyz")), vector(native.get("axis")))
                    limit = joint.find("limit")
                    self.assertEqual((float(limit.get("lower")), float(limit.get("upper"))),
                                     vector(native.get("range")))

    def test_measured_isaac_order_is_a_full_breadth_first_traversal(self):
        urdf = ET.parse(REPOSITORY / n3.N3_URDF_PATH).getroot()
        children = defaultdict(list)
        roots = {link.get("name") for link in urdf.findall("link")}
        for joint in urdf.findall("joint"):
            child = joint.find("child").get("link")
            children[joint.find("parent").get("link")].append((child, joint))
            roots.discard(child)
        self.assertEqual(roots, {"base_link"})
        queue = deque([("base_link", None)])
        bodies, joints, depths = [], [], {"base_link": 0}
        body_to_joint = {}
        while queue:
            body, joint = queue.popleft()
            bodies.append(body)
            if joint is not None and joint.get("type") != "fixed":
                joints.append(joint.get("name"))
                body_to_joint[body] = joint.get("name")
            for child, _ in children[body]:
                depths[child] = depths[body] + 1
            queue.extend(children[body])
        # PhysX chooses its own sibling order. XML declaration order is not
        # the runtime order; validate BFS levels and the measured first level.
        self.assertEqual(set(joints), set(n3.N3_ISAACLAB_DOF_NAMES))
        self.assertEqual(set(bodies), set(n3.N3_ISAACLAB_BODY_NAMES))
        levels = [depths[name] for name in n3.N3_ISAACLAB_BODY_NAMES]
        self.assertEqual(levels, sorted(levels))
        self.assertEqual(n3.N3_ISAACLAB_BODY_NAMES[:4],
                         ("base_link", "l_leg_hip_pitch_link", "r_leg_hip_pitch_link", "waist_yaw_link"))
        self.assertEqual(tuple(body_to_joint[body] for body in n3.N3_ISAACLAB_BODY_NAMES if body in body_to_joint),
                         n3.N3_ISAACLAB_DOF_NAMES)

    def test_joint_and_body_permutations_are_exact_inverses(self):
        for native, isaac, to_native, to_isaac in (
            (n3.N3_MUJOCO_JOINT_NAMES, n3.N3_ISAACLAB_DOF_NAMES,
             n3.N3_ISAACLAB_TO_MUJOCO_DOF, n3.N3_MUJOCO_TO_ISAACLAB_DOF),
            (n3.N3_MUJOCO_BODY_NAMES, n3.N3_ISAACLAB_BODY_NAMES,
             n3.N3_ISAACLAB_TO_MUJOCO_BODY, n3.N3_MUJOCO_TO_ISAACLAB_BODY),
        ):
            self.assertEqual(len(set(native)), len(native))
            self.assertEqual(tuple(isaac[index] for index in to_native), native)
            self.assertEqual(tuple(native[index] for index in to_isaac), isaac)
            values = tuple(range(len(native)))
            isaac_values = tuple(values[index] for index in to_isaac)
            self.assertEqual(tuple(isaac_values[index] for index in to_native), values)
        self.assertEqual(len(n3.N3_TRACKED_BODIES), 15)
        self.assertIn("head_yaw_link", n3.N3_TRACKED_BODIES)
        self.assertEqual(len(n3.N3_LOWER_BODY_JOINT_NAMES), 12)
        self.assertEqual(len(n3.N3_WRIST_JOINT_NAMES), 6)
        self.assertTrue(set(n3.N3_TRACKED_BODIES) <= set(n3.N3_MUJOCO_BODY_NAMES))
        self.assertTrue(set(n3.N3_WRIST_JOINT_NAMES).isdisjoint(n3.N3_LOWER_BODY_JOINT_NAMES))

    def test_mjlab_motor_parameters_and_gr00t_action_scales(self):
        # Golden group values from the source make_mjlab_cfg, not its Isaac variant.
        expected = {
            "legs": (200., 15., 145., 0.06086, 8),
            "ankle_pitch": (80., 4., 50., 0.0029075, 2),
            "ankle_roll": (50., 2., 50., 0.0029075, 2),
            "waist": (80., 3., 145., 0.06086, 1),
            "arms": (50., 3., 50., 0.0029075, 8),
            "hands": (20., 1., 20., 0.0019182, 6),
            "head": (20., 3., 20., 0.0019182, 2),
        }
        covered = []
        for name, (kp, kd, effort, armature, count) in expected.items():
            group = n3.N3_ACTUATOR_GROUPS[name]
            self.assertEqual(len(group["joint_names"]), count)
            for joint in group["joint_names"]:
                covered.append(joint)
                self.assertEqual(n3.N3_JOINT_KP[joint], kp)
                self.assertEqual(n3.N3_JOINT_KD[joint], kd)
                self.assertEqual(n3.N3_JOINT_EFFORT[joint], effort)
                self.assertEqual(n3.N3_JOINT_ARMATURE[joint], armature)
                self.assertEqual(n3.N3_JOINT_FRICTION[joint], 0.01)
                self.assertEqual(n3.N3_JOINT_ACTION_SCALE[joint], 0.25 * effort / kp)
        self.assertEqual(len(set(covered)), 29)
        self.assertEqual(set(covered), set(n3.N3_MUJOCO_JOINT_NAMES))
        self.assertEqual(n3.N3_JOINT_ACTION_SCALE["l_leg_hip_pitch_joint"], 0.18125)
        self.assertEqual(n3.N3_JOINT_ACTION_SCALE["l_leg_ankle_pitch_joint"], 0.15625)
        self.assertEqual(n3.INIT_POS, (0., 0., 0.724))

    def test_policy_parameters_preserve_any_actual_joint_order_and_deployment_aliases(self):
        import yaml

        native_deployment_names = (
            "waist_yaw_joint",
            "l_arm_pitch_joint", "l_arm_roll_joint", "l_arm_yaw_joint", "l_elbow_pitch_joint",
            "l_hand_yaw_joint", "l_hand_roll_joint", "l_hand_pitch_joint",
            "r_arm_pitch_joint", "r_arm_roll_joint", "r_arm_yaw_joint", "r_elbow_pitch_joint",
            "r_hand_yaw_joint", "r_hand_roll_joint", "r_hand_pitch_joint",
            "head_pitch_joint", "head_yaw_joint",
            "l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint", "l_knee_pitch_joint",
            "l_ankle_pitch_joint", "l_ankle_roll_joint",
            "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint", "r_knee_pitch_joint",
            "r_ankle_pitch_joint", "r_ankle_roll_joint",
        )
        expected_aliases = dict(zip(n3.N3_MUJOCO_JOINT_NAMES, native_deployment_names))
        self.assertEqual(n3.N3_TRAINING_TO_DEPLOYMENT_JOINT_NAMES, expected_aliases)
        self.assertEqual(set(native_deployment_names), set(SOURCE_DEPLOYMENT_JOINT_ORDER))
        self.assertEqual(len(set(SOURCE_DEPLOYMENT_JOINT_ORDER)), 29)
        orders = [list(n3.N3_MUJOCO_JOINT_NAMES), list(n3.N3_ISAACLAB_DOF_NAMES),
                  list(reversed(n3.N3_MUJOCO_JOINT_NAMES))]
        generator = random.Random(20261103)
        for _ in range(10):
            order = list(n3.N3_MUJOCO_JOINT_NAMES)
            generator.shuffle(order)
            orders.append(order)
        columns = {"kp": n3.N3_JOINT_KP, "kd": n3.N3_JOINT_KD, "effort": n3.N3_JOINT_EFFORT,
                   "armature": n3.N3_JOINT_ARMATURE, "action_scale": n3.N3_JOINT_ACTION_SCALE}
        for order in orders:
            with self.subTest(first_joint=order[0]):
                parameters = n3.get_n3_policy_parameters(order, control_dt=0.01)
                self.assertEqual(parameters["joint_names"], order)
                self.assertEqual(parameters["deployment_joint_names"], [expected_aliases[name] for name in order])
                self.assertEqual(set(parameters["deployment_joint_names"]), set(SOURCE_DEPLOYMENT_JOINT_ORDER))
                for column, values in columns.items():
                    self.assertEqual(parameters[column], [values[name] for name in order])
                self.assertEqual(parameters["default_joint_pos"], [n3.INIT_JOINT_POS.get(name, 0.) for name in order])
                self.assertEqual(parameters["control_dt"], 0.01)
                self.assertEqual(parameters["num_actions"], 29)
                self.assertEqual(parameters["root_name"], "base_link")
                self.assertEqual(parameters["action_semantics"], "q_target = q_default + action_scale * a")
                self.assertEqual(yaml.safe_load(yaml.safe_dump(parameters)), parameters)
                self.assertEqual(json.loads(json.dumps(parameters, allow_nan=False)), parameters)
        self.assertEqual(n3.get_n3_policy_parameters(n3.N3_MUJOCO_JOINT_NAMES)["control_dt"], 0.02)

    def test_policy_parameters_reject_incomplete_duplicate_unknown_or_unordered_names(self):
        names = list(n3.N3_MUJOCO_JOINT_NAMES)
        for invalid in (None, "waist_yaw_joint", [], names[:-1], names + [names[0]],
                        [names[0]] * 29, ["l_hand_yaw_joint"] + names[1:],
                        [0] + names[1:], set(names), dict.fromkeys(names)):
            with self.subTest(joint_names=invalid):
                with self.assertRaisesRegex(ValueError, "N3 joint_names"):
                    n3.get_n3_policy_parameters(invalid)

    def test_policy_parameters_reject_invalid_control_dt(self):
        for invalid in (None, True, False, 0., -0.02, math.nan, math.inf, -math.inf, "invalid"):
            with self.subTest(control_dt=invalid):
                with self.assertRaisesRegex(ValueError, "N3 control_dt"):
                    n3.get_n3_policy_parameters(n3.N3_MUJOCO_JOINT_NAMES, control_dt=invalid)

    def test_isaac_configuration_passes_the_same_parameters_without_command_lag(self):
        # Inspect constructed config kwargs without importing or starting Isaac.
        class Cfg(SimpleNamespace):
            pass

        class ArticulationCfg(Cfg):
            InitialStateCfg = Cfg

        class JointDriveCfg(Cfg):
            PDGainsCfg = Cfg

        isaac = ModuleType("isaaclab")
        actuators, assets, sim = (ModuleType(f"isaaclab.{name}") for name in ("actuators", "assets", "sim"))
        actuators.ImplicitActuatorCfg = Cfg
        assets.ArticulationCfg = ArticulationCfg
        for name in ("UrdfFileCfg", "RigidBodyPropertiesCfg", "ArticulationRootPropertiesCfg"):
            setattr(sim, name, Cfg)
        sim.UrdfConverterCfg = SimpleNamespace(JointDriveCfg=JointDriveCfg)
        isaac.actuators, isaac.assets, isaac.sim = actuators, assets, sim
        with patch.dict(sys.modules, {"isaaclab": isaac, "isaaclab.actuators": actuators,
                                      "isaaclab.assets": assets, "isaaclab.sim": sim}):
            namespace = runpy.run_path(str(REPOSITORY / "gear_sonic/envs/manager_env/robots/n3.py"))
        cfg = namespace["N3_CFG"]
        self.assertFalse(cfg.spawn.merge_fixed_joints)
        self.assertFalse(cfg.spawn.fix_base)
        self.assertEqual(set(cfg.init_state.joint_pos), set(n3.N3_MUJOCO_JOINT_NAMES))
        for name, actuator in cfg.actuators.items():
            group = n3.N3_ACTUATOR_GROUPS[name]
            self.assertEqual(tuple(actuator.joint_names_expr), group["joint_names"])
            self.assertEqual(actuator.effort_limit_sim, group["effort"])
            self.assertEqual(actuator.stiffness, group["kp"])
            self.assertEqual(actuator.damping, group["kd"])
            self.assertEqual(actuator.armature, group["armature"])
            self.assertEqual(actuator.friction, 0.01)
            self.assertEqual(actuator.velocity_limit_sim, math.inf)
            self.assertFalse(any("delay" in key for key in vars(actuator)))
        self.assertEqual(namespace["N3_ACTION_SCALE"], n3.N3_JOINT_ACTION_SCALE)

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "requires optional CPU MuJoCo")
    def test_mjcf_compiles_with_source_dimensions_and_native_names(self):
        import mujoco
        import numpy as np

        model = mujoco.MjModel.from_xml_path(str(REPOSITORY / n3.N3_MJCF_PATH))
        self.assertEqual((model.nq, model.nv, model.nbody, model.ngeom, model.nmesh), (36, 35, 33, 61, 30))
        names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
                      for index in range(model.njnt) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE)
        self.assertEqual(names, n3.N3_MUJOCO_JOINT_NAMES)
        data = mujoco.MjData(model)
        data.qpos[:3] = n3.INIT_POS
        for name, position in zip(n3.N3_MUJOCO_JOINT_NAMES, n3.N3_DEFAULT_JOINT_POS):
            joint = model.joint(name)
            self.assertLessEqual(joint.range[0], position)
            self.assertLessEqual(position, joint.range[1])
            data.qpos[joint.qposadr] = position
        mujoco.mj_forward(model, data)
        self.assertTrue(np.isfinite(data.xpos).all())


if __name__ == "__main__":
    unittest.main()
