# N3 V3 WAM Training

The `sonic_n3` experiment keeps SONIC's three encoder/FSQ interfaces and trains
them with physical WAM updates. The internal `g1` encoder key means the robot
motion-reference branch; its observations and actions belong to N3 in this
experiment. The other two branches are teleoperation and SMPL.

## Assets and control conventions

- Robot config: `gear_sonic/envs/manager_env/robots/n3.py`.
- Portable metadata: `gear_sonic/envs/env_utils/n3_utils.py`.
- Isaac asset: `gear_sonic/data/assets/robot_description/urdf/n3/n3_v3_isaac.urdf`.
- Motion-library and differentiable FK: `gear_sonic/data/assets/robot_description/mjcf/n3.xml`.
- Experiment: `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_n3.yaml`.

Assets come from the `2026-11-03_n3_current` training snapshot. Kp, Kd, torque
limits, armature and friction follow its `make_mjlab_cfg`; GR00T uses Isaac Lab
implicit actuators, so the physics backend differs. There is no random command
lag or extra joint-velocity clamp. Policy control remains 50 Hz and the action
scale follows GR00T's `0.25 * torque_limit / Kp` convention.

All 29 joints, including head pitch/yaw, remain active. The two fixed toe links
are retained as separate bodies. Fifteen links, including the head, contribute
to tracking rewards. Live joint/body mappings are checked by name; motion/FK
order, expected Isaac order and deployment aliases are explicit metadata.
Do not reuse G1 checkpoints based on tensor dimensions: N3 and G1 have different
joint semantics. N3 checkpoints store and validate their robot/order identity.

## Required motion data

The experiment deliberately sets both dataset paths to `null`. Provide paired
N3-retargeted robot motion and SMPL data; startup fails with a specific message
until both paths exist. No G1 or `n3_0905` dataset is selected automatically.

The robot loader consumes `fps`, `root_trans_offset[T,3]`, and
`pose_aa[T,32,3]`. Named files carry `joint_names` and `body_names`; the loader
normalizes their declared order to the native MJCF layout before any FK or
augmentation. Named 30-body data is also supported and the two fixed toe
identities are inserted by name. Old unnamed PKLs retain their original native
MJCF interpretation. Robot joint coordinates use the native hinge order
recorded in `n3_utils.py`, rather than assuming the first twelve are legs.
SMPL clips must match the robot motion keys and timeline; the existing paired
frame checks apply.

### Convert NPZ motions

The CPU-only converter uses the vendored N3 MJCF for hinge axes and the full
32-body pose layout, including fixed toes. It writes one `{motion_key: entry}`
PKL per clip. The default output stores `dof` and `pose_aa` in actual Isaac/PhysX
BFS order, with names and `motion_order=isaaclab`. Use `--output-order mujoco`
for native storage. The G1 CSV converter, in contrast, takes Isaac-order inputs
and always writes native MJCF order; input and storage order are different
choices. Run from the repository root; no Isaac Lab, CUDA or Torch is required:

```bash
python gear_sonic/data_process/convert_n3_npz_to_motion_lib.py \
  --input data/n3/npz --output data/n3/robot \
  --joint-order n3_0905 --output-order isaaclab
```

The legacy `n3_0905` profile comes from the old Isaac training ONNX's
`joint_names` metadata (SHA256
`838e402e874e7497eb6604d8e6c497e43f8c8dbc58c978ae09e7b132ea70aa80`).
It was verified against the old URDF on 189 NPZ clips / 945 sampled frames:
maximum link-position error `4.37e-6 m`, rotation-matrix error `2.56e-6`.
The GPU14 smoke split's 159 files match that verified source subset by SHA256.
They use actual PhysX BFS order, root body 0 and wxyz quaternions, without
default-angle offsets. Old short names are mapped to the corresponding V3
names; no external joint-name JSON is needed for this profile. Actual sibling
order is measured, rather than inferred from URDF XML declaration order.
The current V3 BFS table was separately checked with a live GPU14 articulation.

Required NPZ fields are `fps`, `joint_pos[T,29]`, `body_pos_w[T,B,3]` and
`body_quat_w[T,B,4]`, with at least three frames. Positions must be metres in a
Z-up world and angles radians. The script preserves frame count, frame rate and
root height; it does not retarget old geometry to V3 or synthesize SMPL data.

For other sources, default `--joint-order auto` reads `joint_names` to identify
columns and `body_names` to identify `base_link`. Legacy files without these
fields require an explicitly confirmed
source order: add `--joint-names-file data/n3/source_joint_names.json` (a JSON
list of the 29 N3 joint names in source column order) and `--root-body-index 0`
if the first source body is the base. Alternatively, `--joint-order isaaclab`
selects the current N3 Isaac order from `n3_utils.py`, and `--joint-order mujoco`
selects its native MJCF order. These flags assert the source order; no order is
inferred merely from filenames. Explicit names conflicting with NPZ
metadata are rejected. Quaternions default to `wxyz`; use `--quat-order xyzw`
only for data exported with that convention.

The known `_from_g1_noetix_n3_0905` filename suffix is removed so PKL stems
match the original SMPL stems. Use `--strip-suffix ''` to retain it or supply a
different suffix. `--smpl-dir` checks same-name file presence; the training
loader still validates paired timelines. Duplicate keys, invalid data and
existing outputs fail before publishing any clips. Writes are atomic per file;
`--overwrite` explicitly allows replacement. Use `--dry-run` to validate without
writing. Original NPZ files are never changed.

From the repository root, activate the existing Isaac Lab training environment
and supply your actual data locations, for example under the untracked `data/`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
N3_ROBOT_MOTIONS=data/n3/robot
N3_SMPL_MOTIONS=data/n3/smpl

python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_n3 \
  +resume=false checkpoint=null auto_load_latest=false \
  num_envs=64 headless=true use_wandb=false \
  experiment_dir=logs_rl/n3_wam_smoke \
  ++algo.config.num_learning_iterations=5 \
  ++trainer.settings.utd=0 \
  ++trainer.settings.collect_steps=32 \
  ++trainer.settings.warmup_steps=32 \
  ++trainer.settings.wm_only_iters=0 \
  ++trainer.settings.world_updates=2 \
  ++trainer.settings.warmup_world_updates=4 \
  ++trainer.settings.batch_size=32 \
  ++trainer.settings.policy_batch_size=32 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$N3_ROBOT_MOTIONS" \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="$N3_SMPL_MOTIONS"
```

This command is a short integration check, not the full training schedule.
For full training, remove the smoke-test overrides and use the existing tmux /
Accelerate workflow. Environment counts and update batch sizes are per rank.
Retain `sonic_n3`'s three nonzero encoder sampling probabilities.

## Evaluation and export

Use `gear_sonic/eval_agent_trl.py` with the N3 checkpoint and its saved
`config.yaml`, as for other SONIC experiments. ONNX export adds explicit N3
robot identity, actual policy order, deployment joint aliases, Kp/Kd, armature,
default angles and action scales to `model_config.yaml`. Apply those named
mappings when integrating with an N3 deployment framework; the Unitree G1 DDS
executable is not an N3 hardware controller.

The STL files follow this repository's Git LFS rules. A Git bundle does not
contain LFS file contents; synchronize the N3 mesh contents when moving the
branch to another machine.

CPU tests cover asset integrity, independent MuJoCo/FK comparisons, ordering,
configuration, physical context, three-encoder gradients and checkpoint
isolation. Actual Isaac Lab import/contact behavior, real motion datasets,
training convergence and hardware deployment still require validation.
