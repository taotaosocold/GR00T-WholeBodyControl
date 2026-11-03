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

The current robot loader consumes its existing PKL contract: `fps`,
`root_trans_offset` with shape `[T,3]`, and `pose_aa` in the N3 MJCF body order
with shape `[T,32,3]`. A `[T,30,3]` root-plus-29-joint representation is also
accepted and the two fixed toe identities are inserted by name. Robot joint
coordinates follow the native hinge order recorded in `n3_utils.py`, rather
than assuming the first twelve coordinates are legs. Raw NPZ/qpos data needs
conversion and joint-order validation before use with this loader.
SMPL clips must match the robot motion keys and timeline; the existing paired
frame checks apply. Data conversion and real-data validation remain pending
until the datasets are supplied.

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
