# N3 V3 training assets

Source snapshot: `2026-11-03_n3_current`, training directory only. Deployment
assets were not mixed into this copy. `source_manifest.json` records the source
paths, sizes and SHA256 hashes from that snapshot's manifest, including
`training/n3_v3_asset_definition.py` as the actuator-parameter provenance.

The Isaac variant `n3_v3_isaac.urdf` preserves the source's primitive collision
geometry and fixed toe links. Its mesh references were changed from
`../meshes/` to `meshes/`. `../../mjcf/n3.xml` is the source training MJCF with
only `compiler.meshdir` relocated to `../urdf/n3/meshes`. The 30 referenced STL
files are byte-for-byte source copies.

The robot configuration uses `make_mjlab_cfg`'s Kp/Kd, torque limits, motor
armatures and friction value, through Isaac Lab implicit actuators. It does
not add random command lag or a velocity clamp. Action scale follows GR00T's
`0.25 * effort_limit / stiffness` convention. Import preserves fixed toe
links (`merge_fixed_joints=False`). Expected breadth-first Isaac Lab ordering
is recorded in `n3_utils.py`; an actual Isaac Lab import and physics comparison
must still be checked in the simulator environment.

These are 29-DoF assets, including two actuated head joints. They do not contain
policy weights, motion datasets or a physical head lock.
