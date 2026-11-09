# Model-based FSQ + WAM training

The opt-in `WAMTrainer` keeps the original `Actor -> UniversalTokenModule ->
FSQ -> decoder` policy. It replaces PPO updates with a SuperMimic-inspired
Dyna loop: collect simulator transitions, fit differentiable dynamics on
8-step replay windows, improve the real FSQ actor through 32 imagined steps,
and fit a critic with GAE/lambda returns and a terminal bootstrap. World-model
and critic weights are frozen during actor backpropagation; gradients still
flow through their inputs. Existing reconstruction and modality-alignment
losses are applied at the imagined rollout seed. The summed discounted return
and weighted auxiliary losses jointly update the encoders and shared decoder.
`actor_return_reduction: sum` follows SuperMimic; `mean` explicitly divides the
return by the configured horizon. Every `gradient_diagnostics_interval`
iterations, logs report return and auxiliary gradient norms for the actor,
encoders and decoders separately. Diagnostics do not update WAM or critic.

## Run

Use an Isaac Lab training environment with the existing training dependencies
and motion data. From the repository root:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  trainer=wam world_model=wam_dynamics \
  num_envs=256 headless=True \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered
```

The WAM path has four separate observation groups. `wam_state` is the
SuperMimic physical state `p(3), quat(wxyz)(4), v(3), w(3), q(29), qd(29)`;
`wam` is the single-step Sonic proprioceptive frame used to reconstruct the
actor input; `wam_reference` is the corresponding reference physical
state; and `wam_context` contains SuperMimic's 33 calibration values.
Settings live
in `gear_sonic/config/trainer/wam.yaml`; the dynamics
architecture lives in `gear_sonic/config/world_model/wam_dynamics.yaml` and
the critic architecture in `gear_sonic/config/world_model/wam_critic.yaml`.
This path supports single-GPU
and data-parallel multi-GPU training with direct joint actions and an FSQ MLP
actor. Actor-side RunningMeanStd/BatchNorm, temporal attention, and
hierarchical action transforms are rejected explicitly. The actor's normal
`actor_obs` history is retained separately and is never fed to WAM as a
dynamics state.

For eight GPUs, keep the same training arguments and launch with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch \
  --multi_gpu --num_machines=1 --num_processes=8 --machine_rank=0 \
  --main_process_ip=127.0.0.1 --main_process_port=29519 \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  trainer=wam world_model=wam_dynamics \
  num_envs=64 headless=True \
  experiment_dir=logs_rl/wam_eight_gpu \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=sample_data/robot_filtered \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=sample_data/smpl_filtered
```

`num_envs`, replay capacity, and `trainer.settings.batch_size` are **per rank**:
64 environments on eight ranks mean 512 simulator environments in total.
`trainer.settings.policy_batch_size` is also per rank. To match SuperMimic's
global batches on eight ranks, use world-model batch 256 per rank (2048 total)
and policy batch 512 per rank (4096 total). Each
rank collects its own trajectories and samples its own replay. Actor, dynamics,
and critic gradients are averaged after each backward pass; initial weights,
world-model statistics, and valid-state critic normalization statistics are
synchronized. Rank-specific unused
modality encoders participate safely. A non-finite loss/gradient on any rank
rejects the update on all ranks. Adaptive motion sampling remains local.

For a five-iteration smoke test that includes actor updates, override
`trainer.settings.utd=0`, `trainer.settings.collect_steps=32`,
`trainer.settings.wm_only_iters=0`, `trainer.settings.warmup_steps=32`,
`trainer.settings.world_updates=2`, `trainer.settings.warmup_world_updates=4`,
and both `trainer.settings.batch_size=32` and `trainer.settings.policy_batch_size=32`.
The default UTD schedule takes precedence over `collect_steps`; the default
500 world-model-only iterations also prevent a five-iteration test from
reaching actor/critic updates. Check for finite `actor/loss` and `critic/loss`
alongside `world/loss`, rather than treating checkpoint creation as coverage.

### N3 MimicLite termination profile

The named experiment
`+exp=manager/universal_token/all_modes/sonic_n3_smpl_vr5_mimiclite` selects the N3
SMPL/VR5 WAM configuration while retaining the SuperMimic physical
randomization, 33-D context, and WAM reward. Its real termination group follows
MimicLite's Adapt/Finetune semantics: the mapped N3 root (`waist_yaw_link`)
must exceed 0.4 m position error for 25 consecutive control steps to produce a
timeout; root orientation uses a 1.2 rad global error for 25 steps, and the
maximum tracked-body local position/orientation errors use 0.4 m / 1.2 rad for
5 steps as failures. A counter resets immediately when the corresponding
maximum error falls below its threshold. Imagined WAM rollouts use a
differentiable run-length surrogate with the same thresholds and counts, while
root-position timeout hazards stay out of the failure mask so the critic
bootstrap path remains available. The motion-clip timeout remains separate.

This profile enables `finite_guard`: a rank that observes a non-finite WAM
state, action, reward, or observation resets its simulator rank and discards
the current transition before replay or running-normalizer updates. This is a
conservative rank-wide reset because Isaac Lab does not expose a stable
selective reset API across the supported versions. Use the profile with:

```bash
+exp=manager/universal_token/all_modes/sonic_n3_smpl_vr5_mimiclite \
trainer=wam world_model=wam_dynamics
```

The equivalent G1 configuration is
`+exp=manager/universal_token/all_modes/sonic_mimiclite`. It inherits the
G1 three-modality SONIC data and maps the MimicLite root to `torso_link`; the
N3 profile above maps it to `waist_yaw_link`.

The extra WAM observation groups are enabled only for WAM. PPO removes these
four reserved groups from older saved configurations as well, avoiding unused
rollout storage and simulator-property reads. PPO's policy, reward, termination,
and optimizer configuration is unchanged. Omitting
`trainer=wam world_model=wam_dynamics` keeps the existing PPO update path.

New physical WAM runs default to `trainer.settings.align_real_termination=true`.
Before saving `config.yaml` and constructing the environment, the launcher
replaces the Sonic termination group with `terminations/tracking/wam.yaml`.
Real failures use the same height and tilt errors as the imagined soft rule:
`abs(z_ref - z) > 0.25` or `abs(R_ref[2,2] - R[2,2]) > 0.8`. The imagined
version applies sigmoid widths 0.02 and 0.04 at those same thresholds. Change
`trainer.settings.termination_height` / `termination_tilt` to update both forms.
Strict foot, hand, full-orientation, and adaptive-height conditions are absent
from this WAM-specific group. Clip completion and the configured episode time
limit (10 seconds by default) are truncations, kept separate from failures.

Isaac Lab checks failure before advancing the command, so the real rule reads
the next reference frame explicitly, clamping at the clip end. Collection also
captures that reference before an auto-reset can replace it with a new clip;
imagination holds the final valid reference beyond replay episode boundaries.
Checkpoint evaluation reuses this selection without changing the evaluator.
Old saved configs without `align_real_termination` retain their original rules;
to opt an old WAM evaluation into the new rules, pass
`++trainer.settings.align_real_termination=true`. This changes its evaluation
failure criterion relative to the historical run.

Physical WAM runs also default to `trainer.settings.align_real_reward=true`.
The launcher replaces the whole reward group with `rewards/tracking/wam.yaml`.
The real RewardManager and imagined BPTT share
`gear_sonic.utils.wam_reward.WAMTrackingReward`, with parameters from
`trainer.reward_config` (`world_model/wam_reward.yaml`). The terms are four
body tracking kernels (position, orientation, linear and angular velocity),
two root tracking kernels, summed action-rate penalty, and summed per-joint
soft-limit penalty. Body targets are re-anchored to the robot's XY and yaw;
root targets and velocity comparisons remain in their specified world frame.
The 14 tracked bodies and 0.9 soft joint limits come from the repository's G1
MJCF. No learned reward head or external SuperMimic checkout is needed.

Real reward functions return unweighted rates, because Isaac Lab applies the
term weights and control dt. Imagination applies the same weights and actual
environment `step_dt` once. The real path caches the FK results for all eight
terms in each physics step. Both paths use the next reference frame, and
reference snapshots are captured before reset. Adjust weights and kernel
widths through `trainer.reward_config.weights` and `.stds` so they stay shared.
PPO retains the Sonic reward group. Old checkpoint configs without
`align_real_reward` preserve their historical behavior; explicitly enabling
`++trainer.settings.align_real_reward=true` changes the reward definition.

Physical state velocity now refers to the pelvis link origin, matching the
reference motion and FK. The adapter prefers `body_link_lin_vel_w`; on older
backends it converts CoM velocity using angular velocity and the rotated CoM
offset. Missing conversion metadata raises an error. Rotation integration uses
quaternion multiplication, and the model's rotation error has finite small-angle
and half-turn branches. Reward scales also
change relative to historical Sonic-reward runs, so compare new runs under a
fixed validation criterion.

`checkpoint=/path/to/last.pt` loads a compatible
FSQ policy; adding `+resume=true` restores WAM, critic, optimizers, and counters
from a WAM checkpoint. Policy weights remain under `policy_state_dict`, so the
existing policy evaluation loader can read them. The trainer uses its own
checkpoint and logging loop rather than PPO-specific callbacks.

Long runs refresh the loaded motion subset every 250 iterations through
`trainer.settings.motion_resample_frequency`. Set it to zero to disable
refreshing. The trainer seals the old replay segment and obtains fresh
observations after a reload, preventing transitions across different resets.
Replay retains its already logged reference timelines from older subsets.

Only rank zero writes the shared `last.pt`. Multi-process saves additionally
write generation-specific rank sidecars beside it for independent RNG states
and optional replay. Keep the referenced sidecars when copying a checkpoint
for a full resume. A full resume requires the same process count. To change
GPU count, use `resume=false` to load the policy and start fresh dynamics/replay.
Resume preserves rank-local replay boundaries but does not restore simulator
physics or promise bitwise continuation. Logs report averages across ranks.
The physical schema is now version 4: policy imagination uses immutable clip
references independently of real episode boundaries, and complete context
includes detailed simulator properties. Full resume from older physical schemas is rejected; use
`+resume=false checkpoint=/path/to/old/last.pt` to load only its FSQ policy and
refit the world model and critic. Full resume also checks the critic future
frame count and interval, physical-context layout and numerical settings;
matching network dimensions alone are insufficient.

Fitting now defaults to `world_model.cfg.error_frame=target`, matching the
source SuperMimic implementation: predicted-vs-real errors use the real next
state's heading/rotation. Current-to-next dynamics increments and their
normalizer still use the current state's frame. Older physical checkpoints
without this setting used `predicted`; full resume requires explicitly setting
`world_model.cfg.error_frame=predicted`. Policy-only evaluation/export is
unaffected. `world_model.cfg.loss_reduction=mean_valid` preserves the previous
valid-transition average; `sum_per_batch` reproduces the source window sum.
Changing either numerical setting requires a fresh model fit rather than a
silent full resume. Checkpoints, including policy-only evaluation, are loaded
on CPU before model parameters are copied to their device.

## Observations, references, and replay

The WAM state is the canonical 71-dimensional physical state. The actor's
single-step proprioceptive frame is reconstructed from predicted physical
states, so it does not need to equal the WAM state. The `tokenizer` group
remains outside the dynamics model: G1 retargeted,
teleoperation, and SMPL references follow the same clip's reference timeline. Robot-relative
G1/teleop/SMPL orientations are stored in world coordinates and transformed
back using the predicted robot rotation before entering Sonic's original
encoder -> FSQ -> decoder at each imagined step. Reference-local VR points and
other exogenous features are retained unchanged. Immutable reference banks
store clean observations; tokenizer noise, clipping and scaling are applied
in the simulator's order after converting to the imagined robot frame. WAM
therefore receives only `(physical_state, action, ξ)` and shares one robot
dynamics model across all three target representations. FSQ tokens are never
passed to WAM. The physical WAM uses SuperMimic's frame-local invariant input
and predicts local `dp, dtheta, dv, dw, dq, dqd` increments before integrating
them back into the 71-dimensional state. New G1 and N3 physical WAM runs use
`domain_randomization: supermimic` with `complete_context: false`. Their raw ξ
is exactly `[friction (1), body-joint default offsets (29), torso CoM offset (3)]`.
The world-model input is therefore 71 state features + 29 actions + 33 context
values = 133 dimensions. Context is not corrupted with sensor noise. Replay
and actor calibration keep raw radian/metre values; privileged model/critic
features first apply `[(mu-.75)/.45, offsets/.01, com/.05]`. World-model context
and complete critic inputs then use their own running normalization.

The matched training events sample one continuous `mu ~ U[.3,1.2]` per
environment at startup, shared by all collision shapes, and one torso CoM
offset with x in +/-0.025 m and y/z in +/-0.05 m. PhysX receives static and
dynamic friction equal to mu, fixed zero restitution, and a ground coefficient
of 0.3 with max combination. Calibration offsets are resampled independently
in +/-0.01 rad at every reset from immutable nominal defaults, never accumulated.
There are no mass-randomization or periodic push events in this profile.
The same settings apply to G1 and N3; N3 uses its named torso and canonical
29-body-joint ordering rather than G1 asset names.

Reset position, velocity and joint-angle perturbation ranges match SuperMimic.
Orientation uses a left-multiplied rotation-vector exponential and reset joint
positions are clipped to hard limits; joint velocities retain their reference
values. Extra dexterous joints receive no additional calibration or reset noise.
The profile uses flat ground and disables reference-clip pose augmentations.
Proprioceptive noise remains q +/-0.01, qd +/-0.5, body angular velocity +/-0.2,
gravity +/-0.05 and previous action zero. Reference-relative orientation noise
is +/-0.05 on the current frame only; future targets and extra SMPL/VR targets
are clean, using matching real and imagined observation transforms.
Collection adds Gaussian action noise (default sigma 0.1) and then clips once
to +/-20; imagined actions receive neither this exploration noise nor clipping.

These are matched distributions and context semantics, not identical robot
dynamics: N3/G1 assets and PhysX versus MuJoCo-Warp still differ. Reset-frame
sampling remains SONIC's failure-based sampler rather than SuperMimic's
return-based sampler. Existing 707-D world-model/critic checkpoints cannot be
resumed under the 33-D profile; load policy weights with `resume=false` and
refit those models. Saved context and numerical schemas reject incompatible
full resumes. Historical configurations without the new profile keep their
old behavior. For a fresh legacy experiment select
`trainer.settings.domain_randomization=sonic` and explicitly choose its terrain
and `complete_context` settings.

The older explicit `complete_context: true` mode retains
SuperMimic's 33-dimensional prefix `[mean static friction, joint-default
offsets (29), torso CoM offset (3)]` and appends every collision shape's static
friction, dynamic friction and restitution; every body's mass, 3x3 inertia and
local CoM pose; and every joint's default-position offset. Its size is
`33 + 3*S + 17*B + J` for S collision shapes, B bodies and J joints. Ordered
body/joint names, shape ordering and tensor conventions are saved and checked
on full resume. Properties are read from PhysX and cached until a material,
mass or CoM event invalidates the snapshot; calibration offsets are read fresh.
The complete vector conditions WAM and critic, never FSQ. Within independent
policy imagination ξ stays fixed to the sampled seed episode.

Replay stores the current and next physical state, reference state,
actor observations, goals and immutable reference indices needed to re-run
the Sonic actor, together with the previous executed body action for the
SuperMimic action-rate term. Those goal tensors are not WAM inputs and no
modality label is used by the dynamics model. The old combined
observation format remains accepted for compatibility with earlier tests and
checkpoints.

Trainer observation keys refer to the dictionary returned by `ManagerEnvWrapper`.
Physical WAM sets `critic_observation_keys: []`: Sonic's `critic_obs` contains
the logged real robot trajectory and is not used for imagined states. The
physical critic receives state invariants, joint/body/root tracking errors,
relative root orientation, ξ, and raw robot future-reference features, with its
own synchronized normalizer. Tracking errors are recomputed through FK using
the imagined robot; they retain gradients to its XY, heading and joint state.
The critic never consumes actor encoder outputs or FSQ tokens. Actor joint
calibration is reconstructed from nominal positions and the sampled ξ offsets,
never from the current simulator environment rows.

With 14 tracked bodies and ten future frames, the shared critic has
`881 + context_dim` inputs (914 in legacy 33-D context mode). It combines
`151 + context_dim` state/tracking/context values with 730 reference-only future
features. Each future frame contains displacement and orientation in
the current reference's heading frame, future local root velocities, joint
positions and scaled joint velocities. All actor modalities use this same
canonical robot representation. The default `critic_future_frames` and
`critic_future_dt` follow the G1 command's frame count and interval (ten frames
at 0.1 seconds in `sonic_release`, starting at the current frame). Set
`++trainer.settings.critic_future_dt=0.02` to use a 20-ms critic grid instead.
The interval must be a positive multiple of the control timestep.

Independent imagination reconstructs critic windows from immutable raw 71-D
clip frames, clamped within that clip. Each clip's physical and multimodal
references are stored once on CPU and survive motion-library subset reloads;
replay stores two integer indices per transition. Old generations are retained
while replay still refers to them. This avoids storing two duplicate critic
windows per replay transition. With `compact_reference: true`, replay also
omits full current/next goals and reference states, storing only clip/frame
indices and the seed's modality/frame routing. The full clean reference goals
are reconstructed from the same bank before applying the original noise,
clipping and scaling. `compact_replay: true` stores single proprioceptive
frames and reconstructs term-major actor histories with a boundary halo and
original noisy segment prefixes. No history is re-noised or spliced across a
reset. Both optimizations have a `false` option for numerical comparisons;
old replay snapshots restore their saved storage format. The per-rank
`replay/host_gib` metric counts allocated replay tensors, halo and prefixes,
excluding reference banks, simulator memory and temporary batches.
Complete context does increase replay storage;
its dimension depends on the actual loaded robot. Sampled policy trajectories
and fitting windows are transferred to GPU.

`SequenceReplayBuffer` stores actual noisy, clipped actions. Samples stay in
one environment and in chronological order across ring wraparound. Reward and
termination supervision include the final transition. Dynamics supervision
uses a pre-reset physical state cached by the aligned termination callbacks for
the final transition. When no current-step cache is available, the terminal
target remains masked rather than learning the reset jump. Later steps are
masked. External velocity-push events are recorded separately; the pushed
transition and the remainder of its autoregressive fitting window do not
supervise deterministic dynamics. Real pushes and their recovery states remain
available to collection and policy-start sampling. Truncations are not learned
as physical termination. Real bool done only controls resets and fitting-window
continuity. With `independent_imagination: true`, actor starts are sampled
independently of subsequent real done values. Their clip references advance for
H steps, limited only by the actual clip end, and analytic soft termination
determines survival weights and their gradients. A behaviour-policy fall does
not stop a new imagined trajectory. Critic statistics include valid imagined
states and their bootstrap state. The
actor history is advanced from the predicted single-step proprioceptive
observation.
Optional replay checkpoints seal episode boundaries before collecting from a
reset simulator.

The WAM defaults follow SuperMimic's training scale: 1000 replay steps per
environment, world-model batch size 2048 (policy imagination batch 4096), action noise 0.1, learning rate
2e-4 with cosine decay to 3% over 250k iterations, five critic epochs with
four minibatches, UTD 0.4, 64-step warmup, and 500 world-model-only
iterations. Physical WAM output deltas use an independent normalizer with
epsilon 1e-4. Rollout errors use `max(delta_std, state_error_floor)` with floors
`[.003, .005, .02, .05, .003, .05]` for position, rotation, linear velocity,
angular velocity, joint position and joint velocity. Input and critic
normalizers retain their separate scales. Reward and termination are analytic functions of predicted and
reference physical states; they are not learned heads. Observation noise on
the WAM actor frame uses uniform half-widths gravity .05, angular velocity .2,
joint position .01, joint velocity .5 and action 0. The same shared definition
configures real training, imagined frames and default checkpoint evaluation;
override `trainer.settings.observation_noise` to change all three together.
IsaacLab remains the simulator.
Non-finite losses or gradients reject the update. Real simulator reward and
imagined reward are logged separately. Validation must compare open-loop model
errors and real tracking returns, not just an increasing imagined return.

These configuration values do not imply exact SuperMimic behavior.
`warmup_before_updates: true` now collects `warmup_steps` real steps before
the first optimization, including when a restart refills an empty replay.
Set it to `false` to reproduce the previous actor-only warmup gate.
`reset_sampling: adaptive_failure` explicitly selects GR00T's existing
failure-based adaptive sampler; the old misleading value `return` is rejected
because SuperMimic's return-based reset sampler is not implemented.
The matched SuperMimic domain enforces flat ground. With the legacy `sonic`
domain profile, `trainer.settings.terrain_type: null` retains the experiment's
terrain. A `trimesh` terrain is not encoded in the robot-only physical context.

For controlled comparisons, use the same robot, fixed held-out motion list,
reward/termination definitions, encoder, observation noise, seeds and GPU-hour
budget for PPO and WAM. Disable motion augmentation for validation and score
complete real simulator trajectories per modality, rather than imagined return.
Evaluate one numerical change at a time; passing unit tests does not establish
a convergence or hardware-performance improvement.

## Reading the training diagnostics

These diagnostics describe the current simulator collection and training replay;
they are not an independent validation set. There are no established convergence
thresholds for this WAM baseline. Compare fixed validation
motions, actual tracking errors, and complete-motion success rates for each
input modality before judging control quality or a speedup over PPO.

| Metric | Meaning and interpretation |
| --- | --- |
| `real/reward` | Mean real simulator reward per environment step, including the environment's reward weights and timestep scaling. It is not an episode return or the original PPO logger's `rewards/total`. |
| `real/reward_std` | Population standard deviation of all real reward samples collected by this rank in the iteration. Compare reward prediction RMSE with this scale and with a mean-reward baseline. |
| `real/done_fraction` | Fraction of collected environment steps that ended an episode, including real terminations and time-limit truncations. |
| `real/terminated_fraction`, `real/truncated_fraction` | Separate rates for real termination and time-limit truncation; a time limit is not treated as a learned physical failure. |
| `real/action_rms`, `real/action_abs_max` | Magnitudes of the noisy, clipped actions actually sent to the simulator. These help identify action growth that may cause rate or joint-limit penalties. |
| `real/action_at_limit_fraction` | Fraction of executed action coordinates at the configured absolute action limit. It is not the fraction of raw policy outputs that were clipped. |
| `world/valid_steps_mean` | Mean number of usable reward/termination steps in each sampled fitting window, up to `window`. |
| `world/dynamics_valid_fraction` | Fraction of fitting-window steps with a usable, non-reset next state for dynamics supervision. |
| `world/last_valid_fraction` | Fraction of sampled windows with a usable final dynamics step. Interpret `world/state_error_last` only when this is positive. |
| `actor/effective_horizon` | Mean number of valid reference steps before the sampled replay boundary, up to `horizon`; this can be far shorter than the configured imagination horizon. |
| `actor/aux_loss` | Coefficient-weighted FSQ auxiliary loss at the imagined seed. `actor/loss` adds this term multiplied by `aux_loss_scale` to the negative discounted imagined return (`actor_return_reduction=sum` by default). |

Scalar environment summaries with the prefixes `Episode_Reward/`,
`Episode_Termination/`, and `Metrics/motion/` are also logged under `real/`.
Each is averaged over the collection steps where that scalar was available;
missing, empty, non-scalar, or non-finite values do not count as zeros. These
summaries retain the simulator's episode-summary definitions and normalization,
so they need not equal instantaneous reward terms.

Final logged statistics are averages across ranks. In particular,
`real/reward_std` is the average of per-rank standard deviations, not a pooled
global standard deviation; `real/action_abs_max` is the average of rank maxima,
not the maximum across all GPUs. `world/*` describe the final fitting batch of
the iteration rather than the average of all world-model updates.

If every sampled trajectory terminates before the fitting window's last step,
`world/state_error_last` is zero because there are no valid final-step targets.
That is missing evidence, not a perfect prediction. The accompanying
`world/last_valid_fraction=0` makes this case explicit. Masking terminal reset
observations must remain enabled: removing the masks would train the model to
predict artificial jumps into a new episode.

Persistently worsening real reward, frequent real terminations, and a short
effective horizon warrant stopping a long run and inspecting the reward
components, executed actions, and checkpoint history. A smaller state loss can
coincide with worse control because it is computed only on surviving valid
steps and does not measure target tracking quality. Likewise,
`actor/survival` is a model prediction, not the real success rate; a small critic
loss only shows agreement with imagined value targets. The actor objective can
also prefer avoiding future negative rewards through predicted termination,
so inspect real failure rates rather than treating rising imagined return as
proof of improvement. These diagnostic changes do not establish convergence or
resolve every possible cause of policy collapse.

A tiny termination fitting loss together with optimistic imagined survival
and frequent real terminations warrants checking the termination model on
held-out real transitions. These statistics use different states and actions,
so they are not directly comparable success rates. Avoid judging calibration
from fitting loss alone.

## Recording real simulator videos

Use the saved FSQ policy to record actual Isaac Lab rollouts, independently of
the world model's imagined survival probability. The existing evaluator and
render recorder already support this; no additional recording script is needed:

```bash
CUDA_VISIBLE_DEVICES=0 python gear_sonic/eval_agent_trl.py \
  checkpoint=/path/to/training_run/last.pt \
  ++headless=true \
  ++num_envs=1 \
  '++eval_callbacks=[]' \
  ++run_eval_loop=true \
  ++run_once=false \
  ++max_render_steps=500 \
  ++manager_env.config.render_results=true \
  ++manager_env.config.save_rendering_dir=/path/to/new_video_dir \
  manager_env/recorders=render
```

Keep `config.yaml` beside the checkpoint. This command records approximately
ten seconds to `/path/to/new_video_dir/000000.mp4`, using the original training
termination conditions and continuing across resets so failures remain visible.
It skips full-dataset evaluation and does not start WAM training. Add
`++use_encoder=g1` or `++use_encoder=smpl` to select one of those modalities.
Rendering requires additional GPU memory; use a GPU with sufficient space.

The evaluator's existing render flags differ from Isaac Lab's standard
`play.py --video --video_length ...` interface. WAM checkpoint evaluation also
requires the step-metadata compatibility fix: read `wam_iteration` when PPO's
`state.global_step` is absent. Policy-only evaluation does not need the rank
sidecars used for full training resume.

## Optional utilities and acceleration

`gear_sonic.trl.world_model` also exposes a separate latent `WAM`,
`FSQObservationAdapter`, `ModelBasedActorCriticTrainer`, and CEM `LatentMPC`
for experiments. These are not the dynamics implementation selected by
`trainer=wam`. `UniversalTokenModule(..., return_dict=True)` returns
`full_tokens` and `full_latent` with their gradients intact for these consumers.
An external latent actor needs its own observation adapter; it is not a drop-in
replacement for the original FSQ actor.

Imagined updates avoid simulator steps. `trainer.settings.compile=true` opts
into compiling the state-space WAM forward pass; initial compilation adds cost.
Neither training speedup nor real robot performance has been measured here.
MPC searches multiple candidates and can cost more than direct actor inference.
The paper's acceleration ratios do not transfer automatically to this baseline.

## Verification

With PyTorch, Hydra, TensorDict, and vector-quantize-pytorch installed:

```bash
python -m unittest gear_sonic.tests.test_wam_termination -v
python -m unittest gear_sonic.tests.test_wam_reward -v
python -m unittest gear_sonic.tests.test_wam_rotations gear_sonic.tests.test_wam_observations -v
python -m unittest gear_sonic.tests.test_wam_state gear_sonic.tests.test_wam_terminal_capture gear_sonic.tests.test_wam_physical_training -v
python -m unittest gear_sonic.tests.test_sequence_replay gear_sonic.tests.test_wam_training -v
python -m gear_sonic.tests.test_wam
GLOO_SOCKET_IFNAME=lo python -m unittest gear_sonic.tests.test_wam_distributed -v
```

The tests include the original `Actor`, `UniversalTokenModule`, `BaseModule`,
and actual FSQ quantizer with all three modality encoders, vector replay,
auto-resets, BPTT parameter updates, frozen dynamics, goal timeline replay,
legacy projection compatibility, checkpoint restoration, and latent utility edge cases. They use a small toy
environment and do not validate humanoid simulation or sim-to-real behavior.
The distributed tests use two real CPU/Gloo processes, different seeds/data,
different modality selections, and rank-local checkpoint restoration. They do
not establish CUDA/NCCL performance or eight-GPU humanoid convergence.
