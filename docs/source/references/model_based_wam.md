# Model-based FSQ + WAM training

The opt-in `WAMTrainer` keeps the original `Actor -> UniversalTokenModule ->
FSQ -> decoder` policy. It replaces PPO updates with a SuperMimic-inspired
Dyna loop: collect simulator transitions, fit differentiable dynamics on
8-step replay windows, improve the real FSQ actor through 32 imagined steps,
and fit a critic with GAE/lambda returns and a terminal bootstrap. World-model
and critic weights are frozen during actor backpropagation; gradients still
flow through their inputs. Existing reconstruction and modality-alignment
losses are applied at the imagined rollout seed.

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

The trainer resolves state and action dimensions from the environment. Settings
live in `gear_sonic/config/trainer/wam.yaml`; the dynamics architecture lives in
`gear_sonic/config/world_model/wam_dynamics.yaml`. This path supports single-GPU
and data-parallel multi-GPU training with direct joint actions and an FSQ MLP
actor. Actor-side RunningMeanStd/BatchNorm, temporal attention, and
hierarchical action transforms are rejected explicitly. Observation histories
already flattened into `actor_obs` remain supported.

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
64 environments on eight ranks mean 512 simulator environments in total. Each
rank collects its own trajectories and samples its own replay. Actor, dynamics,
and critic gradients are averaged after each backward pass; initial weights and
world-model normalization statistics are synchronized. Rank-specific unused
modality encoders participate safely. A non-finite loss/gradient on any rank
rejects the update on all ranks. Adaptive motion sampling remains local.

For a five-iteration smoke test that includes actor updates, override
`trainer.settings.world_updates=2` and
`trainer.settings.warmup_world_updates=4`. With the defaults, five iterations
produce only 50 world updates, below the 100-update actor warmup threshold.

The base config leaves WAM disabled. Omitting `trainer=wam world_model=wam_dynamics`
keeps the existing PPO path. `checkpoint=/path/to/last.pt` loads a compatible
FSQ policy; adding `+resume=true` restores WAM, critic, optimizers, and counters
from a WAM checkpoint. Policy weights remain under `policy_state_dict`, so the
existing policy evaluation loader can read them. The trainer uses its own
checkpoint and logging loop rather than PPO-specific callbacks.

Only rank zero writes the shared `last.pt`. Multi-process saves additionally
write generation-specific rank sidecars beside it for independent RNG states
and optional replay. Keep the referenced sidecars when copying a checkpoint
for a full resume. A full resume requires the same process count. To change
GPU count, use `resume=false` to load the policy and start fresh dynamics/replay.
Resume preserves rank-local replay boundaries but does not restore simulator
physics or promise bitwise continuation. Logs report averages across ranks.

## Observations, references, and replay

This implementation is an **observation-space baseline**. It predicts the
original flat `actor_obs` and robot-relative tokenizer features, rather than
SuperMimic's canonical rigid-body state, analytic kinematics/rewards, and
explicit randomized simulator context. Thus it is not a numerical reproduction
of the paper. Noise and unobserved physical context can reduce its accuracy.

Known action-independent reference features, including reference joint
trajectories, SMPL local joints, VR targets, and reference heights, advance
along their logged replay timeline. The actor cannot change these goals through
model gradients. Robot-relative orientation features remain predictions of the
world model. Encoder routing bits are copied from the seed. New tokenizer terms
must be classified using `settings.exogenous_tokenizer_features`; this setting
replaces the built-in reference-feature list when provided.

`SequenceReplayBuffer` stores actual noisy, clipped actions. Samples stay in
one environment and in chronological order across ring wraparound. Reward and
termination supervision include the final transition. Dynamics supervision
excludes auto-reset next observations; later steps are masked. Truncations are
not learned as physical termination. Imagined rollouts stop at the logged
reference boundary and bootstrap there. Optional replay checkpoints seal that
boundary before collecting from a reset simulator.

The defaults warm up the WAM before updating the actor. Non-finite losses or
gradients reject the update. Real simulator reward and imagined reward are
logged separately. Validation must compare open-loop model errors and real
tracking returns, not just an increasing imagined return.

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
python -m unittest gear_sonic.tests.test_sequence_replay gear_sonic.tests.test_wam_training -v
python -m gear_sonic.tests.test_wam
GLOO_SOCKET_IFNAME=lo python -m unittest gear_sonic.tests.test_wam_distributed -v
```

The tests include the original `Actor`, `UniversalTokenModule`, `BaseModule`,
and actual FSQ quantizer with all three modality encoders, vector replay,
auto-resets, BPTT parameter updates, frozen dynamics, reference projection,
checkpoint restoration, and latent utility edge cases. They use a small toy
environment and do not validate humanoid simulation or sim-to-real behavior.
The distributed tests use two real CPU/Gloo processes, different seeds/data,
different modality selections, and rank-local checkpoint restoration. They do
not establish CUDA/NCCL performance or eight-GPU humanoid convergence.
