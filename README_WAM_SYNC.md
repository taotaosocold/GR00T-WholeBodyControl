# WAM 本地修改、服务器同步与训练

当前分支：`supermimic`  
服务器环境：`/data/ouqin/envs/sonic/bin/python`  
bundle 名称：`GR00T-WholeBodyControl.bundle`

## 1. 本地检查修改

```bash
cd /home/ubuntu/桌面/GR00T-WholeBodyControl

git status --short
git diff --stat
git diff
```

确认修改内容后，再添加需要提交的文件：

```bash
git add <需要提交的文件>
git diff --cached --check
git diff --cached --stat
git commit -m "Update WAM training"
```

`git bundle` 只包含已经提交的内容。未提交的修改不会自动进入 bundle。

## 2. 生成并发送 bundle

```bash
cd /home/ubuntu/桌面/GR00T-WholeBodyControl

rm -f /home/ubuntu/桌面/GR00T-WholeBodyControl.bundle

git bundle create \
  /home/ubuntu/桌面/GR00T-WholeBodyControl.bundle \
  supermimic

git bundle verify /home/ubuntu/桌面/GR00T-WholeBodyControl.bundle

scp /home/ubuntu/桌面/GR00T-WholeBodyControl.bundle \
  hjq@RTX4090-gpu-014:/data/ouqin/
```

## 3. 在 GPU14 同步仓库

先确认服务器上没有正在运行的训练，再执行：

```bash
cd /data/ouqin/GR00T-WholeBodyControl

git status --short
git branch --show-current

git fetch /data/ouqin/GR00T-WholeBodyControl.bundle supermimic
git checkout supermimic
git reset --hard FETCH_HEAD

git log -1 --oneline
git status --short
```

如果服务器仓库有未保存的修改，先备份或处理这些修改，再执行 `git reset --hard`。

## 4. 检查 sonic 环境

```bash
cd /data/ouqin/GR00T-WholeBodyControl

export PY=/data/ouqin/envs/sonic/bin/python
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"$PY" -c 'import sys, torch, accelerate; print(sys.executable); print(torch.__version__); print(accelerate.__version__)'
```

## 5. 运行测试

```bash
"$PY" -m compileall -q gear_sonic/trl gear_sonic/tests

"$PY" -m unittest \
  gear_sonic.tests.test_sequence_replay \
  gear_sonic.tests.test_wam_projection \
  gear_sonic.tests.test_wam_training -v

"$PY" -m gear_sonic.tests.test_wam

GLOO_SOCKET_IFNAME=lo "$PY" \
  -m unittest gear_sonic.tests.test_wam_distributed -v
```

## 6. 8 卡 WAM 冒烟测试

`num_envs` 是每个进程的环境数。下面的 `64` 表示 8 卡共 512 个环境。

```bash
RUN_DIR=/data/ouqin/runs_supermimic/wam_smoke_sonic_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" \
  -m accelerate.commands.launch \
  --multi_gpu \
  --num_machines=1 \
  --num_processes=8 \
  --machine_rank=0 \
  --main_process_ip=127.0.0.1 \
  --main_process_port=29519 \
  --mixed_precision=no \
  --dynamo_backend=no \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  trainer=wam \
  world_model=wam_dynamics \
  +resume=false \
  checkpoint=null \
  auto_load_latest=false \
  use_wandb=false \
  headless=True \
  experiment_dir="$RUN_DIR" \
  num_envs=64 \
  ++algo.config.num_learning_iterations=5 \
  ++trainer.settings.collect_steps=32 \
  ++trainer.settings.replay_steps=64 \
  ++trainer.settings.replay_envs=64 \
  ++trainer.settings.window=8 \
  ++trainer.settings.horizon=8 \
  ++trainer.settings.batch_size=32 \
  ++trainer.settings.world_updates=2 \
  ++trainer.settings.warmup_world_updates=4 \
  ++trainer.settings.actor_updates=1 \
  ++trainer.settings.critic_epochs=2 \
  ++trainer.settings.save_interval=5 \
  ++trainer.settings.compile=false \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/data/ouqin/datasets/g1/train/robot/210531 \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/data/ouqin/datasets/g1/train/smpl \
  ++manager_env.commands.motion.encoder_sample_probs.g1=1.0 \
  ++manager_env.commands.motion.encoder_sample_probs.smpl=1.0 \
  ++manager_env.commands.motion.encoder_sample_probs.teleop=0.0 \
  > "$RUN_DIR/train.log" 2>&1 &

echo $! > "$RUN_DIR/launcher.pid"
tail -f "$RUN_DIR/train.log"
```

看到 `WAM iteration=...` 并生成 `last.pt` 后，再开始正式训练。

## 7. 正式训练

如果目标是 8 卡总共 4096 个环境，应设置每卡 `512` 个环境：

```bash
RUN_DIR=/data/ouqin/runs_supermimic/wam_4096_total_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" \
  -m accelerate.commands.launch \
  --multi_gpu \
  --num_machines=1 \
  --num_processes=8 \
  --machine_rank=0 \
  --main_process_ip=127.0.0.1 \
  --main_process_port=29519 \
  --mixed_precision=no \
  --dynamo_backend=no \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  trainer=wam \
  world_model=wam_dynamics \
  +resume=false \
  checkpoint=null \
  auto_load_latest=false \
  use_wandb=false \
  headless=True \
  experiment_dir="$RUN_DIR" \
  num_envs=512 \
  ++algo.config.num_learning_iterations=50000 \
  ++trainer.settings.collect_steps=32 \
  ++trainer.settings.motion_resample_frequency=250 \
  ++trainer.settings.replay_steps=256 \
  ++trainer.settings.replay_envs=64 \
  ++trainer.settings.window=8 \
  ++trainer.settings.horizon=32 \
  ++trainer.settings.batch_size=32 \
  ++trainer.settings.world_updates=10 \
  ++trainer.settings.warmup_world_updates=100 \
  ++trainer.settings.actor_updates=1 \
  ++trainer.settings.critic_epochs=5 \
  ++trainer.settings.learning_rate=0.0002 \
  ++trainer.settings.save_interval=100 \
  ++trainer.settings.save_replay=false \
  ++trainer.settings.compile=false \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file=/data/ouqin/datasets/g1/train/robot \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/data/ouqin/datasets/g1/train/smpl \
  ++manager_env.commands.motion.encoder_sample_probs.g1=1.0 \
  ++manager_env.commands.motion.encoder_sample_probs.smpl=1.0 \
  ++manager_env.commands.motion.encoder_sample_probs.teleop=0.0 \
  > "$RUN_DIR/train.log" 2>&1 &

echo $! > "$RUN_DIR/launcher.pid"
```

查看训练：

```bash
tail -f "$RUN_DIR/train.log"
nvidia-smi
```

停止本次训练：

```bash
kill -TERM "$(cat "$RUN_DIR/launcher.pid")"
```
