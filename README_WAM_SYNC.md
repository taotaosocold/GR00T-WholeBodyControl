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
  gpu14:/data/ouqin/
```

## 3. 在 GPU14 同步仓库

先按第 7 节停止需要更新的训练，确认进程退出后，再执行：

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

在 GPU14 上先创建 tmux 会话，本节和第 5、6 节的命令都在该会话内执行：

```bash
tmux new -s wam
```

已有 `wam` 会话时使用 `tmux attach -t wam` 返回。
PyTorch Elastic 会注册 SIGHUP 处理器，可能覆盖 `nohup` 的忽略设置，
因此这里使用 tmux 保留训练会话。进入 tmux 后再设置下面的环境变量。

```bash
cd /data/ouqin/GR00T-WholeBodyControl

export PY=/data/ouqin/envs/sonic/bin/python
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

"$PY" -c 'import sys, torch, accelerate; print(sys.executable); print(torch.__version__); print(accelerate.__version__)'
```

## 5. 8 卡 WAM 冒烟测试

`num_envs` 是每个进程的环境数。下面的 `64` 表示 8 卡共 512 个环境。
此命令关闭 UTD 调度以固定每轮采集 32 步，并缩短 actor 预热，
用于在 5 轮内覆盖 WAM、actor BPTT 和 critic 更新。

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
  ++trainer.settings.utd=0 \
  ++trainer.settings.collect_steps=32 \
  ++trainer.settings.wm_only_iters=0 \
  ++trainer.settings.warmup_steps=32 \
  ++trainer.settings.replay_steps=64 \
  ++trainer.settings.replay_envs=64 \
  ++trainer.settings.window=8 \
  ++trainer.settings.horizon=8 \
  ++trainer.settings.batch_size=32 \
  ++trainer.settings.policy_batch_size=32 \
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

确认日志包含 `world/loss`、`actor/loss`、`critic/loss` 且数值有限，
并生成 `last.pt`，才说明这次冒烟覆盖了三类更新。

## 6. 正式训练

下面按 SuperMimic 单进程默认的全局采样规模分配到 8 卡：
每卡 512 个环境，每环境 replay 1000 步；WAM batch 每卡 256（全局 2048），
actor batch 每卡 512（全局 4096）。`utd=0.4` 每轮交替采集 2、3 步，
此时 `collect_steps` 不参与调度。这是采样规模对齐，不代表完整算法已严格复现：
当前 WAM 在 replay 达到 8 步就开始拟合，`warmup_steps` 仅约束 actor；
`reset_sampling: return` 尚未接到 reset 采样逻辑，仍使用 GR00T 的自适应采样。

```bash
RUN_DIR=/data/ouqin/runs_supermimic/wam_4096_total_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -u \
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
  ++algo.config.num_learning_iterations=320000 \
  ++trainer.settings.utd=0.4 \
  ++trainer.settings.motion_resample_frequency=250 \
  ++trainer.settings.replay_steps=1000 \
  ++trainer.settings.replay_envs=512 \
  ++trainer.settings.window=8 \
  ++trainer.settings.horizon=32 \
  ++trainer.settings.batch_size=256 \
  ++trainer.settings.policy_batch_size=512 \
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
  < /dev/null > "$RUN_DIR/train.log" 2>&1
```

正式训练在 tmux 内前台运行。启动后，按 **Ctrl+B**，松开后按 **D** 分离会话，
随后即可断开 SSH。重新登录后执行 `tmux attach -t wam` 返回训练会话。
tmux 不会在服务器重启或训练程序报错后自动恢复训练。

查看训练（另开一个 SSH 终端，将 `RUN_DIR` 设为实际训练目录）：

```bash
RUN_DIR=/data/ouqin/runs_supermimic/wam_4096_total_实际时间戳
tail -f "$RUN_DIR/train.log"
nvidia-smi
```

断点续训时，在 tmux 内重新执行第 4 节的环境设置，将第 6 节开头的 `RUN_DIR`
改为原训练目录，并将原命令中的 `+resume=false` 和 `checkpoint=null` 替换为：

```bash
  +resume=true \
  checkpoint="$RUN_DIR/last.pt" \
```

其余参数保持与原训练一致（包括 8 个进程及每卡环境数），日志重定向改为
`>> "$RUN_DIR/train.log" 2>&1`，保留已有日志。多卡恢复需要保留 `last.pt`
引用的配套 `last.*.rank*.pt` 文件。`save_replay=false` 时不恢复经验池，
会重新采样和预热；恢复的是最近成功保存的 checkpoint。

## 7. 停止训练与检查残留进程

以下命令在运行训练的服务器上执行。`RUN_DIR` 必须是要停止的那次训练目录，
重新登录后需要重新设置。将下面的 `/目录名` 替换为实际目录。
先确认主机；如果有 PID 文件，再查看启动器和 worker：

```bash
RUN_DIR=/data/ouqin/runs_supermimic/wam_4096_total_20261029_095623

for STOP_PID in $(pgrep -u "$(id -u)" -f 'gear_sonic/train_agent_trl[.]py' || true); do
  if [ "$(stat -c %u "/proc/$STOP_PID" 2>/dev/null)" = "$(id -u)" ] &&
     grep -Fzxq -- "experiment_dir=$RUN_DIR" "/proc/$STOP_PID/cmdline" 2>/dev/null; then
    printf '发送 TERM：PID=%s\n' "$STOP_PID"
    kill -TERM "$STOP_PID"
  fi
done

sleep 35
nvidia-smi
pgrep -a -u "$(id -u)" -f 'gear_sonic/train_agent_trl[.]py'
```

## 8. 录制屏幕
```bash
cd /data/ouqin/GR00T-WholeBodyControl
PY=/data/ouqin/envs/sonic/bin/python

RUN_DIR=/data/ouqin/runs_supermimic/wam_4096_total_20261028_144841
MOTION_KEY=walk_ff_loop_180_R_002__A047
MOTION_FILE="/data/ouqin/datasets/g1/train/robot/221010/${MOTION_KEY}.pkl"
VIDEO_DIR="$RUN_DIR/videos_${MOTION_KEY}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=0 "$PY" gear_sonic/eval_agent_trl.py \
  checkpoint="$RUN_DIR/last.pt" \
  ++headless=true \
  ++num_envs=1 \
  '++eval_callbacks=[]' \
  ++run_eval_loop=true \
  ++run_once=false \
  ++max_render_steps=1500 \
  ++use_encoder=g1 \
  ++manager_env.commands.motion.motion_lib_cfg.motion_file="$MOTION_FILE" \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=/data/ouqin/datasets/g1/train/smpl \
  ++manager_env.commands.motion.motion_lib_cfg.paired_frame_alignment.mode=strict \
  "++manager_env.commands.motion.filter_motion_keys=[\"$MOTION_KEY\"]" \
  ++manager_env.commands.motion.start_from_first_frame=true \
  ++manager_env.commands.motion.cat_upper_body_poses=false \
  ++manager_env.commands.motion.freeze_frame_aug=false \
  ++manager_env.observations.policy.enable_corruption=false \
  ++manager_env.events.physics_material=null \
  ++manager_env.events.add_joint_default_pos=null \
  ++manager_env.events.base_com=null \
  ++manager_env.events.randomize_rigid_body_mass=null \
  ++manager_env.config.render_results=true \
  ++manager_env.config.save_rendering_dir="$VIDEO_DIR" \
  manager_env/recorders=render
```

## 9. 导出控制策略 ONNX

下面导出 WAM 训练得到的三个 encoder（含 FSQ）和 shared decoder。
将 `RUN_DIR` 改为实际训练目录，并保留 checkpoint 旁的 `config.yaml`。
导出入口仍会初始化 Isaac Lab 环境，需要在已配置好 Isaac Lab 的 sonic 环境中运行，
且保存配置中的机器人资产和运动数据路径必须可访问。

```bash
cd /data/ouqin/GR00T-WholeBodyControl
PY=/data/ouqin/envs/sonic/bin/python
RUN_DIR=/data/ouqin/runs_supermimic/wam_4096_total_20261029_095623

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'import gear_sonic; print(gear_sonic.__file__)'

CUDA_VISIBLE_DEVICES=0 "$PY" -m gear_sonic.eval_agent_trl \
  checkpoint="$RUN_DIR/last.pt" \
  ++headless=true \
  ++num_envs=1 \
  '++eval_callbacks=[]' \
  ++export_onnx_only=true
```

导入路径应指向 `/data/ouqin/GR00T-WholeBodyControl/gear_sonic/`。
使用仓库根目录下的 `-m gear_sonic.eval_agent_trl` 入口，避免环境中的
`sonic_bumi_full` 等其他仓库抢先提供同名包。导出要求 `num_envs=1`，
无需使用 8 卡训练启动器；导出完成后程序退出，不进入录屏循环。

输出文件位于：

```text
RUN_DIR/
├── exported/
│   ├── model_step_XXXXXX_encoder.onnx
│   ├── model_step_XXXXXX_decoder.onnx
│   ├── model_step_XXXXXX_g1.onnx
│   ├── model_step_XXXXXX_smpl.onnx
│   └── model_step_XXXXXX_teleop.onnx
├── model_config.yaml
└── model_step_XXXXXX.pt
```

## 10. Sim2Sim / PICO / Sim2Real

下面以 tsz 的 G1 仓库和第 6000 步模型为例，命令在 tsz 的图形桌面终端执行。
三种方式共用部署机器上的模型目录：

```text
/home/user/tsz/GR00T-WholeBodyControl/gear_sonic_deploy/policy/wam_step_006000/
├── model_step_006000_encoder.onnx
├── model_step_006000_decoder.onnx
└── observation_config.yaml
```

encoder/decoder 使用同一次导出的文件。当前 `sonic_release` 的配置可从
`gear_sonic_deploy/policy/release/observation_config_sonic_release.yaml` 复制为上述
`observation_config.yaml`。换模型时同步修改下面的 `--cp` 前缀和配置路径。

### Sim2Sim：两个终端

终端 1 启动 MuJoCo（需已安装 `.venv_sim`）：

```bash
cd /home/user/tsz/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

终端 2 加载 WAM 模型；`--cp` 不带 `_encoder.onnx` / `_decoder.onnx` 后缀：

```bash
cd /home/user/tsz/GR00T-WholeBodyControl/gear_sonic_deploy
bash deploy.sh \
  --cp policy/wam_step_006000/model_step_006000 \
  --obs-config policy/wam_step_006000/observation_config.yaml \
  --planner /home/user/Downloads/sonic_bumi-main/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx \
  --input-type keyboard \
  sim
```

终端 2 按 `]` 启动，MuJoCo 窗口按 `9` 放下机器人；回到终端 2 按 `T` 播放，
`N/P` 切换动作，`O` 停止退出。

### PICO：仿真增加第三个终端

沿用上面的两个终端，将部署命令的 `--input-type keyboard` 改为
`--input-type zmq_manager`，保留 planner。终端 3 启动采集（需已配置 `.venv_teleop`）：

```bash
cd /home/user/tsz/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

PICO 按 `A+B+X+Y` 校准并启停控制，`A+X` 切换全身 POSE 模式；仿真中仍需在
MuJoCo 窗口按 `9` 放下机器人。默认采集与部署在同机，完整操作见
[PICO 教程](docs/source/tutorials/vr_wholebody_teleop.md)。

### Sim2Real

```bash
cd /home/user/Desktop/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env_local.sh

ROBOT_IF=enp3s0  # 改为连接 G1 的实际有线网卡名

bash deploy.sh \
  --cp policy/11.2/model_step_007000 \
  --obs-config policy/11.2/observation_config.yaml \
  --planner planner/target_vel/V2/planner_sonic.onnx \
  --motion-data reference/example/ \
  --input-type keyboard \
  --output-type zmq \
  "$ROBOT_IF"
```
### 如果用PICO实机遥操作则
--input-type keyboard改成--input-type zmq_manager，然后在开一个终端执行
```bash
cd /home/user/Desktop/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```
