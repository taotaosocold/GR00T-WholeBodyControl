#!/usr/bin/env bash
# Record a saved FSQ policy in Isaac Lab; no desktop display or WAM rollout.
# Usage: bash gear_sonic/record_wam_video.sh RUN_DIR_OR_CHECKPOINT [OUTPUT_DIR]
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${WAM_VIDEO_PYTHON:-/data/ouqin/envs/sonic_bumi/bin/python}"
source_path="${1:?Usage: bash gear_sonic/record_wam_video.sh RUN_DIR_OR_CHECKPOINT [OUTPUT_DIR]}"
gpu="${WAM_VIDEO_GPU:-0}"
num_envs="${WAM_VIDEO_NUM_ENVS:-1}"
steps="${WAM_VIDEO_STEPS:-500}"
encoder="${WAM_VIDEO_ENCODER:-g1}"

[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'WAM_VIDEO_GPU must be one GPU index' >&2; exit 2; }
[[ "$num_envs" =~ ^[1-9][0-9]*$ ]] || { echo 'WAM_VIDEO_NUM_ENVS must be positive' >&2; exit 2; }
[[ "$steps" =~ ^[1-9][0-9]*$ ]] && (( steps >= 4 )) || {
  echo 'WAM_VIDEO_STEPS must be an integer of at least 4' >&2; exit 2;
}
case "$encoder" in
  g1|smpl|teleop|mixed) ;;
  *) echo 'WAM_VIDEO_ENCODER must be g1, smpl, teleop, or mixed' >&2; exit 2 ;;
esac

[[ "$source_path" == /* ]] || source_path="$PWD/$source_path"
case "$source_path" in
  *.pt|*.pth) source_checkpoint="$source_path"; run_dir="$(dirname -- "$source_path")" ;;
  *) run_dir="${source_path%/}"; source_checkpoint="$run_dir/last.pt" ;;
esac
output_dir="${2:-$run_dir/videos/wam_${encoder}_$(date +%Y%m%d_%H%M%S)}"
[[ "$output_dir" == /* ]] || output_dir="$PWD/$output_dir"
snapshot="$output_dir/checkpoint.pt"

launch_args=(
  "$python_bin" -u gear_sonic/eval_agent_trl.py
  "checkpoint=$snapshot"
  ++headless=true
  "++num_envs=$num_envs"
  ++use_wandb=false
  ++auto_load_latest=false
  ++algo.config.global_rank=0
  ++algo.config.world_size=1
  ++manager_env.config.render_results=true
  "++manager_env.config.save_rendering_dir=$output_dir/mp4"
  manager_env/recorders=render
  "++eval_callbacks=[]"
  ++run_eval_loop=true
  ++run_once=false
  "++max_render_steps=$steps"
  "eval_log_dir=$output_dir/logs"
)
if [[ "$encoder" != mixed ]]; then
  launch_args+=(
    "++use_encoder=$encoder"
    "++manager_env.commands.motion.encoder_sample_probs.$encoder=1.0"
  )
fi

# Allows checking launch arguments without Isaac Lab, files, or CUDA.
if [[ "${WAM_VIDEO_DRY_RUN:-0}" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
  printf '%q ' "${launch_args[@]}"
  printf '\n'
  exit 0
fi

[[ -x "$python_bin" ]] || { echo "Python not executable: $python_bin" >&2; exit 2; }
[[ -f "$source_checkpoint" ]] || { echo "Checkpoint missing: $source_checkpoint" >&2; exit 2; }
source_config="$run_dir/config.yaml"
if [[ ! -f "$source_config" ]]; then
  source_config="$(dirname -- "$run_dir")/config.yaml"
fi
[[ -f "$source_config" ]] || { echo "Training config missing beside $source_checkpoint" >&2; exit 2; }
if [[ -e "$snapshot" || -e "$output_dir/mp4/000000.mp4" ]]; then
  echo "Output already contains a recording. Choose a new output directory: $output_dir" >&2
  exit 2
fi

mkdir -p -- "$output_dir"
# WAM saves last.pt by atomic replacement. Copy once to keep a fixed policy
# even if training continues replacing the source checkpoint during recording.
cp -- "$source_checkpoint" "$snapshot"
cp -- "$source_config" "$output_dir/config.yaml"

export TMPDIR="$(mktemp -d "$HOME/wam-video-tmp.XXXXXX")"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
# This user's Isaac Sim installation has already been used after accepting EULA.
export OMNI_KIT_ACCEPT_EULA=YES
cd -- "$repo_dir"
printf 'Checkpoint snapshot: %s\nVideo directory: %s\nEncoder: %s\nSteps: %s\n' \
  "$snapshot" "$output_dir/mp4" "$encoder" "$steps"
exec "${launch_args[@]}"
