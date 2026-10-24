#!/usr/bin/env python3
"""Check original SMPL/G1 frame counts without changing or resampling motions.

Only joblib and the dependencies needed to unpickle the input files are required.
No Isaac Lab, CUDA, forward kinematics, or training imports are used.

Examples (run from the repository root)::

    python gear_sonic/data_process/check_motion_frame_alignment.py \
        --dataset-root /data/ouqin/datasets/g1 \
        --output-dir /data/ouqin/frame_alignment_report

    # Check only the smoke split.
    python gear_sonic/data_process/check_motion_frame_alignment.py \
        --dataset-root /data/ouqin/datasets/g1/smoke \
        --output-dir /data/ouqin/frame_alignment_smoke

Exit status: 0 = no findings; 1 = data issues found; 2 = invocation/setup error.
"""

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import ExitStack
import csv
import json
import math
import os
from pathlib import Path

import joblib


SMPL_FIELDS = ("smpl_joints", "pose_aa", "transl")
ROBOT_FIELDS = ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints")
REPORT_FIELDS = (
    "dataset", "sequence", "problems", "smpl_frames", "robot_frames", "delta",
    "smpl_fps", "robot_fps", "smpl_lengths", "robot_lengths",
    "smpl_paths", "robot_paths", "error",
)


def discover_datasets(root):
    """Find sibling smpl/robot directories, including incomplete pairs."""
    result = []
    for current, dirs, _ in os.walk(root, onerror=raise_walk_error):
        dirs.sort()
        if "smpl" in dirs or "robot" in dirs:
            result.append(Path(current))
            # Individual motion subdirectories belong to this dataset, not new splits.
            dirs[:] = [d for d in dirs if d not in ("smpl", "robot")]
    return result


def index_sequences(directory):
    result = defaultdict(list)
    if directory.is_dir():
        for current, dirs, names in os.walk(directory, onerror=raise_walk_error):
            dirs.sort()
            for name in sorted(names):
                path = Path(current) / name
                if name.endswith(".pkl") and name != "metadata.pkl":
                    result[path.stem].append(path)
    return result


def raise_walk_error(error):
    # A partially scanned directory must not be reported as a complete audit.
    raise error


def load_sequence(path, primary):
    data = joblib.load(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a dictionary, got {type(data).__name__}")
    if primary not in data:
        # Robot PKLs normally wrap the data as {sequence_name: fields}.
        if len(data) != 1 or path.stem not in data:
            raise ValueError(f"{path}: expected '{primary}' or a single '{path.stem}' entry")
        data = data[path.stem]
    if not isinstance(data, Mapping) or primary not in data:
        raise ValueError(f"{path}: missing required field '{primary}'")
    if "pose_aa" not in data:
        raise ValueError(f"{path}: missing required field 'pose_aa'")
    return data


def field_lengths(data, fields):
    lengths = {}
    for key in fields:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, (str, bytes, Mapping)):
            raise ValueError(f"'{key}' is not a frame array")
        n = len(value)
        if n <= 0:
            raise ValueError(f"'{key}' contains no frames")
        lengths[key] = n
    # original_pose_aa/original_fps describe the source, not the current time grid.
    return lengths


def read_fps(data):
    value = data.get("fps")
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def write_row(writer, row):
    writer.writerow({
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        for key, value in row.items()
    })


def check_dataset(root, dataset, expected_fps, writers, progress_every):
    label = dataset.relative_to(root).as_posix()
    if label == ".":
        label = dataset.name
    smpl = index_sequences(dataset / "smpl")
    robot = index_sequences(dataset / "robot")
    counts = Counter({
        "datasets": 1,
        "smpl_files": sum(map(len, smpl.values())),
        "robot_files": sum(map(len, robot.values())),
        "paired_names": len(smpl.keys() & robot.keys()),
        "missing_robot": len(smpl.keys() - robot.keys()),
        "missing_smpl": len(robot.keys() - smpl.keys()),
        "checked_pairs": 0, "frame_aligned_pairs": 0, "frame_mismatch_pairs": 0,
        "cross_frame_mismatches": 0, "smpl_internal_mismatches": 0,
        "robot_internal_mismatches": 0, "fps_issue_pairs": 0,
        "ambiguous_pairs": 0, "read_errors": 0,
        "duplicate_smpl_names": sum(len(paths) > 1 for paths in smpl.values()),
        "duplicate_robot_names": sum(len(paths) > 1 for paths in robot.values()),
        "missing_directory_datasets": 0, "empty_datasets": 0,
    })
    delta_counts = Counter()
    fps_counts = {"smpl": Counter(), "robot": Counter()}
    examples = []
    names = sorted(smpl.keys() | robot.keys())
    if not (dataset / "smpl").is_dir() or not (dataset / "robot").is_dir():
        counts["missing_directory_datasets"] = 1
    if not names:
        counts["empty_datasets"] = 1
    if counts["missing_directory_datasets"] or not names:
        write_row(writers["issues"], {
            "dataset": label,
            "problems": ["missing_directory"] if names else ["empty_or_missing_directories"],
            "smpl_paths": [str(dataset / "smpl")],
            "robot_paths": [str(dataset / "robot")],
        })

    print(f"\n[{label}] SMPL={counts['smpl_files']}, G1={counts['robot_files']}", flush=True)
    for i, name in enumerate(names, 1):
        sp, rp = smpl.get(name, []), robot.get(name, [])
        problems = []
        row = {
            "dataset": label, "sequence": name, "problems": problems,
            "smpl_paths": [str(p) for p in sp], "robot_paths": [str(p) for p in rp],
        }
        if not sp:
            problems.append("missing_smpl")
        if not rp:
            problems.append("missing_robot")
        if len(sp) > 1:
            problems.append("duplicate_smpl_name")
        if len(rp) > 1:
            problems.append("duplicate_robot_name")

        if len(sp) == 1 and len(rp) == 1:
            try:
                sdata = load_sequence(sp[0], "smpl_joints")
                rdata = load_sequence(rp[0], "root_trans_offset")
                sl = field_lengths(sdata, SMPL_FIELDS)
                rl = field_lengths(rdata, ROBOT_FIELDS)
                sn, rn = sl["smpl_joints"], rl["root_trans_offset"]
                sf, rf = read_fps(sdata), read_fps(rdata)
                row.update({
                    "smpl_frames": sn, "robot_frames": rn, "delta": sn - rn,
                    "smpl_fps": sf, "robot_fps": rf,
                    "smpl_lengths": sl, "robot_lengths": rl,
                })
                counts["checked_pairs"] += 1
                delta_counts[sn - rn] += 1
                fps_counts["smpl"][str(sf)] += 1
                fps_counts["robot"][str(rf)] += 1

                cross = sn != rn
                sinternal = len(set(sl.values())) > 1
                rinternal = len(set(rl.values())) > 1
                counts["cross_frame_mismatches"] += cross
                counts["smpl_internal_mismatches"] += sinternal
                counts["robot_internal_mismatches"] += rinternal
                if cross:
                    problems.append("cross_frame_mismatch")
                if sinternal:
                    problems.append("smpl_internal_mismatch")
                if rinternal:
                    problems.append("robot_internal_mismatch")

                if sf is None or rf is None:
                    problems.append("missing_or_invalid_fps")
                if sf is not None and rf is not None and not math.isclose(sf, rf):
                    problems.append("fps_different")
                if any(f is not None and not math.isclose(f, expected_fps) for f in (sf, rf)):
                    problems.append("fps_not_expected")
                if any("fps" in p for p in problems):
                    counts["fps_issue_pairs"] += 1

                if cross or sinternal or rinternal:
                    counts["frame_mismatch_pairs"] += 1
                    write_row(writers["mismatches"], row)
                    if len(examples) < 10:
                        examples.append(f"{name}: SMPL={sl}, G1={rl}")
                else:
                    counts["frame_aligned_pairs"] += 1
            except Exception as exc:
                counts["read_errors"] += 1
                problems.append("read_or_schema_error")
                row["error"] = f"{type(exc).__name__}: {exc}"
        elif sp and rp:
            # Never silently overwrite duplicate basenames or pick the first file.
            counts["ambiguous_pairs"] += 1

        write_row(writers["all"], row)
        if problems:
            write_row(writers["issues"], row)
        if progress_every and i % progress_every == 0:
            print(f"[{label}] 已检查 {i}/{len(names)} 个序列名", flush=True)

    print(
        f"[{label}] 成功检查={counts['checked_pairs']}, "
        f"帧数不对齐={counts['frame_mismatch_pairs']}, "
        f"读取/格式错误={counts['read_errors']}, 配对歧义={counts['ambiguous_pairs']}",
        flush=True,
    )
    for example in examples:
        print(f"  {example}")
    if counts["paired_names"] and not counts["checked_pairs"]:
        print(f"[{label}] 没有成功检查的配对，不能把不对齐数量 0 当成数据正常。")
    return counts, {
        "path": str(dataset), "counts": dict(counts),
        "frame_delta_distribution": dict(sorted(delta_counts.items())),
        "fps_distribution": {key: dict(value) for key, value in fps_counts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-fps", type=float, default=30, help="Metadata FPS to check; never resamples")
    parser.add_argument("--progress-every", type=int, default=1000, help="Progress interval; 0 disables progress")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset root does not exist: {root}")
    if not math.isfinite(args.expected_fps) or args.expected_fps <= 0:
        parser.error("--expected-fps must be a positive finite number")
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative")
    datasets = discover_datasets(root)
    if not datasets:
        parser.error(f"No smpl/robot directories found under {root}")
    output = args.output_dir.expanduser().resolve()
    # Avoid accidentally using a source motion file/directory as a report target.
    source_dirs = {(d / side).resolve() for d in datasets for side in ("smpl", "robot")}
    if any(output == source or source in output.parents for source in source_dirs):
        parser.error("Reports must be saved outside source smpl/robot directories")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": output / "all_pairs.csv", "mismatches": output / "frame_mismatches.csv",
        "issues": output / "issues.csv",
    }
    summary_path = output / "summary.json"
    for path in (*paths.values(), summary_path):
        if path.is_symlink() or (path.exists() and path.stat().st_nlink > 1):
            parser.error(f"Refusing to overwrite a linked report file: {path}")
    total = Counter()
    summaries = []
    with ExitStack() as stack:
        writers = {}
        for key, path in paths.items():
            handle = stack.enter_context(path.open("w", newline="", encoding="utf-8"))
            writers[key] = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
            writers[key].writeheader()
        for dataset in datasets:
            counts, summary = check_dataset(
                root, dataset, args.expected_fps, writers, args.progress_every
            )
            total.update(counts)
            summaries.append(summary)

    summary_path.write_text(json.dumps({
        "mode": "original_frames_no_resampling", "dataset_root": str(root),
        "expected_fps": args.expected_fps, "total": dict(total), "datasets": summaries,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n========== 总统计：原始帧数，不进行重采样 ==========")
    labels = {
        "datasets": "数据集/划分数量", "smpl_files": "SMPL 文件数",
        "robot_files": "G1 文件数", "paired_names": "同名配对数",
        "checked_pairs": "成功检查配对数", "frame_aligned_pairs": "所有已检查字段帧数一致",
        "frame_mismatch_pairs": "帧数不对齐序列数（包含内部字段不一致，去重）",
        "cross_frame_mismatches": "SMPL 与 G1 主字段帧数不同",
        "smpl_internal_mismatches": "SMPL 内部字段帧数不一致",
        "robot_internal_mismatches": "G1 内部字段帧数不一致",
        "fps_issue_pairs": "帧率异常配对数", "missing_robot": "缺少 G1 配对",
        "missing_smpl": "缺少 SMPL 配对", "ambiguous_pairs": "重名导致无法确定配对",
        "read_errors": "读取或格式错误", "empty_datasets": "空数据集",
        "missing_directory_datasets": "缺少整侧目录的数据集",
    }
    for key, label in labels.items():
        print(f"{label}: {total[key]}")
    print(f"帧数不对齐详细列表: {paths['mismatches']}")
    print(f"所有异常（含缺失/重名/错误/帧率）: {paths['issues']}")
    print(f"完整配对结果: {paths['all']}")
    print(f"汇总: {summary_path}")
    issue_keys = (
        "frame_mismatch_pairs", "fps_issue_pairs", "missing_robot", "missing_smpl",
        "ambiguous_pairs", "read_errors", "empty_datasets", "missing_directory_datasets",
    )
    return int(any(total[key] for key in issue_keys))


if __name__ == "__main__":
    raise SystemExit(main())
