import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

sys.path.insert(0, str(Path(__file__).parent))
from retarget import PnPRetargeter

LIBERO_FPS = 20
DEFAULT_IMAGE_SIZE = 256
VIDEO_TARGET_H = 360
DEFAULT_TASK_STRING = "pick up the alphabet soup and place it in the basket"

# [dx, dy, dz, drx, dry, drz, gripper]
ACTION_MASKS = {
    "move left":        [0, 1, 0, 0, 0, 0, 0],
    "move right":       [0, 1, 0, 0, 0, 0, 0],
    "move forward":     [1, 0, 0, 0, 0, 0, 0],
    "move backward":    [1, 0, 0, 0, 0, 0, 0],
    "move up":          [0, 0, 1, 0, 0, 0, 0],
    "move down":        [0, 0, 1, 0, 0, 0, 0],
    "clockwise":        [1, 1, 0, 0, 0, 0, 0],
    "counterclockwise": [1, 1, 0, 0, 0, 0, 0],
    "wave":             [1, 1, 0, 0, 0, 0, 0],
    "clap":             [0, 0, 0, 0, 0, 0, 1],
    "hold still":       [0, 0, 0, 0, 0, 0, 0],
}


def apply_mask(action, instruction):
    mask = ACTION_MASKS.get(instruction)
    if not mask:
        return action
    return action * np.array(mask, dtype=np.float32)


def stem_to_instruction(stem):
    base = re.sub(r"_poses$", "", stem)
    base = re.sub(r"_\d+$", "", base)
    text = base.replace("_", " ").strip().lower()
    return DEFAULT_TASK_STRING if text.startswith("clip") else text


def resample_poses(frames, src_fps, dst_fps):
    if abs(src_fps - dst_fps) < 1e-6:
        return frames
        
    ratio = src_fps / dst_fps
    n_dst = int(round(len(frames) / ratio))
    out = []
    
    for i in range(n_dst):
        src_idx = min(int(round(i * ratio)), len(frames) - 1)
        src = frames[src_idx]
        out.append({
            "frame": i,
            "time": i / dst_fps,
            "wrist_pos": src.get("wrist_pos"),
            "orientation_matrix": src.get("orientation_matrix"),
            "aperture": src.get("aperture"),
            "landmarks": src.get("landmarks"),
        })
    return out


def to_libero_action(raw_delta):
    constr = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0], dtype=np.float32)
    a = raw_delta.astype(np.float32).copy()
    a[:3] /= constr[:3] # x y z
    a[3:6] /= constr[3:6] # rx ry rz and gripper
    return np.clip(a, -1.0, 1.0)

# calculate axis-angle representation from quaternion (xyzw) (check if clockwise movement works)
def quat_to_axis_angle(quat_xyzw):
    try:
        return Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    except ValueError:
        return np.zeros(3, dtype=np.float32)


def build_state(obs):
    pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
    quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
    axang = quat_to_axis_angle(quat)
    grip = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)
    return np.concatenate([pos, axang, grip])


def resize_square(img, size):
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def resize_to_height(img, target_h):
    ih, iw = img.shape[:2]
    if ih == target_h:
        return img
    scale = target_h / ih
    return cv2.resize(img, (int(iw * scale), target_h), interpolation=cv2.INTER_AREA)


def pad_width(img, target_w):
    ih, iw = img.shape[:2]
    if iw >= target_w:
        return img
    pad = np.zeros((ih, target_w - iw, 3), dtype=img.dtype)
    return np.concatenate([img, pad], axis=1)


def find_source_video(stem, raw_dir):
    for ext in (".mp4"):
        p = raw_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def write_libero_video(agentview_list, wrist_list, out_path, fps=LIBERO_FPS):
    if not agentview_list:
        return
        
    H, W = agentview_list[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 2, H))
    
    for av, wr in zip(agentview_list, wrist_list):
        a = cv2.cvtColor(av, cv2.COLOR_RGB2BGR)
        w = cv2.cvtColor(wr, cv2.COLOR_RGB2BGR)
        writer.write(np.concatenate([a, w], axis=1))
        
    writer.release()

def write_compare_video(agentview_list, source_video, out_path, fps=LIBERO_FPS):
    if not agentview_list:
        return
        
    cap = cv2.VideoCapture(str(source_video))
        
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    advance = max(1, int(round(src_fps / fps)))

    ret, first_src = cap.read()
    if not ret:
        cap.release()
        return

    first_av = cv2.cvtColor(agentview_list[0], cv2.COLOR_RGB2BGR)
    left = resize_to_height(first_src, VIDEO_TARGET_H)
    right = resize_to_height(first_av, VIDEO_TARGET_H)
    
    h = min(left.shape[0], right.shape[0])
    left, right = left[:h], right[:h]
    
    target_w = max(left.shape[1], right.shape[1])
    left = pad_width(left, target_w)
    right = pad_width(right, target_w)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        fps, 
        (left.shape[1] * 2, left.shape[0])
    )
    writer.write(np.concatenate([left, right], axis=1))

    for t in range(1, len(agentview_list)):
        src_frame = None
        for _ in range(advance):
            ret, s = cap.read()
            if not ret:
                break
            src_frame = s
            
        if src_frame is None:
            writer.write(np.concatenate([left, right], axis=1))
            continue
            
        a = cv2.cvtColor(agentview_list[t], cv2.COLOR_RGB2BGR)
        left = resize_to_height(src_frame, VIDEO_TARGET_H)
        right = resize_to_height(a, VIDEO_TARGET_H)
        
        h = min(left.shape[0], right.shape[0])
        left, right = left[:h], right[:h]
        left = pad_width(left, target_w)
        right = pad_width(right, target_w)
        
        writer.write(np.concatenate([left, right], axis=1))

    cap.release()
    writer.release()


def replay_episode(pose_data, task_bddl, init_state, config, camera_size, task_language, max_steps):
    src_fps = pose_data.get("fps", 30.0)
    frames = resample_poses(pose_data["frames"], src_fps, LIBERO_FPS)

    config.width = pose_data.get("width", config.width)
    config.height = pose_data.get("height", config.height)

    retargeter = PnPRetargeter(config)
    calib = retargeter.calibrate(frames)

    raw_actions = []
    n_interp = 0
    for f in frames:
        action, _, interp = retargeter.step(f)
        raw_actions.append(action)
        if interp:
            n_interp += 1

    # Apply per-instruction mask
    raw_actions = [apply_mask(a, task_language) for a in raw_actions]
    n_grip_closed = sum(1 for a in raw_actions if a[6] > 0)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=camera_size,
        camera_widths=camera_size,
    )
    
    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
    obs = env.reset()

    agentview_list, wrist_list = [], []
    state_list, action_list = [], []
    reward_list, done_list = [], []
    eef_list = []

    total_reward = 0.0
    for i, raw_action in enumerate(raw_actions):
        if i >= max_steps:
            break

        action = to_libero_action(raw_action)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        agentview_list.append(resize_square(obs["agentview_image"], camera_size))
        wrist_list.append(resize_square(obs["robot0_eye_in_hand_image"], camera_size))
        state_list.append(build_state(obs))
        action_list.append(action)
        reward_list.append(float(reward))
        done_list.append(bool(done))
        eef_list.append(np.array(obs["robot0_eef_pos"], dtype=np.float32))

        if (i + 1) % 50 == 0:
            eef = obs["robot0_eef_pos"]
            print(f" -> step {i+1}/{len(raw_actions)} eef=({eef[0]:+.2f},{eef[1]:+.2f},{eef[2]:+.2f}) rew={reward:+.2f}")

        if done:
            print(f" -> DONE at step {i+1} (success)")
            break

    env.close()

    return {
        "agentview": np.array(agentview_list, dtype=np.uint8),
        "wrist":     np.array(wrist_list, dtype=np.uint8),
        "state":     np.array(state_list, dtype=np.float32),
        "action":    np.array(action_list, dtype=np.float32),
        "reward":    np.array(reward_list, dtype=np.float32),
        "done":      np.array(done_list, dtype=bool),
        "eef_pos":   np.array(eef_list, dtype=np.float32),
        "task":      task_language,
        "fps":       LIBERO_FPS,
        "success":   bool(np.any(done_list)),
        "total_reward": float(total_reward),
        "n_frames":  len(action_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Replay retargeted actions in LIBERO.")
    
    # Replay configs
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--episodes_out", default="data/episodes")
    parser.add_argument("--videos_out", default="outputs/videos")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--init_state_id", type=int, default=0)
    parser.add_argument("--camera_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--no_libero_video", action="store_true")
    parser.add_argument("--no_compare_video", action="store_true")
    
    # Retarget configs
    parser.add_argument("--poses", default="data/poses")
    parser.add_argument("--out", default="data/actions")
    parser.add_argument("--filenames", default="*_poses.json")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera_pitch_deg", type=float, default=40.0)
    parser.add_argument("--max_step_pos", type=float, default=0.05)
    parser.add_argument("--tvec_ema_alpha", type=float, default=0.3)
    parser.add_argument("--close_thresh", type=float, default=0.04)
    parser.add_argument("--open_thresh", type=float, default=0.07)
    parser.add_argument("--gap_interp_max", type=int, default=6)

    args = parser.parse_args()

    pose_dir = Path(args.poses).expanduser()
    raw_dir = Path(args.raw).expanduser()
    ep_dir = Path(args.episodes_out).expanduser()
    vid_dir = Path(args.videos_out).expanduser()

    pose_files = sorted(pose_dir.glob(args.filenames))

    if not pose_files:
        print(f"No files matched {args.filenames} in {pose_dir}")
        return

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_object"]()
    task = task_suite.get_task(args.task_id)
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    
    init_states = task_suite.get_task_init_states(args.task_id)
    init_state = init_states[args.init_state_id] if init_states is not None else None

    summaries = []
    for p in pose_files:
        stem = p.name.replace("_poses.json", "")
        instruction = stem_to_instruction(stem)
        
        print(f"\nProcessing {p.name}, instruction:'{instruction}'")

        with open(p) as f:
            pose_data = json.load(f)
            
        episode = replay_episode(
            pose_data=pose_data,
            task_bddl=task_bddl,
            init_state=init_state,
            config=args,
            camera_size=args.camera_size,
            task_language=instruction,
            max_steps=args.max_steps,
        )
        
        npz_path = ep_dir / f"{stem}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **episode)
        
        if not args.no_libero_video:
            write_libero_video(list(episode["agentview"]), list(episode["wrist"]), vid_dir / f"{stem}_libero.mp4")

        if not args.no_compare_video:
            source = find_source_video(stem, raw_dir)
            if source:
                write_compare_video(list(episode["agentview"]), source, vid_dir / f"{stem}_compare.mp4")

        summaries.append({
            "clip": p.name,
            "instruction": instruction,
            "n_frames": episode["n_frames"],
            "success": episode["success"],
            "total_reward": episode["total_reward"],
        })


if __name__ == "__main__":
    main()