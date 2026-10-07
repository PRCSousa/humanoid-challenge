import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

INSTRUCTIONS = [
    "move left",
    "move right",
    "move forward",
    "move backward",
    "clockwise",
    "counterclockwise",
    "wave",
    "clap",
    "hold still",
]

def make_env():
    task_suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = task_suite.get_task(0)
    task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
    return env, task_suite

def build_batch(obs, instruction, device):
    av = np.asarray(obs["agentview_image"], dtype=np.uint8)
    wr = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)

    av_t = torch.from_numpy(av).permute(2, 0, 1).float() / 255.0
    wr_t = torch.from_numpy(wr).permute(2, 0, 1).float() / 255.0

    if tuple(av_t.shape[-2:]) != (256, 256):
        av_t = torch.nn.functional.interpolate(av_t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]
    if tuple(wr_t.shape[-2:]) != (256, 256):
        wr_t = torch.nn.functional.interpolate(wr_t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)[0]

    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    try:
        axang = Rotation.from_quat(quat).as_rotvec().astype(np.float32)
    except ValueError:
        axang = np.zeros(3, dtype=np.float32)
        
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    state = np.concatenate([pos, axang, grip]).astype(np.float32)

    return {
        "observation.images.image": av_t.unsqueeze(0).to(device),
        "observation.images.image2": wr_t.unsqueeze(0).to(device),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": [instruction],
    }

def count_sign_changes(arr, min_amp=0.002):
    if len(arr) < 3: return 0
    d = np.diff(arr)
    d = d[np.abs(d) > min_amp]
    if len(d) < 2: return 0
    signs = np.sign(d)
    return int(np.sum(signs[1:] != signs[:-1]))

def trajectory_features(positions, grips):
    if len(positions) < 2:
        return {"path_length": 0.0, "range_x": 0.0, "range_y": 0.0, "range_z": 0.0, "grip_range": 0.0, "sc_x": 0, "sc_y": 0, "sc_z": 0, "sc_grip": 0}

    ranges = positions.max(axis=0) - positions.min(axis=0)
    return {
        "path_length": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "range_x": float(ranges[0]),
        "range_y": float(ranges[1]),
        "range_z": float(ranges[2]),
        "grip_range": float(grips.max() - grips.min()),
        "sc_x": count_sign_changes(positions[:, 0]),
        "sc_y": count_sign_changes(positions[:, 1]),
        "sc_z": count_sign_changes(positions[:, 2]),
        "sc_grip": count_sign_changes(grips, min_amp=0.001),
    }

def run_rollout(policy, env, init_state, instruction, n_steps, device, video_writer=None, pre=None, post=None):
    env.reset()
    if init_state is not None:
        env.set_init_state(init_state)
    obs = env.reset()

    start_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy()
    start_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy()

    if hasattr(policy, "reset"):
        policy.reset()

    pos_traj = [start_pos.copy()]
    q0 = np.asarray(obs["robot0_gripper_qpos"])
    grip_traj = [float(abs(q0[0] - q0[1]))]

    for _ in range(n_steps):
        batch = build_batch(obs, instruction, device)
        if pre: batch = pre(batch)

        with torch.no_grad():
            action = policy.select_action(batch).squeeze(0).detach().cpu().numpy()

        if post: 
            action = post({"action": torch.from_numpy(action).unsqueeze(0)})["action"].squeeze(0).cpu().numpy()

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs, _, _, _ = env.step(action)

        pos_traj.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        q = np.asarray(obs["robot0_gripper_qpos"])
        grip_traj.append(float(abs(q[0] - q[1])))

        if video_writer is not None:
            frame_bgr = cv2.cvtColor(obs["agentview_image"], cv2.COLOR_RGB2BGR)
            if frame_bgr.shape[:2] != (256, 256):
                frame_bgr = cv2.resize(frame_bgr, (256, 256))
            video_writer.write(frame_bgr)

    end_pos = pos_traj[-1]
    try:
        R0 = Rotation.from_quat(start_quat).as_matrix()
        R1 = Rotation.from_quat(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)).as_matrix()
        net_rot = Rotation.from_matrix(R1 @ R0.T).as_rotvec().tolist()
    except Exception:
        net_rot = [0.0, 0.0, 0.0]

    pos_arr = np.stack(pos_traj, axis=0)
    grip_arr = np.array(grip_traj)

    return {
        "net_pos": (end_pos - start_pos).tolist(),
        "net_rot": net_rot,
        "net_grip": float(grip_arr[-1] - grip_arr[0]),
        **trajectory_features(pos_arr, grip_arr),
    }

def load_policy(policy_path, device):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    path = Path(policy_path)
    
    if path.is_dir() and (path / "adapter_config.json").exists() and not (path / "model.safetensors").exists():
        from peft import PeftModel
        print(f"Loading base model and applying LoRA from {path}")
        policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero")
        policy = PeftModel.from_pretrained(policy, str(path))
        try:
            return policy.merge_and_unload()
        except Exception:
            return policy
            
    return SmolVLAPolicy.from_pretrained(str(policy_path))

def load_processors(policy_path):
    try:
        from lerobot.processor import PolicyProcessorPipeline
    except ImportError:
        return None, None

    for ref in [str(policy_path), "lerobot/smolvla_base"]:
        try:
            pre = PolicyProcessorPipeline.from_pretrained(ref, config_filename="policy_preprocessor.json")
            post = PolicyProcessorPipeline.from_pretrained(ref, config_filename="policy_postprocessor.json")
            return pre, post
        except Exception:
            continue
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--n_steps", type=int, default=100)
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default="outputs/primitive_eval.json")
    ap.add_argument("--video_dir", default="outputs/primitive_videos")
    ap.add_argument("--no_videos", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    policy = load_policy(args.policy_path, device).to(device).eval()
    pre, post = load_processors(args.policy_path)

    env, task_suite = make_env()
    fixed_init = task_suite.get_task_init_states(0)[0]

    video_writers = {}
    if not args.no_videos:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for inst in INSTRUCTIONS:
            video_writers[inst] = cv2.VideoWriter(str(Path(args.video_dir) / f"{inst.replace(' ', '_')}.mp4"), fourcc, 20, (256, 256))

    results = {}
    
    # Markdown table header
    print("\n| instruction      | Δx          | Δy          | Δz          | abs. path   | rY       | rG       | sX | sY | sZ | sG |")
    print("| ---------------- | ----------- | ----------- | ----------- | ----------- | -------- | -------- | -- | -- | -- | -- |")

    for inst in INSTRUCTIONS:
        stats = []
        for rep in range(args.n_repeats):
            writer = video_writers.get(inst) if rep == 0 else None
            stats.append(run_rollout(policy, env, fixed_init, inst, args.n_steps, device, writer, pre, post))

        agg = {k: np.mean([s[k] for s in stats], axis=0).tolist() if isinstance(stats[0][k], list) 
               else float(np.mean([s[k] for s in stats])) for k in stats[0]}
        results[inst] = agg

        # Format metrics
        dx = f"{agg['net_pos'][0]:+8.4f}".strip()
        dy = f"{agg['net_pos'][1]:+8.4f}".strip()
        dz = f"{agg['net_pos'][2]:+8.4f}".strip()
        path = f"{agg['path_length']:8.4f}".strip()
        ry = f"{agg['range_y']:6.4f}".strip()
        rg = f"{agg['grip_range']:6.4f}".strip()
        sx = f"{int(agg['sc_x'])}"
        sy = f"{int(agg['sc_y'])}"
        sz = f"{int(agg['sc_z'])}"
        sg = f"{int(agg['sc_grip'])}"

        # Apply bolding to relevant metrics based on instruction
        if inst in ["move forward", "move backward"]:
            dx = f"**{dx}**"
        elif inst in ["move left", "move right"]:
            dy = f"**{dy}**"
        elif inst in ["wave"]:
            sy = f"**{sy}**"
        elif inst in ["clap"]:
            sg = f"**{sg}**"
        elif inst in ["hold still"]:
            path = f"**{path}**"
        elif "clockwise" in inst:
            sx = f"**{sx}**"
            sy = f"**{sy}**"

        print(f"| {inst:<16} | {dx:<11} | {dy:<11} | {dz:<11} | {path:<11} | {ry:<8} | {rg:<8} | {sx:<2} | {sy:<2} | {sz:<2} | {sg:<2} |")

    env.close()
    for w in video_writers.values(): w.release()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()