import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Rigid palm blueprint
PALM_3D_MODEL = np.array([
    [ 0.000,  0.000,  0.0],   # Wrist
    [-0.030, -0.080,  0.0],   # Index MCP
    [-0.010, -0.085,  0.0],   # Middle MCP
    [ 0.010, -0.080,  0.0],   # Ring MCP
    [ 0.030, -0.070,  0.0],   # Pinky MCP
], dtype=np.float32)

PALM_LANDMARK_INDICES = [0, 5, 9, 13, 17]


def kmeans_1d(x, n_iter=50):
    lo, hi = np.percentile(x, 10), np.percentile(x, 90)
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        c1 = x[x < mid]
        c2 = x[x >= mid]
        if len(c1) == 0 or len(c2) == 0:
            break
            
        new_lo, new_hi = c1.mean(), c2.mean()
        if abs(new_lo - lo) < 1e-6 and abs(new_hi - hi) < 1e-6:
            break
            
        lo, hi = new_lo, new_hi
        
    return lo, hi


def clip_vector(v, max_norm):
    n = np.linalg.norm(v)
    if n > max_norm and n > 0:
        return v * (max_norm / n)
    return v


class PnPRetargeter:
    def __init__(self, config):
        self.cfg = config
        self.prev_tvec_ema = None
        self.gripper_closed = False
        self.pending_gap = []
        self.last_action = np.zeros(7, dtype=np.float32)

        # Camera intrinsics approx
        f = self.cfg.width
        self.cam_matrix = np.array([
            [f,   0.0, self.cfg.width  / 2.0],
            [0.0, f,   self.cfg.height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        theta = np.deg2rad(self.cfg.camera_pitch_deg)
        self.R_pitch = np.array([
            [1.0, 0.0,            0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta),  np.cos(theta)],
        ])

    def calibrate(self, frames):
        aps = np.array([f["aperture"] for f in frames if f.get("aperture") is not None])
        if len(aps) < 20:
            return None

        low, high = kmeans_1d(aps)
        if high <= low:
            return None

        self.cfg.close_thresh = low + 0.30 * (high - low)
        self.cfg.open_thresh = low + 0.70 * (high - low)

        return {
            "close_thresh": float(self.cfg.close_thresh),
            "open_thresh": float(self.cfg.open_thresh),
            "kmeans_low": float(low),
            "kmeans_high": float(high),
            "n_samples": len(aps),
            "aperture_min": float(aps.min()),
            "aperture_max": float(aps.max()),
            "aperture_mean": float(aps.mean()),
        }

    def reset(self):
        self.prev_tvec_ema = None
        self.gripper_closed = False
        self.pending_gap.clear()
        self.last_action = np.zeros(7, dtype=np.float32)

    def step(self, frame):
        if not frame.get("landmarks"):
            self.pending_gap.append(frame)
            if len(self.pending_gap) > self.cfg.gap_interp_max:
                self.pending_gap.clear()
            return self.last_action.copy(), self.gripper_closed, True

        self.pending_gap.clear()
        landmarks = np.array(frame["landmarks"])

        img_pts = np.array([
            [landmarks[i][0] * self.cfg.width, landmarks[i][1] * self.cfg.height]
            for i in PALM_LANDMARK_INDICES
        ], dtype=np.float32)

        success, _, tvec = cv2.solvePnP(
            PALM_3D_MODEL, img_pts, self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return np.zeros(7, dtype=np.float32), self.gripper_closed, False

        curr_tvec = tvec.flatten()

        # Bootstrap reference on first frame
        if self.prev_tvec_ema is None:
            self.prev_tvec_ema = curr_tvec.copy()
            return np.zeros(7, dtype=np.float32), self.gripper_closed, False

        # EMA smooth
        a = self.cfg.tvec_ema_alpha
        self.prev_tvec_ema = (1.0 - a) * self.prev_tvec_ema + a * curr_tvec

        delta_cam = curr_tvec - self.prev_tvec_ema
        delta_world = self.R_pitch @ delta_cam

        # Map to LIBERO frame: X forward, Y left, Z up.
        dx =  delta_world[1] * self.cfg.scale
        dy = -delta_world[0] * self.cfg.scale
        dz = -delta_world[2] * self.cfg.scale

        dpos = clip_vector(np.array([dx, dy, dz]), self.cfg.max_step_pos)

        # Gripper pinch via thumb/index tips
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch = np.linalg.norm(thumb_tip[:2] - index_tip[:2])

        if pinch < self.cfg.close_thresh:
            self.gripper_closed = True
        elif pinch > self.cfg.open_thresh:
            self.gripper_closed = False

        grip = 1.0 if self.gripper_closed else -1.0
        self.last_action = np.concatenate([dpos, [0.0, 0.0, 0.0], [grip]]).astype(np.float32)
        
        return self.last_action, self.gripper_closed, False


def process_clip(pose_path, out_path, cfg):
    with open(pose_path) as f:
        data = json.load(f)
        
    cfg.width = data.get("width", cfg.width)
    cfg.height = data.get("height", cfg.height)

    frames = data["frames"]
    retargeter = PnPRetargeter(cfg)
    calib = retargeter.calibrate(frames) # use kmeans to calculate thresholds

    actions = []
    for f in frames:
        action, gripper, interp = retargeter.step(f)
        actions.append({
            "frame": f["frame"],
            "time": f["time"],
            "action": action.tolist(),
            "gripper_closed": gripper,
            "interpolated": interp,
        })

    out = {
        "video": data.get("video", "unknown"),
        "fps": data.get("fps", 30),
        "width": cfg.width,
        "height": cfg.height,
        "n_frames": len(actions),
        "n_interpolated": sum(a["interpolated"] for a in actions),
        "aperture_calibration": calib,
        "config": vars(cfg),
        "frames": actions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
        
    return out


def main():
    parser = argparse.ArgumentParser(description="Retarget MediaPipe hand poses to LIBERO actions.")
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
    out_dir = Path(args.out).expanduser()
    pose_files = sorted(pose_dir.filenames(args.filenames))

    all_actions = []
    for p in pose_files:
        out_path = out_dir / p.name.replace("_poses.json", "_actions.json")
        all_actions.append(process_clip(p, out_path, args))

    total = sum(r["n_frames"] for r in all_actions)
    interp = sum(r["n_interpolated"] for r in all_actions)
    
    print(f"\nTotal frames: {total}, Interpolated: {interp} ({(interp/max(total, 1))*100:.1f}%)")

if __name__ == "__main__":
    main()